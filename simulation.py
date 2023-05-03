# %% Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

# %% Load data
df_vars = pd.read_csv("physicalData.csv")
df_vars = df_vars.set_index("var")

innerBucket_height = df_vars.loc["innerBucket_height"]["val"]
outerBucket_height = df_vars.loc["outerBucket_height"]["val"]
conduction_rockwool = df_vars.loc["conduction_rockwool"]["val"]
conduction_glasswool = df_vars.loc["conduction_glasswool"]["val"]
conduction_PP = df_vars.loc["conduction_PP"]["val"]
convection_lid = df_vars.loc["convection_lid"]["val"]
convection_side = df_vars.loc["convection_side"]["val"]
innerBucket_radius = df_vars.loc["innerBucket_radius"]["val"]
outerBucket_radius = df_vars.loc["outerBucket_radius"]["val"]
convection_bottom = df_vars.loc["convection_bottom"]["val"]
specific_heat_water = df_vars.loc["specific_heat_water"]["val"]
width_bucket = df_vars.loc["width_bucket"]["val"]

# %% Resistance equation
side = outerBucket_radius - innerBucket_radius
resistance_conduction_side_RW = sp.ln(
    (innerBucket_radius + side) / innerBucket_radius
) / (2 * sp.pi * innerBucket_height * conduction_rockwool)

resistance_conduction_side_B1 = sp.ln(
    (innerBucket_radius) / innerBucket_radius - width_bucket
) / (2 * sp.pi * innerBucket_height * conduction_PP)

resistance_conduction_side_B2 = sp.ln(
    (outerBucket_radius) / innerBucket_radius - width_bucket
) / (2 * sp.pi * outerBucket_height * conduction_PP)

resistance_total_side = (
    resistance_conduction_side_RW
    + resistance_conduction_side_B1
    + resistance_conduction_side_B2
)

lid = 0.051
resistance_conduction_lid_RW = lid / (
    conduction_rockwool * sp.pi * outerBucket_radius**2
)

resistance_conduction_lid_B1 = width_bucket / (
    conduction_PP * sp.pi * innerBucket_radius**2
)

resistance_conduction_lid_B2 = width_bucket / (
    conduction_PP * sp.pi * outerBucket_radius**2
)

resistance_total_lid = (
    resistance_conduction_lid_RW
    + resistance_conduction_lid_B1
    + resistance_conduction_lid_B2
)

bottom = 0.051
resistance_conduction_bottom_RW = bottom / (
    conduction_rockwool * sp.pi * outerBucket_radius**2
)

resistance_conduction_bottom_B1 = width_bucket / (
    conduction_PP * sp.pi * innerBucket_radius**2
)

resistance_conduction_bottom_B2 = width_bucket / (
    conduction_PP * sp.pi * outerBucket_radius**2
)

resistance_total_bottom = (
    resistance_conduction_bottom_RW
    + resistance_conduction_bottom_B1
    + resistance_conduction_bottom_B2
)


def resistance_convection(deltaT):
    h = 1.42 * ((deltaT / (outerBucket_height)) ** 0.25)
    A = (
        2 * sp.pi * outerBucket_radius**2
        + 2 * sp.pi * outerBucket_radius * outerBucket_height
    ).evalf()
    return 1 / (h * A)


def resistance_radiation(t1, t2):
    sigma = 5.67e-8
    emissivity = 0.03
    return 1 / (sigma * emissivity * (t1**2 + t2**2) * (t1 + t2))


def resistance_total(tHigh, tLow):
    resistance_conduction = (
        1
        / (
            1 / resistance_total_bottom
            + 1 / resistance_total_lid
            + 1 / resistance_total_side
        ).evalf()
    )

    resistance_outer = 1 / (
        1 / resistance_convection(tHigh - tLow) + 1 / resistance_radiation(tHigh, tLow)
    )

    return resistance_conduction + resistance_outer


# %% Differential equation
def heatTransfer(Tin, Tout, C):
    """
    Parameters
    ----------
    Tin : Temperature inside storage tank [K]
    Tout : Temperature outside storage tank [K]
    R : Total thermal resistance [K/W]
    C : Total heat capacity of system

    Returns
    --------
    qx : Heat flow [W]
    """
    if Tin == Tout:
        return 0

    qx = (Tout - Tin) / resistance_total(Tin, Tout)
    dT = qx / C
    return dT


def warmUp(U, I, C):
    q = U * I
    dT = q / C
    return dT


def warmUp_heatTransfer(Tin, Tout, U, I, C):
    return heatTransfer(Tin, Tout, C) + warmUp(U, I, C)


def eulersMethod(de, x0: float, y0: float, h: float, xn: float, *args):
    """
    Parameters
    ----------
    de : Differential equation returning dx/dy
    x0 : Initial x-value
    y0 : Initial y-value
    h : Simulation step size
    xn : Final x-value defining end of simulation
    *args : Function arguments to be passed to de

    Returns
    --------
    x : Array of x-values
    y : Array of y-values
    """
    x = np.arange(x0, xn, h)
    y = np.zeros(len(x))

    y[0] = y0

    for i in range(1, len(x)):
        y[i] = y[i - 1] + de(y[i - 1], *args) * h

    return x, y


# %% Main code

# Thermal resistance and heat capacity for simulation'

mass_water = 3.21  # kg
heat_capacity = specific_heat_water * mass_water

# Simulation
start_time = 0
step_size = 120  # s
end_time = 60 * 60 * 72  # s
outside_temperature = 23.4  # C
start_temperature = outside_temperature  # C
voltage = 30  # V
amperage = 6.2  # A
switchTime = 47.5 * 60  # s

t, T = eulersMethod(
    warmUp_heatTransfer,
    start_time,
    start_temperature,
    step_size,
    switchTime,
    outside_temperature,
    voltage,
    amperage,
    heat_capacity,
)

t2, T2 = eulersMethod(
    heatTransfer,
    t[-1],
    T[-1],
    step_size,
    end_time,
    outside_temperature,
    heat_capacity,
)

t = np.concatenate((t, t2), axis=0)
T = np.concatenate((T, T2), axis=0)

print("Simulation complete!")

# %% Export data
df_sim = pd.DataFrame({"t": t, "T": T})
df_sim.set_index("t").to_csv("sim.csv")

df_param = pd.DataFrame(columns=["param", "val"])
df_param.loc[len(df_param)] = ["start_time", start_time]
df_param.loc[len(df_param)] = ["start_temperature", start_temperature]
df_param.loc[len(df_param)] = ["step_size", step_size]
df_param.loc[len(df_param)] = ["end_time", end_time]
df_param.loc[len(df_param)] = ["outside_temperature", outside_temperature]
df_param.set_index("param").to_csv("param.csv")

# %%
