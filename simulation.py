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
side = (outerBucket_radius - innerBucket_radius-0.005)
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

lid = 0.05
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

bottom = 0.05
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
    A = 2.3*(
        2 * sp.pi * outerBucket_radius**2
        + 2 * sp.pi * outerBucket_radius * outerBucket_height
    ).evalf()
    return 1 / (h * A)


def resistance_radiation(t1, t2):
    sigma = 5.67e-8
    emissivity = 0.03
    A = 2.3*(
        2 * sp.pi * outerBucket_radius**2
        + 2 * sp.pi * outerBucket_radius * outerBucket_height
    ).evalf()
    return (1 / (A * sigma * emissivity * (t1**2 + t2**2) * (t1 + t2)))


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
        1 / resistance_convection(tHigh - tLow) + 1 / resistance_radiation(tHigh+273.15, tLow+273.15)
    )

    return resistance_conduction + resistance_outer


# %% Differential equation
def heatTransfer(Tin, Tmid, Tout):
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
        return 0, 0

    resistance_conduction = (
        1
        / (
            1 / resistance_total_bottom
            + 1 / resistance_total_lid
            + 1 / resistance_total_side
        ).evalf()
    )
    
    resistance_outer = 1 / (
        1 / resistance_convection(Tmid - Tout) + 1 / resistance_radiation(Tmid+273.15, Tout+273.15)
    )

    qx1 = (Tmid - Tin) / resistance_conduction
    qx2 = (Tout - Tmid) / resistance_outer
    
    dT1 = qx1 / cWater
    dT2 = (-qx1 + qx2) / cAlum
    
    return dT1, dT2


def warmUp(U, I):
    q = U * I
    dT = q / cWater
    return dT


def warmUp_heatTransfer(Tin, Tmid, Tout, U, I):
    dT1, dT2 = heatTransfer(Tin, Tmid, Tout)
    dT1 = dT1  + warmUp(U, I)
    return dT1, dT2


def eulersMethod(de, x0: float, y0: float, yy0: float, h: float, xn: float, *args):
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
    yy = np.zeros(len(x))

    y[0] = y0
    yy[0] = yy0

    for i in range(1, len(x)):
        dT1, dT2 = de(y[i - 1], yy[i-1], *args)
        y[i] = y[i - 1] + dT1 * h
        yy[i] = yy[i - 1] + dT2 * h

    return x, y, yy


# %% Main code

# Thermal resistance and heat capacity for simulation'

mass_water = 3.036  # kg
cWater = specific_heat_water * mass_water
cAlum = 900 * 0.02

# Simulation
start_time = 0
step_size = 10  # s
end_time = 60 * 60 * 24 * 5 # s
outside_temperature = 24.5  # C
start_temperature = 24.7  # C
voltage = 27.5  # V
amperage = 5.9  # A
switchTime = 35 * 60  # s


t, T, TT = eulersMethod(
    warmUp_heatTransfer,
    start_time,
    start_temperature,
    outside_temperature,
    step_size,
    switchTime,
    outside_temperature,
    voltage,
    amperage,
)

t2, T2, TT2 = eulersMethod(
    heatTransfer,
    t[-1],
    T[-1],
    TT[-1],
    step_size,
    end_time,
    outside_temperature,
)

t = np.concatenate((t, t2), axis=0)
T = np.concatenate((T, T2), axis=0)
TT = np.concatenate((TT, TT2), axis=0)

print("Simulation complete!")

# %% Export data
df_sim = pd.DataFrame({"t": t, "T": T, "TT": TT})
df_sim.set_index("t").to_csv("sim.csv")

df_param = pd.DataFrame(columns=["param", "val"])
df_param.loc[len(df_param)] = ["start_time", start_time]
df_param.loc[len(df_param)] = ["start_temperature", start_temperature]
df_param.loc[len(df_param)] = ["step_size", step_size]
df_param.loc[len(df_param)] = ["end_time", end_time]
df_param.loc[len(df_param)] = ["outside_temperature", outside_temperature]
df_param.set_index("param").to_csv("param.csv")

# %%
