# %% Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

# %% Load data
df_vars = pd.read_csv("physicalData.csv")
df_vars = df_vars.set_index("var")

innerBucket_height = df_vars.loc["innerBucket_height"]["val"]
conduction_rockwool = df_vars.loc["conduction_rockwool"]["val"]
convection_lid = df_vars.loc["convection_lid"]["val"]
convection_side = df_vars.loc["convection_side"]["val"]
innerBucket_radius = df_vars.loc["innerBucket_radius"]["val"]
convection_bottom = df_vars.loc["convection_bottom"]["val"]
specific_heat_water = df_vars.loc["specific_heat_water"]["val"]

# %% Resistance equation
side = sp.Symbol("s")
resistance_conduction_side = sp.ln((innerBucket_radius + side) / innerBucket_radius) / (
    2 * sp.pi * innerBucket_height * conduction_rockwool
)
resistance_convection_side = 1 / (convection_side * 2 * sp.pi * innerBucket_height)
resistance_total_side = resistance_conduction_side + resistance_convection_side

lid = sp.Symbol("l")
resistance_conduction_lid = lid / (
    conduction_rockwool * sp.pi * innerBucket_radius**2
)
resistance_convection_lid = 1 / (convection_lid * sp.pi * innerBucket_radius**2)
resistance_total_lid = resistance_conduction_lid + resistance_convection_lid

bottom = sp.Symbol("b")
resistance_conduction_bottom = bottom / (
    conduction_rockwool * sp.pi * innerBucket_radius**2
)
resistance_convection_bottom = 1 / (convection_bottom * sp.pi * innerBucket_radius**2)
resistance_total_bottom = resistance_conduction_bottom + resistance_convection_bottom

resistance_total = 1 / (
    1 / resistance_total_bottom + 1 / resistance_total_side + 1 / resistance_total_lid
)


# %% Differential equation
def heatTransfer(Tin, Tout, R, C):
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
    qx = (Tout - Tin) / R
    dT = qx / C
    return dT


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

# Thermal resistance and heat capacity for simulation
Resistance_test = resistance_total.subs({lid: side, bottom: side, side: 0.1})
mass_water = 2.7  # kg
heat_capacity = specific_heat_water * mass_water

# Simulation
start_time = 0
start_temperature = 60  # C
step_size = 60  # s
end_time = 720000  # s
outside_temperature = 21  # C
t, T = eulersMethod(
    heatTransfer,
    start_time,
    start_temperature,
    step_size,
    end_time,
    outside_temperature,
    Resistance_test,
    heat_capacity,
)

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
