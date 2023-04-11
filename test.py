# %% Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

# %% Load data

# %% Physical constants
innerBucket_height = 0.14  # m
innerBucket_radius = 0.085  # m
conduction_rockwool = 35e-3  # W/mK

convection_lid = 25  # W(m2*K)
convection_side = 10  # W(m2*K)
convection_bottom = 5  # W(m2*K)

specific_heat_water = 4180  # J/(kg*K)

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
# Plot of total thermal resistance depending on insolation thickness
# p1 = sp.plotting.plot(
#     resistance_total.subs({lid: side, bottom: side}),
#     (side, 0, 1),
#     xlabel="Tykkelse [m]",
#     ylabel="R [K/W]",
# )

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
df_sim = pd.DataFrame({"t": t, "T": T})

# Data processing
plt.plot(df_sim["t"] / 3600, df_sim["T"], label="Tank temperature")
plt.axhline(
    y=outside_temperature,
    color="orange",
    linestyle="dashed",
    xmax=end_time,
    label="Outside Temperature",
)
plt.xlabel("Time [h]")
plt.ylabel("Temperature [C]")
plt.title("Cooling of thermal storage tank")
plt.grid()
plt.legend()
plt.tight_layout()

# %%
