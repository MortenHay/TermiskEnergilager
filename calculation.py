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

# %% Main code
# Plot of total thermal resistance depending on insolation thickness
p1 = sp.plotting.plot(
    resistance_total.subs({lid: side, bottom: side}),
    (side, 0, 1),
    xlabel="Tykkelse [m]",
    ylabel="R [K/W]",
)
