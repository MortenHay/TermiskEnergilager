import pandas as pd

# %%
df_vars = pd.DataFrame(columns=["var", "val"])

df_vars.loc[len(df_vars)] = ["innerBucket_height", 0.14]  # m
df_vars.loc[len(df_vars)] = ["conduction_rockwool", 35e-3]  # W/mK
df_vars.loc[len(df_vars)] = ["conduction_glasswool", 37e-3] #W/mK
df_vars.loc[len(df_vars)] = ["convection_lid", 25]  # W(m2*K)
df_vars.loc[len(df_vars)] = ["convection_side", 10]  # W(m2*K)
df_vars.loc[len(df_vars)] = ["innerBucket_radius", 0.085]  # m
df_vars.loc[len(df_vars)] = ["convection_bottom", 5]  # W(m2*K)
df_vars.loc[len(df_vars)] = ["specific_heat_water", 4180]  # J/(kg*K)

df_vars = df_vars.set_index("var")
df_vars.to_csv("physicalData.csv")
