import pandas as pd

# %%
df_vars = pd.DataFrame(columns=["var", "val"])

df_vars.loc[len(df_vars)] = ["innerBucket_height", 0.142]  # m
df_vars.loc[len(df_vars)] = ["outerBucket_height", 0.25]  # m
df_vars.loc[len(df_vars)] = ["conduction_rockwool", 37e-3]  # W/mK
df_vars.loc[len(df_vars)] = ["conduction_glasswool", 37e-3]  # W/mK
df_vars.loc[len(df_vars)] = ["conduction_PP", 0.11]  # W/mK
df_vars.loc[len(df_vars)] = ["convection_lid", 25]  # W(m2*K)
df_vars.loc[len(df_vars)] = ["convection_side", 10]  # W(m2*K)
df_vars.loc[len(df_vars)] = ["innerBucket_radius", (0.176 + 0.188) / 4]  # m
df_vars.loc[len(df_vars)] = ["outerBucket_radius", (0.29 + 0.254) / 4]  # m
df_vars.loc[len(df_vars)] = ["convection_bottom", 5]  # W(m2*K)
df_vars.loc[len(df_vars)] = ["specific_heat_water", 4180]  # J/(kg*K)
df_vars.loc[len(df_vars)] = ["width_bucket", 0.0007]  # m


df_vars = df_vars.set_index("var")
df_vars.to_csv("physicalData.csv")
