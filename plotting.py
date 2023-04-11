# %% Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Load data
df_sim = pd.read_csv("sim.csv")
df_sim = df_sim.set_index("t")

df_param = pd.read_csv("param.csv")
df_param = df_param.set_index("param")

# %% Plots
plt.plot(df_sim.index / 3600, df_sim["T"], label="Tank temperature")
plt.axhline(
    y=df_param.loc["outside_temperature"]["val"],
    color="orange",
    linestyle="dashed",
    xmax=df_param.loc["end_time"]["val"],
    label="Outside Temperature",
)
plt.xlabel("Time [h]")
plt.ylabel("Temperature [C]")
plt.title("Cooling of thermal storage tank")
plt.grid()
plt.legend()
plt.tight_layout()
