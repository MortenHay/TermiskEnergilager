# %% Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

# %% Load data
df_sim = pd.read_csv("sim.csv")
df_sim = df_sim.set_index("t")

df_param = pd.read_csv("param.csv")
df_param = df_param.set_index("param")

df_run = pd.read_csv('imports/run1.csv')
df_run['Time'] = pd.to_datetime(df_run["Time"])
df_run['dt'] = (df_run['Time'] - df_run['Time'].shift(1)).dt.total_seconds()
df_run['dt'].iloc[0] = 0
df_run['t'] = np.cumsum(df_run['dt'])
df_run=df_run.set_index('t')


# %% Plots
plt.plot(df_sim.index / 3600, df_sim["T"], label="Tank temperature")

probes = ["top", "mid","bot", "room"]
for i in range(1,5):
    plt.plot(df_run.index / 3600, df_run["Probe-{}".format(i)], label="Probe {}".format(probes[i-1]))
    
    
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
