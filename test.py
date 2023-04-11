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

# %% Resistance equation
side = sp.Symbol('s')
resistance_conduction_side = sp.ln(
    (innerBucket_radius+side)/innerBucket_radius)/(2*sp.pi*innerBucket_height*conduction_rockwool)

lid = sp.Symbol('l')
resistance_conduction_lid = lid / \
    (conduction_rockwool*sp.pi*innerBucket_radius**2)

bottom = sp.Symbol('b')
resistance_conduction_bottom = bottom / \
    (conduction_rockwool*sp.pi*innerBucket_radius**2)

resistance_conduction_total = 1 / \
    (1/resistance_conduction_bottom + 1 /
     resistance_conduction_lid + 1/resistance_conduction_side)

p1 = sp.plotting.plot(resistance_conduction_total.subs(
    {lid: side, bottom: side}), (side, 0, 1), ylabel='R')
