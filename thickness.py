import numpy as np
import math

# design variables 
P_des = 85 # bar
T_des = 270 # °C
D_bar = 2.5 # m
D_ves = 3.0 # m
thick_insulation = 5 # cm
S_m = 124e6 # Pa

# radius
R_bar = D_bar / 2
R_ves = D_ves / 2

P_des = P_des * 1e5

thickness = (P_des * R_ves) / (S_m-0.5*P_des)

print(f'minimum thickness: {thickness}')