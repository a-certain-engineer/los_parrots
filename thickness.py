import numpy as np
import math

# variables 
P_des = 75 # bar
T_des = 270 # °C
D_bar = 2.5 # m
D_ves = 3.0 # m
thick_insulation = 5 # cm
S_m = 124e6 # Pa
density = 852.50 # Kg/m3
viscosity = 1.259e-14 # Pa s
D_e = D_ves - D_bar # m
flow_rate = 3227 # Kg / s
pi = 3.1415653595
C_p = 4534 # J / Kg K
thermal_conductivity = 0.658 # W / m K


# radius
R_bar = D_bar / 2
R_ves = D_ves / 2

P_des = P_des * 1e5 # Pa

thickness = (P_des * R_ves) / (S_m - 0.5 * P_des)

print(f'Minimum thickness: {thickness}')

velocity = flow_rate / (density * pi / 4 * (D_ves**2 - D_bar**2))

Re = (density * velocity * D_e) / viscosity 
Pr = (velocity * C_p) / thermal_conductivity 
Nu = 0.023 * Re**0.8 * Pr**0.4

h1 = (Nu * thermal_conductivity) / D_e

print(f'heat {h1}')