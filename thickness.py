import numpy as np
import math

# variables 
P_des = 85 # bar
T_des = 270 # °C
D_bar = 2.5 # m
D_ves = 3.0 # m
thick_insulation = 5 # cm
S_m = 124e6 # Pa
density = 852.50 # Kg/m3
flow_rate = 3227 # Kg / s
viscosity_I = 1.259e-4 # Pa s
Cp_I = 4534 # J / Kg K
thermal_conductivity_I = 0.658 # W / m K
thermal_conductivity_II = 0.666 # W / m K
Cp_II = 4172.5 # J / Kg s
viscosity_II = 4.06e-4 # Pa s
delta_T = 30 # °C
alpha_p = 5.57e-4 # K^-1

# contants
g = 9.806 # m / s
pi = 3.1415653595

# geometry
D_e = D_ves - D_bar # m
R_bar = D_bar / 2
R_ves = D_ves / 2

P_des = P_des * 1e5 # Pa

# calculate thickness
thickness = (P_des * R_ves) / (S_m - 0.5 * P_des)

print(f'Minimum thickness: {thickness}')

# convective heat transfer coefficient for primary fluid
area = pi * (D_ves**2 - D_bar**2) / 4
velocity = flow_rate / (density * area)

Re = (density * velocity * D_e) / viscosity_I
Pr_I = (viscosity_I * Cp_I) / thermal_conductivity_I
Nu_I = 0.023 * Re**0.8 * Pr_I**0.4

h1 = (Nu_I * thermal_conductivity_I) / D_e


# convective heat transfer coefficient for secondary fluid
# external diameter
D_ext = D_ves + 2 * thickness + 2 * thick_insulation

Pr_II = (viscosity_II * Cp_II) / thermal_conductivity_II
Gr = (g * alpha_p * delta_T * density**2 * D_ext**3) / viscosity_II**2
Nu_II = 0.13 * (Gr * Pr_II)**(1 / 3)

h2 = (Nu_II * thermal_conductivity_II) / D_ext

# global heat transfer coefficient vessel-thermal insulation
U1 = 1 / (R_ves / thermal_conductivity_I * math.log(R_ves / R_bar) + R_ves / (R_bar * h1))

# global heat transfer coefficient insulator-cpp
U2 = 1 / (R_bar / thermal_conductivity_I * math.log(R_ves / R_bar) + 1 /  h1)

print(f'{U1}; {U2}')