import math
import matplotlib.pyplot as plt
import numpy as np

# variables
P_des = 85  # bar
T_des = 270  # °C
D_bar = 2.5  # m
D_ves = 3.0  # m
thick_insulation = 0.05  # m
S_m = 124e6  # Pa
density = 852.50  # Kg/m3
flow_rate = 3227  # Kg / s
viscosity_I = 1.259e-4  # Pa s
Cp_I = 4534  # J / Kg K
thermal_conductivity_I = 0.658  # W / m K
thermal_conductivity_II = 0.666  # W / m K
Cp_II = 4172.5  # J / Kg s
viscosity_II = 4.06e-4  # Pa s
delta_T = 30  # °C
alpha_p = 5.57e-4  # K^-1
thermal_conductivity_ins = 1.4  # W / m K
intensity_0 = 1.44e5  # W / m2
phi_0 = 1.5e13  # 1 / cm2 s
energy_gamma = 6.0e6  # eV
build_up = 1.4
mu_steel = 24  # 1 / m
thermal_conductivity_steel = 48.1  #  W / m K
T1 = 214  # °C
T2 = 70  # °C
TK = 273.25  # K
S_y = 186e6  # Pa
E = 177e9  # Pa
nu = 0.3
alpha_T = 1.7e-5  # 1 / K
sigma_T = 0.56

# contants
g = 9.806  # m / s
eV = 1.6e-19  # J

# geometry
D_e = D_ves - D_bar  # m
R_bar = D_bar / 2
R_ves = D_ves / 2

P_des = P_des * 1e5  # Pa

# calculate thickness
thickness = (P_des * R_ves) / (S_m - 0.5 * P_des)

print(f"Minimum thickness= {thickness * 100:.5} cm")

# convective heat transfer coefficient for primary fluid
area = math.pi * (D_ves**2 - D_bar**2) / 4
velocity = flow_rate / (density * area)

Re = (density * velocity * D_e) / viscosity_I
Pr_I = (viscosity_I * Cp_I) / thermal_conductivity_I
Nu_I = 0.023 * Re**0.8 * Pr_I**0.4

h1 = (Nu_I * thermal_conductivity_I) / D_e

print(f"Convective heat transfer coefficient 1= {h1:.5} W / m^2 K")

# convective heat transfer coefficient for secondary fluid
# external diameter
D_ext = D_ves + 2 * thickness + 2 * thick_insulation

Pr_II = (viscosity_II * Cp_II) / thermal_conductivity_II
Gr = (g * alpha_p * delta_T * density**2 * D_ext**3) / viscosity_II**2
Nu_II = 0.13 * (Gr * Pr_II) ** (1 / 3)

h2 = (Nu_II * thermal_conductivity_II) / D_ext

print(f"Convective heat transfer coefficient 2= {h2:.5} W / m^2 K")


R1 = R_ves + thickness
R2 = R1 + thick_insulation

# global heat transfer coefficient vessel-thermal insulation
U1 = 1 / (R1 / thermal_conductivity_ins * math.log(R2 / R1) + R1 / (R2 * h1))

# global heat transfer coefficient insulator-cpp
U2 = 1 / (R2 / thermal_conductivity_ins * math.log(R2 / R1) + 1 / h2)

print(f"U1 vessel-thermal insulation= {U1:.5} W / K")
print(f"U2 insulator-cpp= {U2:.5} W / K")

# Point 6
phi_0 = phi_0 * 1e4
energy_gamma = energy_gamma * eV
q03 = phi_0 * energy_gamma * mu_steel * build_up

T1 = T1 + TK
T2 = T2 + TK
term1 = (
    -(q03 * thick_insulation)
    / (mu_steel * thermal_conductivity_ins)
    * np.exp(-mu_steel * thickness)
)
term2 = -(thick_insulation / (h2 * thermal_conductivity_ins)) * np.exp(
    -mu_steel * thickness
)
term3 = (q03 / (mu_steel**2 * thermal_conductivity_steel)) * np.exp(
    -mu_steel * thickness
)
term4 = -T1
term5 = -q03 / (mu_steel**2 * thermal_conductivity_steel)
term6 = -q03 / (mu_steel * h1)

numerator = term1 + term2 + term3 + term4 + term5 + term6
denominator = (
    thickness
    + thermal_conductivity_steel / h1
    + thermal_conductivity_steel / h2
    + (thermal_conductivity_steel * thick_insulation) / thermal_conductivity_ins
)

A = numerator / denominator

B = (
    T1
    + q03 / (mu_steel**2 * thermal_conductivity_steel)
    + (thermal_conductivity_steel * A) / h1
    + q03 / (mu_steel * h1)
)

x = np.linspace(0, thickness, 100)
T = (
    -q03 / (mu_steel**2 * thermal_conductivity_steel) * np.exp(-mu_steel * x)
    + A * x
    + B
)
plt.figure(figsize=(6, 5))
plt.plot(x, T, label="Temperature profile")
plt.xlabel("x (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature profile inside RPV wall")
plt.minorticks_on()
plt.grid(True)
plt.show()

q_second = (T1 - T2) / (
    1 / h1
    + thickness / (thermal_conductivity_steel * area)
    + thick_insulation / thermal_conductivity_ins
    + 1 / h2
)

# Cylinder
U1_c = 1 / (
    1 / h1
    + R_ves / thermal_conductivity_steel * np.log((R_ves + thickness) / R_ves)
    + R_ves
    / thermal_conductivity_ins
    * np.log((R_ves + thickness + thick_insulation) / (R_ves + thick_insulation))
    + R_ves / (R_ves + thickness + thick_insulation) * 1 / h2
)

# global heat transfer coefficient insulator-cpp
U2_c = 1 / (
    (R_ves + thickness) / R_ves * 1 / h1
    + (R_ves + thickness)
    / thermal_conductivity_steel
    * np.log((R_ves + thickness) / R_ves)
    + (R_ves + thickness)
    / thermal_conductivity_ins
    * np.log((R_ves + thickness + thick_insulation) / (R_ves + thick_insulation))
    + (R_ves + thickness) / (R_ves + thickness + thick_insulation) * 1 / h2
)

print(f"U1 vessel-thermal insulation= {U1_c:.5} W / K")
print(f"U2 insulator-cpp= {U2_c:.5} W / K")

A_c = -R_ves / thermal_conductivity_steel * U1_c * (T1 - T2)
B_c = -(R_ves + thickness) / thermal_conductivity_steel * U2 * (T1 - T2)
r = np.linspace(1e-5, thickness, 100)

T_c = A_c * np.log(r) + B_c

plt.figure(figsize=(6, 5))
plt.plot(r, T_c, label="Temperature profile")
plt.xlabel("x (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature profile inside RPV wall (cylinder)")
plt.minorticks_on()
plt.grid(True)
plt.show()

check = (U1_c * R_ves) / (U2_c * (R_ves + thickness))

if round(check, 4) == 1:
    print(f"ok, ratio is: {round(check, 4)}")
else:
    print(f"not ok, ratio is: {round(check, 4)}")


# Point 8
P_m = P_des * R_ves / thickness + P_des / 2
if P_m <= S_m:
    print(f"Good, {P_m:.5} is less than {S_m:.5}")
else:
    print(f"Not good, {P_m:.5} is more than {S_m:.5}")

Q = (sigma_T * alpha_T * E * q03) / (
    thermal_conductivity_steel * (1 - nu) * mu_steel**2
)

test = Q + P_m
test_y = 2 * S_y
if test <= test_y:
    print(f"Good, {test:.5} is less than {test_y:.5}")
else:
    print(f"Not good, {test:.5} is more than {test_y:.5}")
