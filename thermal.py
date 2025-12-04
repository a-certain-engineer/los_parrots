# import math
import matplotlib.pyplot as plt
import numpy as np
# import scipy.integrate as integrate

# variables
P_des = 85  # bar
T_des = 214  # °C
T_fluid = 214  # °C
D_bar = 2.5  # m
D_ves = 3.0  # m
Thick_insulation = 0.05  # m
Density = 852.50  # Kg/m3
Flow_rate = 3227  # Kg / s
Viscosity_I = 1.259e-4  # Pa s
Viscosity_II = 4.06e-4  # Pa s
Cp_I = 4534  # J / Kg K
Cp_II = 4172.5  # J / Kg s
Thermal_conductivity_I = 0.658  # W / m K
Thermal_conductivity_II = 0.666  # W / m K
Delta_T = 30  # °C
Alpha_p = 5.57e-4  # K^-1
Thermal_conductivity_ins = 1.4  # W / m K
Intensity_0 = 1.44e5  # W / m2
Phi_0 = 1.5e13  # 1 / cm2 s
Energy_gamma = 6.0e6  # eV
Build_up = 1.4
Mu_steel = 24  # 1 / m
Thermal_conductivity_steel = 48.1  #  W / m K
T_1 = 214.0  # °C
T_2 = 70.0  # °C
Kelvin = 273.15  # K
E = 177.0e9  # Pa
Nu = 0.3
Alpha_T = 1.7e-5  # 1 / K
Sigma_T = 0.56

# ASME III data for considered steel
Temperature = np.array(
    [
        40,
        65,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        275,
        300,
        325,
        350,
        375,
        400,
        425,
    ]
)
S_m = np.array(
    [160, 155, 148, 144, 140, 136, 133, 130, 127, 124, 121, 118, 114, 110, 105, 98]
)
S_y = np.array(
    [
        240,
        232.5,
        222,
        216,
        210,
        204,
        199.5,
        195,
        190.5,
        186,
        181.5,
        177,
        171,
        165,
        157.5,
        147,
    ]
)

# contants
g = 9.806  # m / s
eV = 1.6e-19  # J

# geometry
D_e = D_ves - D_bar  # m
R_bar = D_bar / 2
R_ves = D_ves / 2

# Unit conversion
P_des = P_des * 1e5  # Pa
T_1 = T_1 + Kelvin
T_2 = T_2 + Kelvin
T_fluid = T_fluid + Kelvin
Energy_gamma = Energy_gamma * eV
Phi_0 = Phi_0 * 1e4
q03 = Energy_gamma * Phi_0 * Build_up * Mu_steel

# From vessel
idx = 11
Thickness = 0.112087912088  # m
h_1 = 7498.1  # W / m^2 K


# Point 4
q03_prime = (3 * S_m[idx] * 1e6 - P_des / 2 - P_des * R_ves / Thickness) / (
    (Sigma_T * Alpha_T * E) / (Thermal_conductivity_steel * (1 - Nu) * Mu_steel**2)
)

Shield_thickness = -1 / Mu_steel * np.log(q03_prime / (q03 * Build_up))

print(f"Thermal shield thickness: {Shield_thickness * 100:.5} cm")


# Point 5
A = (
    np.exp(-Mu_steel * Shield_thickness)
    * ((q03 * h_1) / (Mu_steel**2 * Thermal_conductivity_steel) - q03 / Mu_steel)
    - ((q03 * h_1) / (Mu_steel**2 * Thermal_conductivity_steel) - q03 / Mu_steel)
) / (2 * Thermal_conductivity_steel + Shield_thickness * h_1)
B = (
    T_fluid
    + q03 / (Mu_steel**2 * Thermal_conductivity_steel)
    + (Thermal_conductivity_steel * A) / h_1
    + q03 / (Mu_steel * h_1)
)

print(f"{A}")
print(f"{B}")

x = np.linspace(0, Shield_thickness, 100)
T_shield = (
    -q03 / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
    + A * x
    + B
)

plt.figure(figsize=(6, 5))
plt.plot(x, T_shield, label="Temperature progile in the thermal shield")
plt.xlabel("x (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature profile in the thermal shield")
plt.grid()
plt.show()

print(f"{-q03 / (Mu_steel**2 * Thermal_conductivity_steel) + B}")
