import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

# Variables
P_des = 85  # bar
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
S_corradi = 2
P_des_ext = 75  # bar

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

# Contants
g = 9.806  # m / s
eV = 1.6e-19  # J

# Geometry
D_e = D_ves - D_bar  # m
R_bar = D_bar / 2
R_ves = D_ves / 2

# Change measurement units
T_1 = T_1 + Kelvin
T_2 = T_2 + Kelvin
Phi_0 = Phi_0 * 1e4
Energy_gamma = Energy_gamma * eV


# Define temperature function
def Temperature_profile(x):
    return (
        -q03 / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
        + A * x
        + B
    )


# Point 2
# Design pressure
P_des = P_des * 1e5  # Pa
P_des_ext = P_des_ext * 1e5  # Pa

# Guess design temperature
T_des = 270  # °C


# Point 3
# Guess thickness
Thickness = (P_des * R_ves) / (S_m[4] * 1e6 - 0.5 * P_des)


# Point 4
# Convective heat transfer coefficient for primary fluid
Area = math.pi * (D_ves**2 - D_bar**2) / 4
Velocity = Flow_rate / (Density * Area)

Re = (Density * Velocity * D_e) / Viscosity_I
Pr_I = (Viscosity_I * Cp_I) / Thermal_conductivity_I

# Dittus-Boelter correlation
Nu_I = 0.023 * Re**0.8 * Pr_I**0.4

h_1 = (Nu_I * Thermal_conductivity_I) / D_e

# Convective heat transfer coefficient for secondary fluid
D_ext = D_ves + 2 * Thickness + 2 * Thick_insulation

Pr_II = (Viscosity_II * Cp_II) / Thermal_conductivity_II
Gr = (g * Alpha_p * Delta_T * Density**2 * D_ext**3) / Viscosity_II**2

# Mc Adams correlation
Nu_II = 0.13 * (Gr * Pr_II) ** (1 / 3)

h_2 = (Nu_II * Thermal_conductivity_II) / D_ext

# Global heat transfer coefficient vessel-thermal insulation
U_1 = 1 / (
    (R_ves + Thickness)
    / Thermal_conductivity_ins
    * np.log((R_ves + Thickness + Thick_insulation) / (R_ves + Thickness))
    + (R_ves + Thickness) / ((R_ves + Thickness + Thick_insulation) * h_1)
)

# Global heat transfer coefficient insulator-cpp
U_2 = 1 / (
    (R_ves + Thickness + Thick_insulation)
    / Thermal_conductivity_ins
    * np.log((R_ves + Thickness + Thick_insulation) / (R_ves + Thickness))
    + 1 / h_2
)


# Point 5
Intensity_0 = Phi_0 * Energy_gamma
Intensity = Intensity_0 * np.exp(-Mu_steel * Thickness) * Build_up
Vol_q03 = Mu_steel * Intensity_0

# Point 6
q03 = Phi_0 * Energy_gamma * Mu_steel * Build_up

# Point 2 and 3
# Iterative cycle to find the design teperature, thickness
T_avg = (T_1 + T_2) / 2

T_prev = T_avg
toll = 10
idx = 4

while toll >= 1:
    Thickness = (P_des * R_ves) / (S_m[idx] * 1e6 - 0.5 * P_des)
    # Linear system for A and B
    # Equation for A
    term1 = (
        -(q03 * Thick_insulation)
        / (Mu_steel * Thermal_conductivity_ins)
        * np.exp(-Mu_steel * Thickness)
    )
    term2 = -(Thick_insulation / (h_2 * Thermal_conductivity_ins)) * np.exp(
        -Mu_steel * Thickness
    )
    term3 = (q03 / (Mu_steel**2 * Thermal_conductivity_steel)) * np.exp(
        -Mu_steel * Thickness
    )
    term4 = -T_1
    term5 = -q03 / (Mu_steel**2 * Thermal_conductivity_steel)
    term6 = -q03 / (Mu_steel * h_1)

    numerator = term1 + term2 + term3 + term4 + term5 + term6
    denominator = (
        Thickness
        + Thermal_conductivity_steel / h_1
        + Thermal_conductivity_steel / h_2
        + (Thermal_conductivity_steel * Thick_insulation) / Thermal_conductivity_ins
    )
    alpha1 = denominator
    beta1 = 0
    gamma1 = numerator

    # Equation for B
    alpha2 = -(Thermal_conductivity_steel / h_1)
    beta2 = 1
    gamma2 = (
        T_1 + q03 / (Mu_steel**2 * Thermal_conductivity_steel) + q03 / (Mu_steel * h_1)
    )

    # Matrices for the system
    A_mat = np.array([[alpha1, beta1], [alpha2, beta2]])

    # Coefficients vector
    b_vec = np.array([gamma1, gamma2])

    # Solutions
    A, B = np.linalg.solve(A_mat, b_vec)

    # Print average pressure
    if T_avg == T_prev:
        print(f"Temperatura media: {T_avg} K")

    # Calculate and print average pressure
    T_int, err = integrate.quad(Temperature_profile, 0.0, Thickness)
    T_des = T_int / Thickness
    print(f"Temperatura design: {T_des:.5} K")

    # Calculate tollerance and update previous pressure
    toll = np.absolute(T_prev - T_des)
    T_prev = T_des
    T_des = T_des - Kelvin

    # Get index for the closest pressure
    idx = np.where(Temperature > T_des, Temperature - T_des, np.inf).argmin()

    # Print results
    print(
        f"Spessore: {np.round(Thickness * 100, 10)} cm | Temperatura: {T_des:.5} C | Indice: {idx} | Tolleranza: {toll:.5} | Ultimate: {S_m[idx]} MPa"
    )


# Buckling
D_1 = (D_ves + 1.25) / 200
D_2 = D_ves / 100
Delta_D_max = min(D_1, D_2)
W = Delta_D_max / D_ves
t_cr = D_ves / (math.sqrt(E / (S_y[idx] * 1e6 * (1 - Nu**2))))
sigma_lim = 2 / 3 * S_y[idx] * 1e6
max_iter = 1000
increment = 1e-4
P_all = 0
iteration = 0
# print(f"{Delta_D_max}")
# print(f"{W}")
# print(f"{sigma_lim}")

Thickness_min = (P_des_ext * R_ves) / (sigma_lim - 0.5 * P_des_ext)

while P_all < P_des_ext and iteration <= max_iter:
    Thickness_buckling = Thickness_min + iteration * increment

    slender = D_ves / Thickness_buckling
    D_ext = D_ves + 2 * Thickness_buckling

    Z = math.sqrt(3) / 4 * (2 * D_ext / Thickness_buckling + 1) * W

    q_E = (
        (2 * E)
        / (1 - Nu**2)
        * 1
        / (D_ext / Thickness_buckling * (D_ext / Thickness_buckling - 1) ** 2)
    )
    q_0 = (
        2
        * S_y[idx]
        * 1e6
        * Thickness_buckling
        / D_ext
        * (1 + 0.5 * Thickness_buckling / D_ext)
    )

    q_U = q_0 / math.sqrt(1 + Z**2)
    sqrt_term = (q_0 + q_E * (1 + Z)) ** 2 - 4 * q_0 * q_E
    if sqrt_term < 0:
        sqrt_term = 0
    q_L = 0.5 * (q_0 + q_E * (1 + Z) - math.sqrt(sqrt_term))

    q_ratio = q_0 / q_E

    if q_ratio < 0.04:
        mu = 1.0
    elif q_ratio > 0.7:
        mu = 0.0
    else:
        mu = 0.35 * np.log(q_E / q_0) - 0.125

    q_C = mu * q_U + (1 - mu) * q_L

    P_all = q_C / S_corradi
    # print(
    #     f"Iteration {iteration}: Thickness = {Thickness_buckling:.5f}, P_all = {P_all:.2f}"
    # )

    iteration += 1

print(f"Final thickness for buckling: {Thickness_buckling * 100:.5f} cm")

Thickness = Thickness_buckling
T_prev = T_avg
toll = 10

while toll >= 1:
    # Linear system for A and B
    # Equation for A
    term1 = (
        -(q03 * Thick_insulation)
        / (Mu_steel * Thermal_conductivity_ins)
        * np.exp(-Mu_steel * Thickness)
    )
    term2 = -(Thick_insulation / (h_2 * Thermal_conductivity_ins)) * np.exp(
        -Mu_steel * Thickness
    )
    term3 = (q03 / (Mu_steel**2 * Thermal_conductivity_steel)) * np.exp(
        -Mu_steel * Thickness
    )
    term4 = -T_1
    term5 = -q03 / (Mu_steel**2 * Thermal_conductivity_steel)
    term6 = -q03 / (Mu_steel * h_1)

    numerator = term1 + term2 + term3 + term4 + term5 + term6
    denominator = (
        Thickness
        + Thermal_conductivity_steel / h_1
        + Thermal_conductivity_steel / h_2
        + (Thermal_conductivity_steel * Thick_insulation) / Thermal_conductivity_ins
    )
    alpha1 = denominator
    beta1 = 0
    gamma1 = numerator

    # Equation for B
    alpha2 = -(Thermal_conductivity_steel / h_1)
    beta2 = 1
    gamma2 = (
        T_1 + q03 / (Mu_steel**2 * Thermal_conductivity_steel) + q03 / (Mu_steel * h_1)
    )

    # Matrices for the system
    A_mat = np.array([[alpha1, beta1], [alpha2, beta2]])

    # Coefficients vector
    b_vec = np.array([gamma1, gamma2])

    # Solutions
    A, B = np.linalg.solve(A_mat, b_vec)

    # Print average pressure
    if T_avg == T_prev:
        print(f"Temperatura media: {T_avg} K")

    # Calculate and print average pressure
    T_int, err = integrate.quad(Temperature_profile, 0.0, Thickness)
    T_des = T_int / Thickness
    print(f"Temperatura design: {T_des:.5} K")

    # Calculate tollerance and update previous pressure
    toll = np.absolute(T_prev - T_des)
    T_prev = T_des
    T_des = T_des - Kelvin

    # Get index for the closest pressure
    idx = np.where(Temperature > T_des, Temperature - T_des, np.inf).argmin()

    # Print results
    print(
        f"Spessore: {np.round(Thickness * 100, 10)} cm | Temperatura: {T_des:.5} C | Indice: {idx} | Tolleranza: {toll:.5} | Ultimate: {S_m[idx]} MPa"
    )

# Point ???
# Convective heat transfer coefficient for primary fluid
print(f"Convective heat transfer coefficient 1= {h_1:.5} W / m^2 K")

# Convective heat transfer coefficient for secondary fluid
print(f"Convective heat transfer coefficient 2= {h_2:.5} W / m^2 K")

R_1 = R_ves + Thickness
R_2 = R_1 + Thick_insulation

# global heat transfer coefficient vessel-thermal insulation
U_1 = 1 / (R_1 / Thermal_conductivity_ins * math.log(R_2 / R_1) + R_1 / (R_2 * h_1))

# global heat transfer coefficient insulator-cpp
U_2 = 1 / (R_2 / Thermal_conductivity_ins * math.log(R_2 / R_1) + 1 / h_2)

print(f"U1 vessel-thermal insulation= {U_1:.5} W / K")
print(f"U2 insulator-cpp= {U_2:.5} W / K")

# Point 5
x = np.linspace(0, Thickness, 100)
Vol_q3 = Vol_q03 * np.exp(-Mu_steel * x) * Build_up

plt.figure(figsize=(6, 5))
plt.plot(
    x, Vol_q3 * 1e-6, label="Volumetric heat source radial profile"
)  # x-axis = radius
plt.xlabel("x (m)")
plt.ylabel("Volumetric heat source (MW / m^3)")
plt.title("Radial profile of volumetric heat source")
plt.grid(True)

# Point 6
# Plot temperature profile
T = (
    -q03 / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
    + A * x
    + B
)

plt.figure(figsize=(6, 5))
plt.plot(x, T, label="Temperature profile")
plt.xlabel("x (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature profile inside vessel")
plt.minorticks_on()
plt.grid()

# Point 7 (slab)
# Calculate heating
# q_second = (T_1 - T_2) / (
#     1 / h_1
#     + Thickness / (Thermal_conductivity_steel * Area)
#     + Thick_insulation / Thermal_conductivity_ins
#     + 1 / h_2
# )

# Point 7
U_1c = 1 / (
    1 / h_1
    + R_ves / Thermal_conductivity_steel * np.log((R_ves + Thickness) / R_ves)
    + R_ves
    / Thermal_conductivity_ins
    * np.log((R_ves + Thickness + Thick_insulation) / (R_ves + Thick_insulation))
    + R_ves / (R_ves + Thickness + Thick_insulation) * 1 / h_2
)

# global heat transfer coefficient insulator-cpp
U_2c = 1 / (
    (R_ves + Thickness) / R_ves * 1 / h_1
    + (R_ves + Thickness)
    / Thermal_conductivity_steel
    * np.log((R_ves + Thickness) / R_ves)
    + (R_ves + Thickness)
    / Thermal_conductivity_ins
    * np.log((R_ves + Thickness + Thick_insulation) / (R_ves + Thick_insulation))
    + (R_ves + Thickness) / (R_ves + Thickness + Thick_insulation) * 1 / h_2
)

print(f"U1 vessel-thermal insulation= {U_1c:.5} W / K")
print(f"U2 insulator-cpp= {U_2c:.5} W / K")


check = (U_1c * R_ves) / (U_2c * (R_ves + Thickness))

if round(check, 4) == 1:
    print(f"ok, ratio is: {round(check, 4)}")
else:
    print(f"not ok, ratio is: {round(check, 4)}")


A_c = -R_ves / Thermal_conductivity_steel * U_1c * (T_1 - T_2)
B_c = -(U_1c / h_1 * (T_1 - T_2) - T_1 + A_c * np.log(R_ves))
r = np.linspace(R_ves, R_ves + Thickness, 100)

T_c = A_c * np.log(r) + B_c

plt.figure(figsize=(6, 5))
plt.plot(r, T_c, label="Temperature profile")
plt.xlabel("x (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature profile inside RPV wall (cylinder)")
plt.minorticks_on()
plt.grid(True)
plt.show()


# Point 8
# Mechanical stresses
P_m = P_des * R_ves / Thickness + P_des / 2
Sigma_r = (
    -(R_ves**2)
    / ((R_ves + Thickness) ** 2 - R_ves**2)
    * ((R_ves + Thickness) ** 2 / r**2 - 1)
    * P_des
)
Sigma_theta = (
    +(R_ves**2)
    / ((R_ves + Thickness) ** 2 - R_ves**2)
    * ((R_ves + Thickness) ** 2 / r**2 + 1)
    * P_des
)

Sigma_Lame = np.max(Sigma_theta - Sigma_r)
idx_max = np.argmax(Sigma_theta - Sigma_r)
pos = r[idx_max]

Stress_I = S_m[idx] * 1e6

if P_m <= Stress_I:
    print(f"Good, {P_m:.5} is less than {Stress_I:.5}")
else:
    print(f"Not good, {P_m:.5} is more than {Stress_I:.5}")

# Thermal stresses
Q = (Sigma_T * Alpha_T * E * q03) / (
    Thermal_conductivity_steel * (1 - Nu) * Mu_steel**2
)

# Mechanical and thermal stresses
test = Q + Sigma_Lame
test_y = 2 * S_y[idx] * 1e6
if test < test_y:
    print(f"Good, {test:.5} is less than {test_y:.5}")
else:
    print(f"Not good, {test:.5} is more than {test_y:.5}")
