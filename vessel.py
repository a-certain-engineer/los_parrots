import math
import matplotlib.pyplot as plt
import numpy as np

# variables
P_des = 85  # bar
T_des = 270  # °C
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

P_des = P_des * 1e5  # Pa
T_1 = T_1 + Kelvin
T_2 = T_2 + Kelvin


# calculate integral
def T_integral_norm(t, q03, Mu_steel, Thermal_conductivity_steel, A, B):
    term_exp = (q03 / (Mu_steel**3 * Thermal_conductivity_steel)) * (
        np.exp(-Mu_steel * t) - 1
    )
    term_A = 0.5 * A * t
    term_B = B
    return (term_exp + term_A * t + term_B * t) / t


# Point 3
# calculate thickness
Thickness = (P_des * R_ves) / (S_m[4] * 1e6 - 0.5 * P_des)

# print(f"Minimum thickness= {Thickness * 100:.5} cm")

# Point 4
# convective heat transfer coefficient for primary fluid
Area = math.pi * (D_ves**2 - D_bar**2) / 4
Velocity = Flow_rate / (Density * Area)

Re = (Density * Velocity * D_e) / Viscosity_I
Pr_I = (Viscosity_I * Cp_I) / Thermal_conductivity_I
Nu_I = 0.023 * Re**0.8 * Pr_I**0.4

h_1 = (Nu_I * Thermal_conductivity_I) / D_e

print(f"Convective heat transfer coefficient 1= {h_1:.5} W / m^2 K")

# convective heat transfer coefficient for secondary fluid
# external diameter
D_ext = D_ves + 2 * Thickness + 2 * Thick_insulation

Pr_II = (Viscosity_II * Cp_II) / Thermal_conductivity_II
Gr = (g * Alpha_p * Delta_T * Density**2 * D_ext**3) / Viscosity_II**2
Nu_II = 0.13 * (Gr * Pr_II) ** (1 / 3)

h_2 = (Nu_II * Thermal_conductivity_II) / D_ext

print(f"Convective heat transfer coefficient 2= {h_2:.5} W / m^2 K")


R_1 = R_ves + Thickness
R_2 = R_1 + Thick_insulation

# global heat transfer coefficient vessel-thermal insulation
U_1 = 1 / (R_1 / Thermal_conductivity_ins * math.log(R_2 / R_1) + R_1 / (R_2 * h_1))

# global heat transfer coefficient insulator-cpp
U_2 = 1 / (R_2 / Thermal_conductivity_ins * math.log(R_2 / R_1) + 1 / h_2)

print(f"U1 vessel-thermal insulation= {U_1:.5} W / K")
print(f"U2 insulator-cpp= {U_2:.5} W / K")

# Point 6
Phi_0 = Phi_0 * 1e4
Energy_gamma = Energy_gamma * eV
q03 = Phi_0 * Energy_gamma * Mu_steel * Build_up

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

A = numerator / denominator

B = (
    T_1
    + q03 / (Mu_steel**2 * Thermal_conductivity_steel)
    + (Thermal_conductivity_steel * A) / h_1
    + q03 / (Mu_steel * h_1)
)

x = np.linspace(0, Thickness, 100)
T = (
    -q03 / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
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

q_second = (T_1 - T_2) / (
    1 / h_1
    + Thickness / (Thermal_conductivity_steel * Area)
    + Thick_insulation / Thermal_conductivity_ins
    + 1 / h_2
)

# Point 7 cylinder
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

A_c = -R_ves / Thermal_conductivity_steel * U_1c * (T_1 - T_2)
B_c = -(R_ves + Thickness) / Thermal_conductivity_steel * U_2c * (T_1 - T_2)
r = np.linspace(1e-5, Thickness, 100)

T_c = A_c * np.log(r) + B_c

plt.figure(figsize=(6, 5))
plt.plot(r, T_c, label="Temperature profile")
plt.xlabel("x (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature profile inside RPV wall (cylinder)")
plt.minorticks_on()
plt.grid(True)


# Point 2
T_avg = (T_1 + T_2) / 2

T_prev = T_avg
toll = 10
idx = 4

while toll >= 1:
    Thickness = (P_des * R_ves) / (S_m[idx] * 1e6 - 0.5 * P_des)
    # Costruisco il sistema lineare per A e B

    # Equazione 1 ricavata dal tuo calcolo di A:
    # A = numerator / denominator
    #  → A * denominator - numerator = 0
    alpha1 = denominator
    beta1 = 0
    gamma1 = numerator

    # Equazione 2 ricavata dalla tua formula per B:
    # B = T_1 + q03/(mu^2 k) + (k/h1)*A + q03/(mu h1)
    #  → (-k/h1) * A + B = T1 + ...
    alpha2 = -(Thermal_conductivity_steel / h_1)
    beta2 = 1
    gamma2 = (
        T_1 + q03 / (Mu_steel**2 * Thermal_conductivity_steel) + q03 / (Mu_steel * h_1)
    )

    # Matrici per il sistema
    A_mat = np.array([[alpha1, beta1], [alpha2, beta2]])

    b_vec = np.array([gamma1, gamma2])

    # 🔥 Soluzione immediata
    A, B = np.linalg.solve(A_mat, b_vec)

    T = (
        -q03 / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
        + A * x
        + B
    )

    if T_avg == T_prev:
        print(f"Temperatura media: {T_avg}")

    T_des = T_integral_norm(Thickness, q03, Mu_steel, Thermal_conductivity_steel, A, B)

    print(f"Temperatura design: {T_des}")
    toll = np.absolute(T_prev - T_des)
    T_prev = T_des
    T_des = T_des - Kelvin

    idx = np.where(Temperature > T_des, Temperature - T_des, np.inf).argmin()

    print(
        f"Spessore: {np.round(Thickness * 100, 10)} | Temperatura: {T_des:.5} | Indice: {idx} | Tolleranza: {toll:.5} | Ultimate: {S_m[idx]}"
    )


Area = math.pi * (D_ves**2 - D_bar**2) / 4
Velocity = Flow_rate / (Density * Area)

Re = (Density * Velocity * D_e) / Viscosity_I
Pr_I = (Viscosity_I * Cp_I) / Thermal_conductivity_I
Nu_I = 0.023 * Re**0.8 * Pr_I**0.4

h_1 = (Nu_I * Thermal_conductivity_I) / D_e

print(f"Convective heat transfer coefficient 1= {h_1:.5} W / m^2 K")

# convective heat transfer coefficient for secondary fluid
# external diameter
D_ext = D_ves + 2 * Thickness + 2 * Thick_insulation

Pr_II = (Viscosity_II * Cp_II) / Thermal_conductivity_II
Gr = (g * Alpha_p * Delta_T * Density**2 * D_ext**3) / Viscosity_II**2
Nu_II = 0.13 * (Gr * Pr_II) ** (1 / 3)

h_2 = (Nu_II * Thermal_conductivity_II) / D_ext

print(f"Convective heat transfer coefficient 2= {h_2:.5} W / m^2 K")


R_1 = R_ves + Thickness
R_2 = R_1 + Thick_insulation

# global heat transfer coefficient vessel-thermal insulation
U_1 = 1 / (R_1 / Thermal_conductivity_ins * math.log(R_2 / R_1) + R_1 / (R_2 * h_1))

# global heat transfer coefficient insulator-cpp
U_2 = 1 / (R_2 / Thermal_conductivity_ins * math.log(R_2 / R_1) + 1 / h_2)

print(f"U1 vessel-thermal insulation= {U_1:.5} W / K")
print(f"U2 insulator-cpp= {U_2:.5} W / K")

# Point 6
Phi_0 = Phi_0 * 1e4
Energy_gamma = Energy_gamma * eV
q03 = Phi_0 * Energy_gamma * Mu_steel * Build_up

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

A = numerator / denominator

B = (
    T_1
    + q03 / (Mu_steel**2 * Thermal_conductivity_steel)
    + (Thermal_conductivity_steel * A) / h_1
    + q03 / (Mu_steel * h_1)
)

x = np.linspace(0, Thickness, 100)
T = (
    -q03 / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
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

q_second = (T_1 - T_2) / (
    1 / h_1
    + Thickness / (Thermal_conductivity_steel * Area)
    + Thick_insulation / Thermal_conductivity_ins
    + 1 / h_2
)

# Point 7 cylinder
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
B_c = -(R_ves + Thickness) / Thermal_conductivity_steel * U_2c * (T_1 - T_2)
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


check = (U_1c * R_ves) / (U_2c * (R_ves + Thickness))

if round(check, 4) == 1:
    print(f"ok, ratio is: {round(check, 4)}")
else:
    print(f"not ok, ratio is: {round(check, 4)}")

# Point 8
P_m = P_des * R_ves / Thickness + P_des / 2
if P_m <= (S_m[4] * 1e6):
    print(f"Good, {P_m:.5} is less than {S_m[4]}")
else:
    print(f"Not good, {P_m:.5} is more than {S_m[4]}")

Q = (Sigma_T * Alpha_T * E * q03) / (
    Thermal_conductivity_steel * (1 - Nu) * Mu_steel**2
)

test = Q + P_m
test_y = 2 * S_y[4]
if test <= test_y:
    print(f"Good, {test:.5} is less than {test_y}")
else:
    print(f"Not good, {test:.5} is more than {test_y}")
