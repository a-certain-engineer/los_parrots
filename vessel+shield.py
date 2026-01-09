import math
import functions
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.constants as constants

plt.rc("font", size=13)  # Increase size of axis numbers and titles

# variables
P_des = 82.5  # bar
P_des_ext = 75  # bar
T_des = 214  # °C
T_fluid = 214  # °C
D_bar = 2.5  # m
D_ves = 3.0  # m
Thickness_insulation = 0.05  # m
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
S_corradi = 2
Sigma_T_2 = 0.08  # From graph
a = 2.75  # m

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

# From vessel
h_1 = 7498.1  # W / m^2 K
h_2 = 1060.5  # W / m^2 K

# From shield
Thickness_shield = 0.021699  # m
idx_S_m = 9
q03_prime = 2855147.1704780236  # W / m^2

# geometry
D_e = D_ves - (a + 2 * Thickness_shield)  # m
R_bar = D_bar / 2
R_ves = D_ves / 2

# Unit conversion
P_des = P_des * 1e5  # Pa
P_des_ext = P_des_ext * 1e5  # Pa
T_1 = T_1 + Kelvin
T_2 = T_2 + Kelvin
T_fluid = T_fluid + Kelvin
Energy_gamma = Energy_gamma * constants.eV
Phi_0 = Phi_0 * 1e4
q03 = Energy_gamma * Phi_0 * Build_up * Mu_steel

# Thickness_vessel = 1.1 * Thickness_vessel

# h1  per calcoli
# Convective heat transfer coefficient for primary fluid
Area = math.pi * (D_ves**2 - D_bar**2) / 4 - math.pi / 4 * (
    (a + 2 * Thickness_shield) ** 2 - a**2
)
Velocity = Flow_rate / (Density * Area)

Re = (Density * Velocity * D_e) / Viscosity_I
Pr_I = (Viscosity_I * Cp_I) / Thermal_conductivity_I

# Dittus-Boelter correlation
Nu_I = 0.023 * Re**0.8 * Pr_I**0.4

h_1 = (Nu_I * Thermal_conductivity_I) / D_e


# Point 3
# Guess thickness
T_avg = (T_1 + T_2) / 2
idx = functions.find_index(T_avg)
Thickness_tresca = (P_des * R_ves) / (S_m[idx] * 1e6 - 0.5 * P_des)

# Point 2 and 3
# Iterative cycle to find the design teperature, thickness
T_prev_tresca = T_avg
toll = 10

print("Tresca")
while toll >= 1:
    Thickness_tresca = (P_des * R_ves) / (S_m[idx] * 1e6 - 0.5 * P_des)

    # Solve linear equation for A and B coefficients
    A_tresca, B_tresca = functions.solve_coefficients_prime(
        Thickness_tresca, h_1, h_2, q03_prime
    )

    # Calculate average temperature
    T_int_tresca, err = integrate.quad(
        functions.Temperature_profile_prime,
        0.0,
        Thickness_tresca,
        args=(A_tresca, B_tresca, q03_prime),
    )
    T_des_tresca = T_int_tresca / Thickness_tresca

    # Calculate tollerance and update previous temperature
    toll = np.absolute(T_prev_tresca - T_des_tresca)
    T_prev_tresca = T_des_tresca

    # Get index for the closest temperature
    idx_tresca = functions.find_index(T_des_tresca)

    # Print results
    print(
        f"Thickness: {Thickness_tresca * 100:.5} cm | Desgin temperature: {T_des_tresca - Kelvin:.5} C | Index: {idx_tresca} | Tollerance: {toll:.5}"
    )

# Buckling
print("\nBuckling")

# Ovality
D_1 = (D_ves + 1.25) / 200
D_2 = D_ves / 100
Delta_D_max = min(D_1, D_2)
W = Delta_D_max / D_ves

# Ovality check
if W < 0.025:
    print(f"Ovality check respected: W ({W:.4}) is less than 2.5%")
else:
    print(f"Ovality check not respected: W ({W:.4}) is more than 2.5%")

# Parameters for the loop
max_iter = 1000
increment = 1e-4
P_all = 0
iteration = 0
T_prev_buckling = T_avg

# Minimum guess thickness
Thickness_buckling = Thickness_tresca

while P_all < P_des_ext and iteration <= max_iter:
    # Guess thickness
    Thickness_buckling += increment

    # Solutions
    A_buckling, B_buckling = functions.solve_coefficients_prime(
        Thickness_buckling, h_1, h_2, q03_prime
    )

    # Calculate average temperature
    T_int_buckling, err = integrate.quad(
        functions.Temperature_profile_prime,
        0.0,
        Thickness_buckling,
        args=(A_buckling, B_buckling, q03_prime),
    )
    T_des_buckling = T_int_buckling / Thickness_buckling

    # Calculate tollerance and update previous temperature
    toll = np.absolute(T_prev_buckling - T_des_buckling)
    T_prev_buckling = T_des_buckling

    # Get index for the closest temperature
    idx_buckling = functions.find_index(T_des)

    # Geometric parameters
    Slender = D_ves / Thickness_buckling
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
        * S_y[idx_buckling]
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

    # Calculate allowable pressure
    P_all = q_C / S_corradi

    iteration += 1

    if iteration == 1000:
        print("System does not converge after 1000 iterations.")

print(
    f"Thickness: {Thickness_buckling * 100:.5f} cm | Iteration count: {iteration} | Design temperature: {T_des_buckling - Kelvin:.5f} C |"
)

# Critical thickness
t_critical = D_ves / (math.sqrt(E / (S_y[idx_buckling] * 1e6 * (1 - Nu**2))))

if Thickness_buckling < t_critical:
    print("Elastic buckling regime")
else:
    print("Plastic collapse regime")

# Update variables considering the limiting criterion
if Thickness_buckling > Thickness_tresca:
    Thickness_vessel = Thickness_buckling
    A, B = A_buckling, B_buckling
    T_des = T_des_buckling
    idx = idx_buckling
    print("Governing criterion: Buckling")
else:
    Thickness_vessel = Thickness_tresca
    A, B = A_tresca, B_tresca
    T_des = T_des_buckling
    idx = idx_tresca
    print("Governing criterion: Tresca")

# Point 4
# Area of the two cirular crowns
Area = math.pi * (D_ves**2 - D_bar**2) / 4 - math.pi / 4 * (
    (a + Thickness_shield) ** 2 - a**2
)

# Fluid velocity
Velocity = Flow_rate / (Density * Area)

Re = (Density * Velocity * D_e) / Viscosity_I
Pr_I = (Viscosity_I * Cp_I) / Thermal_conductivity_I

# Dittus-Boelter correlation
Nu_I = 0.023 * Re**0.8 * Pr_I**0.4

# Convective heat transfer coefficient for primary fluid
h_1 = (Nu_I * Thermal_conductivity_I) / D_e

# External length
D_ext = D_ves + 2 * Thickness_vessel + 2 * Thickness_insulation

Pr_II = (Viscosity_II * Cp_II) / Thermal_conductivity_II
Gr = (constants.g * Alpha_p * Delta_T * Density**2 * D_ext**3) / Viscosity_II**2

# Mc Adams correlation
Nu_II = 0.13 * (Gr * Pr_II) ** (1 / 3)

# Convective heat transfer coefficient for secondary fluid
h_2 = (Nu_II * Thermal_conductivity_II) / D_ext

# Convective heat transfer coefficient for primary fluid
print(f"\nConvective heat transfer coefficient 1= {h_1:.5} W / m^2 K")

# Convective heat transfer coefficient for secondary fluid
print(f"Convective heat transfer coefficient 2= {h_2:.5} W / m^2 K")

# Global heat transfer coefficient vessel-thermal insulation
U_1 = 1 / (
    (R_ves + Thickness_vessel)
    / Thermal_conductivity_ins
    * np.log(
        (R_ves + Thickness_vessel + Thickness_insulation) / (R_ves + Thickness_vessel)
    )
    + (R_ves + Thickness_vessel)
    / ((R_ves + Thickness_vessel + Thickness_insulation) * h_1)
)

# Global heat transfer coefficient insulator-cpp
U_2 = 1 / (
    (R_ves + Thickness_vessel + Thickness_insulation)
    / Thermal_conductivity_ins
    * np.log(
        (R_ves + Thickness_vessel + Thickness_insulation) / (R_ves + Thickness_vessel)
    )
    + 1 / h_2
)

# Point 5
x = np.linspace(0, Thickness_vessel, 100)

Vol_q3_prime = q03_prime * np.exp(-Mu_steel * x)

print(
    f"Volumetric heat flux at the vessel inner surface: {Vol_q3_prime[0] / 1e6:.5f} MW / m^3"
)
print(
    f"Volumetric heat flux at the vessel outer surface: {Vol_q3_prime[-1] / 1e6:.5f} MW / m^3"
)

plt.figure(figsize=(10, 6))
plt.plot(x + R_ves, Vol_q3_prime / 1e6, "b-", linewidth=2)
plt.xlabel("Radial position [m]")
plt.ylabel("Volumetric heat [MW/m³]")
plt.title("Volumetric heat source radial profile across vessel's thickness")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Point 6
# Plot temperature profile
T_profile = functions.Temperature_profile_prime(x, A, B, q03_prime)

plt.figure(figsize=(10, 6))
plt.plot(x + R_ves, T_profile, "b-", label="Temperature profile", linewidth=2)
plt.xlabel("Radial position [m]")
plt.ylabel("Temperature [K]")
plt.title("Temperature profile inside vessel")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

T_inner = T_profile[0]
T_outer = T_profile[-1]
idx_max_temperature = np.argmax(T_profile)
pos_max_temperature = x[idx_max_temperature]
T_max = np.max(T_profile)

print(f"Inner vessel temperature: {T_inner - Kelvin:.5f} C")
print(f"Outer vessel temperature: {T_outer - Kelvin:.5f} C")
print(f"Maximum temperature: {T_max - Kelvin:.5f} C")
print(f"Position of maximum temperature: {pos_max_temperature * 100:.5f} cm")

# Point 7
U_1c = 1 / (
    1 / h_1
    + R_ves / Thermal_conductivity_steel * np.log((R_ves + Thickness_vessel) / R_ves)
    + R_ves
    / Thermal_conductivity_ins
    * np.log(
        (R_ves + Thickness_vessel + Thickness_insulation) / (R_ves + Thickness_vessel)
    )
    + R_ves / (R_ves + Thickness_vessel + Thickness_insulation) * 1 / h_2
)

# global heat transfer coefficient insulator-cpp
U_2c = 1 / (
    (R_ves + Thickness_vessel) / R_ves * 1 / h_1
    + (R_ves + Thickness_vessel)
    / Thermal_conductivity_steel
    * np.log((R_ves + Thickness_vessel) / R_ves)
    + (R_ves + Thickness_vessel)
    / Thermal_conductivity_ins
    * np.log(
        (R_ves + Thickness_vessel + Thickness_insulation) / (R_ves + Thickness_vessel)
    )
    + (R_ves + Thickness_vessel)
    / (R_ves + Thickness_vessel + Thickness_insulation)
    * 1
    / h_2
)

print(f"U1 vessel-thermal insulation= {U_1c:.5} W / K")
print(f"U2 insulator-cpp= {U_2c:.5} W / K")


check = (U_1c * R_ves) / (U_2c * (R_ves + Thickness_vessel))

if round(check, 4) == 1:
    print(f"ok, ratio is: {round(check, 4)}")
else:
    print(f"not ok, ratio is: {round(check, 4)}")


A_c = -R_ves / Thermal_conductivity_steel * U_1c * (T_1 - T_2)
B_c = -(U_1c / h_1 * (T_1 - T_2) - T_1 + A_c * np.log(R_ves))
r = np.linspace(R_ves, R_ves + Thickness_vessel, 100)

T_c = A_c * np.log(r) + B_c

plt.figure(figsize=(10, 6))
plt.plot(r, T_c, "b-", linewidth=2, label="Temperature profile")
plt.xlabel("Radial position [m]")
plt.ylabel("Temperature [K]")
plt.title("Temperature profile inside vessel's wall")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

q_flux_in = U_1c * (T_1 - T_2)
q_flux_out = q_flux_in * R_ves / (R_ves + Thickness_vessel)
print(f"Inner thermal power: {q_flux_in / 1000:.5f} KW / m^2")
print(f"Outer thermal power: {q_flux_out / 1000:.5f} KW / m^2")

plt.figure(figsize=(10, 6))
plt.plot(r, T_profile, "b-", linewidth=2, label="With gamma radiation")
plt.plot(r, T_c, "r-", linewidth=2, label="Without gamma radiation")
plt.xlabel("Radial position [m]")
plt.ylabel("Temperature [K]")
plt.title("Temperature profile with vs without heat source")
plt.legend(fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Point 8
# Mechanical stresses
P_m = P_des * R_ves / Thickness_vessel + P_des / 2
Stress_I = S_m[idx] * 1e6

if P_m <= Stress_I:
    print(f"Good, Mariotte ({P_m:.5}) is less than yield ({Stress_I:.5})")
else:
    print(f"Not good, Mariotte ({P_m:.5}) is more than yield ({Stress_I:.5})")

# Mechanical stresses
Sigma_r_M = (
    -(R_ves**2)
    / ((R_ves + Thickness_vessel) ** 2 - R_ves**2)
    * ((R_ves + Thickness_vessel) ** 2 / r**2 - 1)
    * P_des
)
Sigma_theta_M = (
    +(R_ves**2)
    / ((R_ves + Thickness_vessel) ** 2 - R_ves**2)
    * ((R_ves + Thickness_vessel) ** 2 / r**2 + 1)
    * P_des
)

Sigma_z_M = 2 * Nu * R_ves**2 / ((R_ves + Thickness_vessel) ** 2 - R_ves**2) * P_des

# Thermal stresses
# Temperature profile integral from a to b
avg_radial_temperature, err = integrate.quad(
    functions.integrand_function,
    R_ves,
    R_ves + Thickness_vessel,
    args=(A, B, q03_prime),
)

# Initialize arrays
Sigma_r_T = np.zeros(len(r))
Sigma_theta_T = np.zeros(len(r))
Sigma_z_T = np.zeros(len(r))

# Constant factors
geom_denom = (R_ves + Thickness_vessel) ** 2 - R_ves**2
const_factor = (Alpha_T * E) / (1 - Nu)

# Calculation
for i in range(len(r)):
    radius = r[i]

    # Variable integral from a to radius
    avg_cylinder_temperature, err = integrate.quad(
        functions.integrand_function, R_ves, radius, args=(A, B, q03_prime)
    )

    # Temperature at current radius
    Temp_val = functions.Temperature_profile_prime(radius - R_ves, A, B, q03_prime)

    # Radial stress formula
    term_1 = (
        (radius**2 - R_ves**2) / (geom_denom * radius**2)
    ) * avg_radial_temperature
    term_2 = (1 / radius**2) * avg_cylinder_temperature
    Sigma_r_T[i] = const_factor * (term_1 - term_2)

    # Hoop stress formula
    term_3 = (
        (radius**2 + R_ves**2) / (geom_denom * radius**2)
    ) * avg_radial_temperature
    term_4 = (1 / radius**2) * avg_cylinder_temperature
    term_5 = Temp_val
    Sigma_theta_T[i] = const_factor * (term_3 + term_4 - term_5)

    Sigma_z_T[i] = Sigma_r_T[i] + Sigma_theta_T[i]

# Principal stresses
S_I = Sigma_r_M + Sigma_r_T
S_II = Sigma_theta_M + Sigma_theta_T
S_III = Sigma_z_M + Sigma_z_T

# Tresca criterion
diff_1 = np.abs(S_I - S_II)
diff_2 = np.abs(S_II - S_III)
diff_3 = np.abs(S_III - S_I)

Sigma_comp = np.max([diff_1, diff_2, diff_3], axis=0)
Sigma_comp_max = np.max(Sigma_comp)

Test_S_y = 2 * S_y[idx] * 1e6

if Sigma_comp_max <= Test_S_y:
    print(f"Good, Sigma_comp ({Sigma_comp_max:.5}) is less than 2 S_y ({Test_S_y:.5})")
else:
    print(
        f"Not good, Sigma_comp ({Sigma_comp_max:.5}) is more than 2 S_y ({Test_S_y:.5})"
    )
