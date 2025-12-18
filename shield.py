# import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

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
Sigma_T_1 = 0.56  # From graph
Sigma_T_2 = 0.08  # From graph
a = 2.75

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
idx = 12
Thickness = 0.157468271335  # m
h_1 = 7498.1  # W / m^2 K
Sigma_Lame = 93930646.91706014  # Pa


def Tempreture_profile(x):
    return (
        -q03 / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
        + A * x
        + B
    )


# Point 4
q03_prime = (3 * S_m[idx] * 1e6 - Sigma_Lame) / (
    (Sigma_T_1 * Alpha_T * E) / (Thermal_conductivity_steel * (1 - Nu) * Mu_steel**2)
)

print(f"{q03_prime}")
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

x = np.linspace(0, Shield_thickness, 100)
T_shield = (
    -q03 / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
    + A * x
    + B
)

T_shield_avg = np.average(T_shield)
T_shield_max = np.max(T_shield)
T_shield_min = np.min(T_shield)
print(f"{T_shield_avg}")
print(f"{T_shield_min}")
print(f"{T_shield_max}")

idx_max = np.argmax(T_shield)
pos = x[idx_max]
print(f"{pos * 100}")

plt.figure(figsize=(6, 5))
plt.plot(x, T_shield, label="Temperature progile in the thermal shield")
plt.xlabel("x (m)")
plt.ylabel("Temperature (K)")
plt.title("Temperature profile in the thermal shield")
plt.grid()
plt.show()

# Point 6
# Mechanical stresses
P_m = P_des * R_ves / Thickness + P_des / 2
Stress_I = S_m[idx] * 1e6
if P_m <= Stress_I:
    print(f"Good, {P_m:.5} is less than {Stress_I:.5}")
else:
    print(f"Not good, {P_m:.5} is more than {Stress_I:.5}")

# Thermal stresses
Q = (Sigma_T_2 * Alpha_T * E * q03) / (
    Thermal_conductivity_steel * (1 - Nu) * Mu_steel**2
)

# --- DEBUGGED SECTION FOR POINT 6 ---

# 1. SETUP GEOMETRY (Absolute Radius)
# We assume 'a' is the inner radius of the shield.
# If the shield is inside the vessel, check if a = R_ves - Shield_thickness or similar.
# Based on your variable 'R_ves', let's assume the shield starts at 'a' and goes out.
b = a + Shield_thickness

# r must be absolute radius from center (a -> b), NOT 0 -> thickness
r = np.linspace(a, b, 100)


# 2. DEFINE INTEGRAND FUNCTION
# The integral requires T(rho) * rho.
# Since T_profile takes 'x' (0 to thickness), we pass (rho - a).
def integrand_function(rho):
    x_local = rho - a
    return Tempreture_profile(x_local) * rho


# 3. CALCULATE CONSTANT INTEGRAL (Total integral from a to b)
integral_ab, err = integrate.quad(integrand_function, a, b)

# Initialize arrays
Sigma_r = np.zeros(len(r))
Sigma_theta = np.zeros(len(r))
Sigma_Lame = np.zeros(len(r))

# 4. CONSTANTS FOR LOOP
# Correct denominator is (b^2 - a^2)
geom_denom = b**2 - a**2
const_factor = (Alpha_T * E) / (1 - Nu)

print("Calculating thermal stresses...")

for i in range(len(r)):
    radius = r[i]

    # Calculate variable integral from a to current radius
    integral_ar, err = integrate.quad(integrand_function, a, radius)

    # Get Temperature at this radius
    Temp_val = Tempreture_profile(radius - a)

    # Radial Stress Formula
    term_1 = ((radius**2 - a**2) / (geom_denom * radius**2)) * integral_ab
    term_2 = (1 / radius**2) * integral_ar
    Sigma_r[i] = const_factor * (term_1 - term_2)

    # Tangential (Hoop) Stress Formula
    term_3 = ((radius**2 + a**2) / (geom_denom * radius**2)) * integral_ab
    term_4 = (1 / radius**2) * integral_ar
    term_5 = Temp_val
    Sigma_theta[i] = const_factor * (term_3 + term_4 - term_5)

    # Lame Stress (Von Mises equivalent or Tresca)
    # Usually absolute difference |sigma_theta - sigma_r|
    Sigma_Lame[i] = abs(Sigma_theta[i] - Sigma_r[i])

# 5. CHECK RESULTS
Sigma_Lame_max = np.max(Sigma_Lame)
Limit_Stress = 3 * S_m[idx] * 1e6

print(f"Max Sigma_Lame: {Sigma_Lame_max / 1e6:.2f} MPa")
print(f"Limit Stress: {Limit_Stress / 1e6:.2f} MPa")

if Sigma_Lame_max <= Limit_Stress:
    print("Good: Thermal stress is within limits.")
else:
    print("Bad: Thermal stress exceeds limits.")
