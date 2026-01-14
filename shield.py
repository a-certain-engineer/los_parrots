import functions
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.constants as constants

plt.rc("font", size=13)  # Increase size of axis numbers and titles

# variables
P_des = 82.5  # bar
T_fluid = 214  # °C
D_ves = 3.0  # m
Phi_0 = 1.5e13  # 1 / cm2 s
Energy_gamma = 6.0e6  # eV
Build_up = 1.4
Mu_steel = 24  # 1 / m
Thermal_conductivity_steel = 48.1  #  W / m K
T_1 = 214.0  # °C
T_2 = 70.0  # °C
Kelvin = constants.zero_Celsius  # K
E = 177.0e9  # Pa
Nu = 0.3
Alpha_T = 1.7e-5  # 1 / K
Sigma_T = 0.56  # From graph
a = 2.75 / 2  # m

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

# Geometry calculations
R_ves = D_ves / 2

# Unit conversion
Energy_gamma = Energy_gamma * constants.eV
Phi_0 = Phi_0 * 1e4
q03 = Energy_gamma * Phi_0 * Mu_steel * Build_up

# Results form vessel analysis
idx_vessel = 12
Thickness_vessel = 0.161471789242587  # m
h_1 = 7498  # W / m^2 K
Sigma_Lame = 89224458.10435033  # Pa


# Point 4 - Shield thickness
q03_prime = (
    (Thermal_conductivity_steel * (1 - Nu) * Mu_steel**2)
    / (Sigma_T * Alpha_T * E)
    * (3 * S_m[idx_vessel] * 1e6 - Sigma_Lame)
)

Shield_thickness = -1 / Mu_steel * np.log(q03_prime / q03)

print(f"Thermal shield thickness: {Shield_thickness * 100:.2f} cm")


# Point 5 - Radial temperature profile
# Constants from boundary condition
A = (
    np.exp(-Mu_steel * Shield_thickness)
    * ((q03 * h_1) / (Mu_steel**2 * Thermal_conductivity_steel) - q03 / Mu_steel)
    - ((q03 * h_1) / (Mu_steel**2 * Thermal_conductivity_steel) + q03 / Mu_steel)
) / (2 * Thermal_conductivity_steel + Shield_thickness * h_1)
B = (
    T_fluid
    + q03 / (Mu_steel**2 * Thermal_conductivity_steel)
    + (Thermal_conductivity_steel * A) / h_1
    + q03 / (Mu_steel * h_1)
)

# Temperature profile inside the shield
x = np.linspace(0, Shield_thickness, 100)
T_shield = (
    -q03 / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
    + A * x
    + B
)

plt.figure(figsize=(10, 6))
plt.plot(
    x + a,
    T_shield,
    "b-",
    linewidth=2,
    label="Temperature profile in the thermal shield",
)
plt.xlabel("Radial position [m]")
plt.ylabel("Temperature [K]")
plt.title("Temperature profile in the thermal shield")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Extract values from the temperature profile
T_shield_avg = np.average(T_shield)
T_shield_max = np.max(T_shield)
T_shield_min = np.min(T_shield)

# Find position of maximum temperature
idx_max = np.argmax(T_shield)
pos = x[idx_max]

print(f"Inner surface shield temperature: {T_shield[0] - Kelvin:.0f} C")
print(f"Outer surface shield temperature: {T_shield[-1] - Kelvin:.0f} C")
print(f"Maximum shield temperature: {T_shield_max - Kelvin:.0f} C")
print(f"Position of maximum temperature in the shield: {pos * 1e2:.2f} cm")


# Point 6 - Thermal stresses evaluation
# Mechanical stresses
P_m = P_des * R_ves / Thickness_vessel + P_des / 2
Stress_I = S_m[idx_vessel] * 1e6
if P_m <= Stress_I:
    print(f"Good, {P_m / 1e6:.3f} is less than {Stress_I / 1e6:.3f}")
else:
    print(f"Not good, {P_m / 1e6:.3f} is more than {Stress_I / 1e6:.3f}")


# Point 3 - Design conditions
# Constant factors
b = a + Shield_thickness
geom_denom = b**2 - a**2
const_factor = (Alpha_T * E) / (1 - Nu)

r = np.linspace(a, b, 100)

# Average temperature inside shield
avg_temperature, err = integrate.quad(
    functions.integrand_function, a, b, args=(A, B, q03)
)

# Initialize arrays
Sigma_r = np.zeros(len(r))
Sigma_theta = np.zeros(len(r))
Sigma_thermal = np.zeros(len(r))

# Find stress intensity index
idx_S_m = functions.find_index(T_shield_avg)

for i in range(len(r)):
    radius = r[i]

    # Fictitius temperature profile calculation
    integral_ar, err = integrate.quad(
        functions.integrand_function, a, radius, args=(A, B, q03)
    )

    # Temperature at current radius
    Temp_val = functions.Temperature_profile(radius - a, A, B, q03)

    # Radial stress
    term_1 = ((radius**2 - a**2) / (geom_denom * radius**2)) * avg_temperature
    term_2 = (1 / radius**2) * integral_ar
    Sigma_r[i] = const_factor * (term_1 - term_2)

    # Hoop stress
    term_3 = ((radius**2 + a**2) / (geom_denom * radius**2)) * avg_temperature
    term_4 = (1 / radius**2) * integral_ar
    term_5 = Temp_val
    Sigma_theta[i] = const_factor * (term_3 + term_4 - term_5)

    # Lamé Stress
    Sigma_thermal[i] = abs(Sigma_theta[i] - Sigma_r[i])

# Maximum Lamé stress and stress intensity
Sigma_thermal_max = np.max(Sigma_thermal)
Limit_Stress = 3 * S_m[idx_S_m] * 1e6

# Verify that thermal stresses are below 3S_m
if Sigma_thermal_max <= Limit_Stress:
    print(
        f"Verified: Q = {Sigma_thermal_max / 1e6:.2f} MPa <= 3 S_m = {Limit_Stress / 1e6:.2f} MPa"
    )
else:
    print(
        f"Not verified: Q = {Sigma_thermal_max / 1e6:.2f} MPa > 3 S_m = {Limit_Stress / 1e6:.2f} MPa"
    )
