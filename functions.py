import numpy as np

# variables
P_des = 85  # bar
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
Sigma_T_1 = 0.56  # From graph
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

# contants
g = 9.806  # m / s
eV = 1.6e-19  # J

# geometry
D_e = D_ves - D_bar  # m
R_bar = D_bar / 2
R_ves = D_ves / 2

T_1 = T_1 + Kelvin
T_2 = T_2 + Kelvin


def Temperature_profile_prime(x, A, B, q03_prime):
    return (
        -q03_prime / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
        + A * x
        + B
    )


def Temperature_profile(x, A, B, q03):
    return (
        -q03 / (Mu_steel**2 * Thermal_conductivity_steel) * np.exp(-Mu_steel * x)
        + A * x
        + B
    )


def integrand_function(rho, A, B, q03_prime):
    x_local = rho - R_ves
    return Temperature_profile_prime(x_local, A, B, q03_prime) * rho


def solve_coefficients(t, h_1, h_2, q03):
    # Linear system for A and B
    # Equation for A
    term1 = (
        -(q03 * Thickness_insulation)
        / (Mu_steel * Thermal_conductivity_ins)
        * np.exp(-Mu_steel * t)
    )
    term2 = -(Thickness_insulation / (h_2 * Thermal_conductivity_ins)) * np.exp(
        -Mu_steel * t
    )
    term3 = (q03 / (Mu_steel**2 * Thermal_conductivity_steel)) * np.exp(-Mu_steel * t)
    term4 = -T_1
    term5 = -q03 / (Mu_steel**2 * Thermal_conductivity_steel)
    term6 = -q03 / (Mu_steel * h_1)

    numerator = term1 + term2 + term3 + term4 + term5 + term6
    denominator = (
        t
        + Thermal_conductivity_steel / h_1
        + Thermal_conductivity_steel / h_2
        + (Thermal_conductivity_steel * Thickness_insulation) / Thermal_conductivity_ins
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

    # Solve matrix equation
    A, B = np.linalg.solve(A_mat, b_vec)

    return (A, B)


def solve_coefficients_prime(t, h_1, h_2, q03_prime):
    term1 = (
        -(q03_prime * Thickness_insulation)
        / (Mu_steel * Thermal_conductivity_ins)
        * np.exp(-Mu_steel * t)
    )
    term2 = -(Thickness_insulation / (h_2 * Thermal_conductivity_ins)) * np.exp(
        -Mu_steel * t
    )
    term3 = (q03_prime / (Mu_steel**2 * Thermal_conductivity_steel)) * np.exp(
        -Mu_steel * t
    )
    term4 = -T_1
    term5 = -q03_prime / (Mu_steel**2 * Thermal_conductivity_steel)
    term6 = -q03_prime / (Mu_steel * h_1)

    numerator = term1 + term2 + term3 + term4 + term5 + term6
    denominator = (
        t
        + Thermal_conductivity_steel / h_1
        + Thermal_conductivity_steel / h_2
        + (Thermal_conductivity_steel * Thickness_insulation) / Thermal_conductivity_ins
    )
    alpha1 = denominator
    beta1 = 0
    gamma1 = numerator

    # Equation for B
    alpha2 = -(Thermal_conductivity_steel / h_1)
    beta2 = 1
    gamma2 = (
        T_1
        + q03_prime / (Mu_steel**2 * Thermal_conductivity_steel)
        + q03_prime / (Mu_steel * h_1)
    )

    # Matrices for the system
    A_mat = np.array([[alpha1, beta1], [alpha2, beta2]])

    # Coefficients vector
    b_vec = np.array([gamma1, gamma2])

    # Solve matrix equation
    A, B = np.linalg.solve(A_mat, b_vec)

    return (A, B)


def find_index(T_des):
    T_des_celsius = T_des - Kelvin

    if T_des_celsius > Temperature[-1]:
        return len(Temperature) - 1

    idx = np.where(
        Temperature >= T_des_celsius, Temperature - T_des_celsius, np.inf
    ).argmin()

    return idx
