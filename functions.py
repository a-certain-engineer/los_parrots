import numpy as np
import scipy.constants as constants

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

# contants
g = 9.806  # m / s
eV = 1.6e-19  # J

# geometry
D_e = D_ves - D_bar  # m
R_bar = D_bar / 2
R_ves = D_ves / 2

T_1 = T_1 + Kelvin
T_2 = T_2 + Kelvin


def Temperature_profile(x, A, B, q_volumetric):
    """Calculates the temperature T(x) at a specific depth 'x' inside the vessel wall with the minimized heat flux q03_prime.
    T(x) = - (q0 / (k * mu^2)) * exp(-mu * x) + A * x + B
    """
    return (
        -q_volumetric
        / (Mu_steel**2 * Thermal_conductivity_steel)
        * np.exp(-Mu_steel * x)
        + A * x
        + B
    )


def integrand_function(rho, A, B, q03_prime):
    """Calculates T(r) * r"""
    x_local = rho - a
    return Temperature_profile(x_local, A, B, q03_prime) * rho


def solve_coefficients(t, h_1, h_2, q_volumetric):
    """Solves the linear system of equations to find integration constants A and B given by the following BC:
    1. Inner BC (Convection): -k * dT/dx = h1 * (T_fluid - T_wall) at x=0
    2. Outer BC (Convection + Insulation): Heat flux through wall = Heat flux through insulation."""
    # Terms for A
    term1 = (
        -(q_volumetric * Thickness_insulation)
        / (Mu_steel * Thermal_conductivity_ins)
        * np.exp(-Mu_steel * t)
    )
    term2 = -(Thickness_insulation / (h_2 * Thermal_conductivity_ins)) * np.exp(
        -Mu_steel * t
    )
    term3 = (q_volumetric / (Mu_steel**2 * Thermal_conductivity_steel)) * np.exp(
        -Mu_steel * t
    )
    term4 = -T_1
    term5 = -q_volumetric / (Mu_steel**2 * Thermal_conductivity_steel)
    term6 = -q_volumetric / (Mu_steel * h_1)

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

    # Terms for B
    alpha2 = -(Thermal_conductivity_steel / h_1)
    beta2 = 1
    gamma2 = (
        T_1
        + q_volumetric / (Mu_steel**2 * Thermal_conductivity_steel)
        + q_volumetric / (Mu_steel * h_1)
    )

    # Matrix for the system
    A_mat = np.array([[alpha1, beta1], [alpha2, beta2]])

    # Coefficients vector
    b_vec = np.array([gamma1, gamma2])

    # Solve matrix equation
    A, B = np.linalg.solve(A_mat, b_vec)

    return (A, B)


def find_index(T_des):
    """Finds the index corresponding to the given temperature of Table 2"""
    T_des_celsius = T_des - Kelvin

    # Get index of last temperature if it's higher than the highest temperature in the table
    if T_des_celsius > Temperature[-1]:
        return len(Temperature) - 1

    # Get index of temperature in the table
    idx = np.where(
        Temperature >= T_des_celsius, Temperature - T_des_celsius, np.inf
    ).argmin()

    return idx
