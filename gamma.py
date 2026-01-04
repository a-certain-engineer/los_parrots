import numpy as np
import matplotlib.pyplot as plt

# Given data
Initial_flux = 1.5e13  # photons/cm²·s
Avg_gamma_en = 6.0e6 * 1.6e-19  # MeV to J (6 MeV * 1.6e-19 J/eV)
mu = 24  # m⁻¹

# Geometry
R_internal = 1.5  # m
t = 16.145e-2  # m (vessel thickness)
R_external = R_internal + t

Initial_flux_m2 = Initial_flux * 1e4  # photons/m²·s


N_steps = 300
r = np.linspace(R_internal, R_external, N_steps)

# Volumetric heat source calculation
def volumetric_heat_source(r, I0, mu, E_gamma):
    """
    Calculate volumetric heat source due to gamma absorption
    q'''(r) = μ * I(r) * E_gamma
    where I(r) = I0 * exp(-μ*(r - R_internal))
    """
    B=np.linspace(1, 1.4,N_steps)
    I_r = I0 *B* np.exp(-mu * (r - R_internal))
    q_vol = mu * I_r * E_gamma
    return q_vol


q_vol = volumetric_heat_source(r, Initial_flux_m2, mu, Avg_gamma_en)


q0_inner = q_vol[0]  
q0_outer = q_vol[-1]  

print(f"Volumetric heat source at inner surface (r = {R_internal} m): {q0_inner/1e6:.2f} MW/m³")
print(f"Volumetric heat source at outer surface (r = {R_external:.3f} m): {q0_outer/1e6:.4f} MW/m³")

# Plot the radial profile
plt.figure(figsize=(10, 6))
plt.plot(r, q_vol/1e6, 'b-', linewidth=2)
plt.xlabel('Radial position [m]')
plt.ylabel('Volumetric heat source [MW/m³]')
plt.title('Radial profile of volumetric heat source in vessel wall')
plt.grid(True, alpha=0.3)

# Add some key points to the plot
plt.scatter([R_internal, R_external], [q0_inner/1e6, q0_outer/1e6], 
           color='red', zorder=5, s=50)
plt.annotate(f'Inner surface\n{q0_inner/1e6:.1f} MW/m³', 
            xy=(R_internal, q0_inner/1e6), 
            xytext=(R_internal+0.02, q0_inner/1e6+0.5-1))
plt.annotate(f'Outer surface\n{q0_outer/1e6:.4f} MW/m³', 
            xy=(R_external, q0_outer/1e6), 
            xytext=(R_external-0.02, q0_outer/1e6+0.1+0.3))

plt.tight_layout()
plt.show()
