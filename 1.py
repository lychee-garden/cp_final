# -*- coding: GB2312 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from matplotlib.collections import LineCollection
import time

c = 3e8
mu0 = 4 * np.pi * 1e-7
eps0 = 8.854e-12
eta0 = 377.0

freq = 200e6
lam = c / freq
k = 2 * np.pi / lam
omega = 2 * np.pi * freq

radius = 1.5
height = 6.0
z_bottom = 0.0
z_top = height

n_circum = 40
n_height = 30
n_radial = 12

print(f"Frequency={freq/1e6} MHz, Wavelength={lam} m")
print(f"Cylinder: radius={radius} m, height={height} m")

class MeshElement:
    def __init__(self, pos, normal, area, u_vec, v_vec):
        self.pos = np.array(pos)
        self.normal = np.array(normal)
        self.area = area
        self.u_vec = np.array(u_vec)
        self.v_vec = np.array(v_vec)

elements = []

dz = height / n_height
dphi = 2 * np.pi / n_circum

for i in range(n_height):
    z = z_bottom + (i + 0.5) * dz
    for j in range(n_circum):
        phi = j * dphi
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        pos = [x, y, z]
        normal = [np.cos(phi), np.sin(phi), 0]
        area = (radius * dphi) * dz
        u_vec = [0, 0, 1]
        v_vec = [-np.sin(phi), np.cos(phi), 0]
        elements.append(MeshElement(pos, normal, area, u_vec, v_vec))

def create_cap_mesh(z_level, is_top):
    dr = radius / n_radial
    nz = 1 if is_top else -1
    normal = [0, 0, nz]
    for i in range(n_radial):
        r_inner = i * dr
        r_outer = (i + 1) * dr
        r_mid = (r_inner + r_outer) / 2
        current_circum_n = max(4, int(n_circum * (r_mid / radius)))
        dphi_cap = 2 * np.pi / current_circum_n
        for j in range(current_circum_n):
            phi = j * dphi_cap + (0.5 * dphi_cap if i%2 else 0)
            x = r_mid * np.cos(phi)
            y = r_mid * np.sin(phi)
            pos = [x, y, z_level]
            area = (np.pi * (r_outer**2 - r_inner**2)) / current_circum_n
            u_vec = [np.cos(phi), np.sin(phi), 0]
            v_vec = [-np.sin(phi), np.cos(phi), 0]
            elements.append(MeshElement(pos, normal, area, u_vec, v_vec))

create_cap_mesh(z_bottom, is_top=False)
create_cap_mesh(z_top, is_top=True)

N_elem = len(elements)
N_unknowns = 2 * N_elem
print(f"Mesh: {N_elem} elements, {N_unknowns}x{N_unknowns} matrix")

Z = np.zeros((N_unknowns, N_unknowns), dtype=complex)
V = np.zeros(N_unknowns, dtype=complex)

print("Filling impedance matrix...")
start_time = time.time()

positions = np.array([e.pos for e in elements])
areas = np.array([e.area for e in elements])
basis_vecs = []
for e in elements:
    basis_vecs.append(e.u_vec)
    basis_vecs.append(e.v_vec)
basis_vecs = np.array(basis_vecs)

const_factor = 1j * omega * mu0 / (4 * np.pi)

for m in range(N_elem):
    r_m = positions[m]
    for i_pol in range(2):
        row = 2 * m + i_pol
        t_vec = elements[m].u_vec if i_pol == 0 else elements[m].v_vec
        for n in range(N_elem):
            r_n = positions[n]
            dist = np.linalg.norm(r_m - r_n)
            for j_pol in range(2):
                col = 2 * n + j_pol
                s_vec = elements[n].u_vec if j_pol == 0 else elements[n].v_vec
                if m != n:
                    g = np.exp(-1j * k * dist) / dist
                    elem_z = const_factor * np.dot(t_vec, s_vec) * g * areas[n]
                else:
                    a_eq = np.sqrt(areas[n] / np.pi)
                    elem_z = const_factor * (2 * np.pi * a_eq) * (1 - 0.5j * k * a_eq)
                Z[row, col] = elem_z
    if m % 50 == 0:
        print(f"\rProgress: {m}/{N_elem}", end="")

print(f"\nMatrix filled in {time.time() - start_time:.2f}s")

E0_amp = 1.0
k_vec_inc = np.array([0, 0, -k])

for m in range(N_elem):
    r = positions[m]
    phase = np.exp(-1j * np.dot(k_vec_inc, r))
    E_inc_vector = np.array([1.0, 0.0, 0.0]) * E0_amp * phase
    V[2*m] = np.dot(E_inc_vector, elements[m].u_vec)
    V[2*m + 1] = np.dot(E_inc_vector, elements[m].v_vec)

print("Solving linear system...")
I_coeffs = scipy.linalg.solve(Z, V)
print("Done.")

def calculate_rcs(theta_list, phi=0):
    rcs_values = []
    r_const = 1j * k * eta0 / (4 * np.pi)
    for theta_deg in theta_list:
        theta = np.deg2rad(theta_deg)
        rx = np.sin(theta) * np.cos(phi)
        ry = np.sin(theta) * np.sin(phi)
        rz = np.cos(theta)
        r_hat = np.array([rx, ry, rz])
        E_theta = 0
        E_phi = 0
        for n in range(N_elem):
            J_n = I_coeffs[2*n] * elements[n].u_vec + I_coeffs[2*n+1] * elements[n].v_vec
            phase_factor = np.exp(1j * k * np.dot(positions[n], r_hat))
            theta_hat = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
            phi_hat = np.array([-np.sin(phi), np.cos(phi), 0])
            E_theta += np.dot(J_n, theta_hat) * phase_factor * areas[n]
            E_phi += np.dot(J_n, phi_hat) * phase_factor * areas[n]
        E_s_mag = np.sqrt(abs(E_theta)**2 + abs(E_phi)**2) * k * eta0 / (4*np.pi)
        sigma = 4 * np.pi * E_s_mag**2
        rcs_db = 10 * np.log10(sigma + 1e-20)
        rcs_values.append(rcs_db)
    return rcs_values

thetas = np.linspace(0, 180, 1801)
rcs_result = calculate_rcs(thetas, phi=np.pi/2)

plt.figure(figsize=(10, 6))
plt.plot(thetas, rcs_result, label='MoM Calculated', linewidth=2)
plt.title(f'Bistatic RCS of PEC Cylinder (Freq={freq/1e6} MHz)\nRadius={radius}m, Height={height}m')
plt.xlabel('Theta (degrees)')
plt.ylabel('RCS (dBsm)')
plt.grid(True)
plt.legend()
plt.axvline(x=90, color='r', linestyle='--', alpha=0.3, label='Broadside')
plt.text(10, min(rcs_result)+5, 'Backscatter\n(Reflection)', ha='center')
plt.text(180, min(rcs_result)+5, 'Forward Scatter\n(Shadowing)', ha='center')
plt.tight_layout()
plt.show()

theta_rad = np.deg2rad(thetas)
theta_full = np.concatenate([theta_rad, 2*np.pi - theta_rad[-2:0:-1]])
rcs_full = np.concatenate([rcs_result, rcs_result[-2:0:-1]])
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
points = np.array([theta_full, rcs_full]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(vmin=np.min(rcs_full), vmax=np.max(rcs_full))
lc = LineCollection(segments, cmap='jet', norm=norm)
lc.set_array(rcs_full)
lc.set_linewidth(2)
ax.add_collection(lc)

# change the y axis range
ax.set_ylim(0, np.max(rcs_full))
cbar = fig.colorbar(lc, ax=ax, pad=0.1)
cbar.set_label('RCS (dBsm)')
ax.grid(True, alpha=0.5, color='gray')
ax.set_title(f'Bistatic RCS Polar Plot (Freq={freq/1e6} MHz)', va='bottom')
ax.text(0, np.max(rcs_full)+3, '0° (Top/Backscatter)', ha='center', color='red', fontsize=8)
ax.text(np.pi, np.max(rcs_full)+3, '180° (Bottom/Forward)', ha='center', color='blue', fontsize=8)
plt.tight_layout()
plt.show()