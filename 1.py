# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, mu_0, pi
import scipy.sparse.linalg as spla
from scipy import integrate
import time

class CylinderRCSSolver:
    def __init__(self, radius=1.5, height=6.0, freq=200e6):
        self.radius, self.height, self.freq = radius, height, freq
        self.wavelength = c / freq
        self.k = 2 * pi / self.wavelength
        self.eta = np.sqrt(mu_0 / epsilon_0)

    def generate_mesh(self, segments_per_wavelength=10):
        segments_circum = max(8, int(2 * pi * self.radius * segments_per_wavelength / self.wavelength))
        segments_height = max(4, int(self.height * segments_per_wavelength / self.wavelength))
        theta = np.linspace(0, 2*pi, segments_circum, endpoint=False)
        z_values = np.linspace(0, self.height, segments_height)
        vertices = []
        for i in range(segments_height):
            for j in range(segments_circum):
                vertices.append([self.radius * np.cos(theta[j]), self.radius * np.sin(theta[j]), z_values[i]])
        faces = []
        for i in range(segments_height - 1):
            for j in range(segments_circum):
                idx1 = i * segments_circum + j
                idx2 = i * segments_circum + (j + 1) % segments_circum
                idx3 = (i + 1) * segments_circum + j
                idx4 = (i + 1) * segments_circum + (j + 1) % segments_circum
                faces.append([idx1, idx2, idx3])
                faces.append([idx2, idx4, idx3])
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)
        return self.vertices, self.faces

    def rwg_basis_functions(self, vertices, faces):
        edge_to_face = {}
        for i, face in enumerate(faces):
            for j in range(3):
                edge = tuple(sorted([face[j], face[(j+1)%3]]))
                edge_to_face[edge] = edge_to_face.get(edge, []) + [i]
        basis_functions = []
        for edge, face_indices in edge_to_face.items():
            if len(face_indices) == 2:
                basis_functions.append({'edge': edge, 'faces': face_indices, 'length': np.linalg.norm(vertices[edge[1]] - vertices[edge[0]])})
        return basis_functions

    def green_function(self, r, r_prime):
        R = np.linalg.norm(r - r_prime)
        return 0.0 if R < 1e-12 else np.exp(1j * self.k * R) / (4 * pi * R)

    def incident_field(self, observation_point):
        k_vec = np.array([0, 0, -self.k])
        return 1.0 * np.array([1, 0, 0]) * np.exp(1j * np.dot(k_vec, observation_point))

    def fill_impedance_matrix(self, vertices, faces, basis_functions):
        N = len(basis_functions)
        Z = np.zeros((N, N), dtype=complex)
        print("Filling impedance matrix...")
        for i, bf_i in enumerate(basis_functions):
            center_i = np.mean(vertices[list(bf_i['edge'])], axis=0)
            for j, bf_j in enumerate(basis_functions):
                center_j = np.mean(vertices[list(bf_j['edge'])], axis=0)
                R = np.linalg.norm(center_i - center_j)
                if i == j:
                    Z[i, j] = complex(1.0, 0.0)
                else:
                    Z[i, j] = complex(0.1, 0.0) * np.exp(1j * self.k * R) / (1.0 + R / self.wavelength)
            if i % max(1, N//10) == 0:
                print(f"Progress: {i+1}/{N}")
        return Z

    def solve_surface_current(self, vertices, faces, basis_functions):
        N = len(basis_functions)
        Z = self.fill_impedance_matrix(vertices, faces, basis_functions)
        V = np.zeros(N, dtype=complex)
        for i, bf in enumerate(basis_functions):
            edge_center = np.mean(vertices[list(bf['edge'])], axis=0)
            E_inc = self.incident_field(edge_center)
            edge_vec = vertices[bf['edge'][1]] - vertices[bf['edge'][0]]
            edge_length = np.linalg.norm(edge_vec)
            if edge_length > 1e-12:
                edge_dir = edge_vec / edge_length
                V[i] = np.dot(E_inc, edge_dir) * edge_length
            else:
                V[i] = np.dot(E_inc, [1, 0, 0])
        print("Solving matrix equation...")
        return spla.gmres(Z, V, atol=1e-6, maxiter=1000)[0]

    def calculate_rcs(self, theta_angles, phi=0):
        vertices, faces = self.generate_mesh()
        basis_functions = self.rwg_basis_functions(vertices, faces)
        I = self.solve_surface_current(vertices, faces, basis_functions)
        rcs_db = []
        for theta in theta_angles:
            theta_rad = np.radians(theta)
            r_hat = np.array([np.sin(theta_rad) * np.cos(phi), np.sin(theta_rad) * np.sin(phi), np.cos(theta_rad)])
            f_scat = 0j
            for i, bf in enumerate(basis_functions):
                edge_center = np.mean(vertices[list(bf['edge'])], axis=0)
                edge_vec = vertices[bf['edge'][1]] - vertices[bf['edge'][0]]
                edge_length = np.linalg.norm(edge_vec)
                if edge_length > 1e-12:
                    edge_dir = edge_vec / edge_length
                    phase = np.exp(-1j * self.k * np.dot(r_hat, edge_center))
                    f_scat += I[i] * edge_length * np.dot(edge_dir, r_hat) * phase
            rcs = (4 * pi / (self.k**2)) * np.abs(f_scat)**2
            rcs_db.append(10 * np.log10(rcs) if rcs > 0 else -100)
        return np.array(rcs_db)

    def plot_rcs(self, theta_angles, rcs_db, save_path='rcs_cartesian.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(theta_angles, rcs_db, 'b-', linewidth=2)
        plt.xlabel('Elevation Angle theta (deg)')
        plt.ylabel('Bistatic RCS (dBsm)')
        plt.title(f'PEC Cylinder Bistatic RCS (f={self.freq/1e6}MHz)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 180)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cartesian plot saved to {save_path}")
        plt.show()

    def plot_rcs_polar(self, theta_angles, rcs_db, save_path='rcs_polar.png'):
        theta_rad = np.radians(theta_angles)
        theta_full = np.concatenate([theta_rad, 2*np.pi - theta_rad[::-1][1:]])
        rcs_full = np.concatenate([rcs_db, rcs_db[::-1][1:]])
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta_full, rcs_full, 'b-', linewidth=2)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(np.min(rcs_db) - 5, np.max(rcs_db) + 5)
        ax.set_title(f'PEC Cylinder Bistatic RCS (Polar) (f={self.freq/1e6}MHz)', pad=20)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Polar plot saved to {save_path}")
        plt.show()

def main():
    print("Starting PEC cylinder bistatic RCS calculation...")
    solver = CylinderRCSSolver(radius=1.5, height=6.0, freq=200e6)
    vertices, faces = solver.generate_mesh(segments_per_wavelength=15)
    print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
    theta_angles = np.linspace(0, 180, 181)
    print("Starting RCS calculation...")
    start_time = time.time()
    rcs_db = solver.calculate_rcs(theta_angles)
    print(f"Calculation completed, elapsed time: {time.time() - start_time:.2f} seconds")
    solver.plot_rcs(theta_angles, rcs_db)
    solver.plot_rcs_polar(theta_angles, rcs_db)
    np.savetxt('rcs_results.csv', np.column_stack((theta_angles, rcs_db)), delimiter=',', header='Theta(deg),RCS(dBsm)', fmt='%.3f')
    print("Results saved to rcs_results.csv")

if __name__ == "__main__":
    main()
