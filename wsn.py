import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import Delaunay

class wsn:
    def __init__(self, area_size, sensing_radius, r_error, grid_resolution=1.0):
        self.Lx, self.Ly = area_size
        self.rc = sensing_radius
        self.re = r_error

        # parameter for eq 2
        self.alpha_const = 0.5
        self.beta = 0.5
        self.lambda_param = 0.5

        # eq 4
        x = np.arange(0, self.Lx, grid_resolution)
        y = np.arange(0, self.Ly, grid_resolution)
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        self.grid_points = np.column_stack((self.grid_x.ravel(), self.grid_y.ravel()))
        self.total_points = self.grid_points.shape[0]

    def van_der_corput(self, n, base=2):
        # eq 10
        vdc = 0
        denom = 1
        while n > 0:
            denom *= base
            remainder = n % base
            n //= base
            vdc += remainder / denom

        return vdc

    def generate_hammersley_positions(self, num_particles, dim, bounds):
        # eq 10
        positions = np.zeros((num_particles, dim))

        for i in range(num_particles):
            for d in range(dim):
                base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                               59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

                base = base_primes[d % len(base_primes)]

                if d == 0:
                    val = (i + 1) / num_particles
                else:
                    val = self.van_der_corput(i + 1, base)

                positions[i, d] = bounds[0] + val * (bounds[1] - bounds[0])

        return positions

    def get_triangle_area(self, a, b, c):
        # eq 13
        return 0.5 * np.abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))

    def calculate_delaunay_holes(self, node_positions):
        tri = Delaunay(node_positions)

        total_hole_area = 0
        hole_triangles = []

        for simplex in tri.simplices:
            pts = node_positions[simplex]
            A, B, C = pts[0], pts[1], pts[2]

            a = np.linalg.norm(B - C)
            b = np.linalg.norm(A - C)
            c = np.linalg.norm(A - B)

            area = self.get_triangle_area(A, B, C)

            if area == 0: continue

            circumradius = (a * b * c) / (4 * area)

            if circumradius > self.rc:
                total_hole_area += area
                hole_triangles.append(pts)

        return total_hole_area, hole_triangles

    def calculate_coverage_rate(self, node_positions):
        # eq 4
        diff = self.grid_points[:, np.newaxis, :] - node_positions[np.newaxis, :, :]
        # eq 1
        dists = np.linalg.norm(diff, axis=2)

        probs = np.zeros_like(dists)

        mask_sure = dists <= (self.rc - self.re)
        mask_zero = dists >= (self.rc + self.re)
        mask_decay = (~mask_sure) & (~mask_zero)

        probs[mask_sure] = 1.0

        # decay calculation
        if np.any(mask_decay):
            d_decay = dists[mask_decay]
            # eq 2
            alpha = d_decay - (self.rc - self.re)
            probs[mask_decay] = np.exp(-self.lambda_param * (alpha ** self.beta))

        not_probs = 1.0 - probs

        prob_not_probs = np.prod(not_probs, axis=1)

        p_joint = 1.0 - prob_not_probs

        coverage_rate = np.sum(p_joint) / self.total_points

        return coverage_rate

    def plot_results(self, nodes, area, radius, history, coverage):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Convergence curve
        ax1.plot(history, 'b-o', linewidth=2)
        ax1.set_title('Convergence Curve (VASF-PSO)')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Converage Rate')
        ax1.grid(True)

        # 2. Final Deployment
        ax2.set_xlim(0, area[0])
        ax2.set_ylim(0, area[1])
        ax2.set_aspect('equal')
        ax2.set_title(f"VASF-PSO Coverages {coverage*100:.2f}%")
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')

        _, hole_triangles = self.calculate_delaunay_holes(nodes)
        for tri_pts in hole_triangles:
            polygon = patches.Polygon(tri_pts, closed=True,
                                      edgecolor='blue', facecolor='blue',
                                      alpha=0.1, linestyle='--')
            ax2.add_patch(polygon)

        for node in nodes:
            circle = plt.Circle(node, radius, color='g', alpha=0.3, fill=True)
            ax2.add_artist(circle)
            circle_border = plt.Circle(node, radius, color='k', alpha=0.5, fill=False)
            ax2.add_artist(circle_border)
            ax2.plot(node[0], node[1], 'k.', markersize=5)

        tri = Delaunay(nodes)
        ax2.triplot(nodes[:,0], nodes[:,1], tri.simplices.copy(), color='black', alpha=0.3, linewidth=0.5)

        plt.tight_layout()
        plt.show()