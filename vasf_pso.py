import numpy as np
from tqdm import tqdm

class vasf_pso:
    def __init__(self, num_particles, dim, bounds, max_iter, fitness_function):
        self.num_particles = num_particles
        self.dim = dim
        self.bounds = bounds
        self.max_iter = max_iter
        self.fitness_function = fitness_function

        # vasf_pso paramenters
        self.w_min = 0.2
        self.w_max = 0.9
        self.v_max = (bounds[1] - bounds[0]) * 0.2

        self.c1_i, self.c1_f = 2.5, 0.5
        self.c2_i, self.c2_f = 0.5, 2.5

        self.gbest_position = None
        self.gbest_fitness = np.inf

        self.positions = np.zeros((num_particles, dim))
        self.velocities = np.zeros((num_particles, dim))
        self.pbest_positions = np.zeros((num_particles, dim))
        self.pbest_fitness = np.full(num_particles, np.inf)

        self.initialize_swarm()

    def initialize_swarm(self):
        lower_bound, upper_bound = self.bounds
        self.positions = np.random.uniform(lower_bound, upper_bound, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-self.v_max, self.v_max, (self.num_particles, self.dim))
        self.pbest_positions = np.copy(self.positions)

    def update_search_factor(self, velocity_magnitude):
        # eq 9
        ratio = velocity_magnitude / self.v_max
        w = self.w_min + (self.w_max - self.w_min) * (1.0 / (1.0 + ratio))
        return w

    def optimize(self):
        history_fitness = []

        for k in tqdm(range(self.max_iter), desc="VASF-PSO Progress: "):
            # eq 7 & 8
            c1 = (self.c1_f - self.c1_i) * (k / self.max_iter) + self.c1_i
            c2 = (self.c2_f - self.c2_i) * (k / self.max_iter) + self.c2_i

            for i in range(self.num_particles):
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
                node_positions = self.positions[i].reshape(-1, 2)
                # eq 4
                fitness = self.fitness_function(self.positions[i])

                # update Pbest
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()

                # update Gbest
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()

            history_fitness.append(self.gbest_fitness)

            for i in range(self.num_particles):
                vel_meg = np.linalg.norm(self.velocities[i])

                # eq 9
                w = self.update_search_factor(vel_meg)

                # random number r1, r2
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # eq 5
                new_v = (w * self.velocities[i] +
                        c1 * r1 * (self.pbest_positions[i] - self.positions[i]) +
                        c2 * r2 * (self.gbest_position - self.positions[i]))

                new_v = np.clip(new_v, -self.v_max, self.v_max)
                self.velocities[i] = new_v

                # eq 6
                new_pos = self.positions[i] + self.velocities[i]

                new_pos = np.clip(new_pos, self.bounds[0], self.bounds[1])
                self.positions[i] = new_pos

        return self.gbest_position, self.gbest_fitness, history_fitness