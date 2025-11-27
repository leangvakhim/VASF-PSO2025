import numpy as np
from vasf_pso import vasf_pso
from wsn import wsn
from benchmark import run_benchmark

print("1. Coverage")
print("2. Benchmark")
opt = int(input("Enter an options: "))

if opt == 1:
    area_size = (50, 50)
    num_nodes = 32
    sensing_radius = 5
    sensing_error = 1
    swarm_size = 30
    max_iter = 500
    dim = num_nodes * 2
    bounds = [0, 50]

    wsn_env = wsn(area_size, sensing_radius, sensing_error, grid_resolution=1.0)

    optimizer = vasf_pso(swarm_size, dim, bounds, max_iter, wsn_env)

    best_pos_flat, best_cov, history = optimizer.optimize()

    best_node = best_pos_flat.reshape(-1, 2)

    wsn_env.plot_results(best_node, area_size, sensing_radius, history, best_cov)

elif opt == 2:
    run_benchmark()
