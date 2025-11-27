import numpy as np
import matplotlib.pyplot as plt
from vasf_pso import vasf_pso

def f1_schwefel_2_21(x):
    """Schwefel 2.21: Max|x_i|"""
    return np.max(np.abs(x))

def f2_sphere(x):
    """Sphere: Sum(x^2)"""
    return np.sum(x**2)

def f3_schwefel_2_22(x):
    """Schwefel 2.22: Sum|x| + Prod|x|"""
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def f4_schwefel_1_2(x):
    """Schwefel 1.2: Sum(Sum(x_j)^2)"""
    dim = len(x)
    total = 0
    running_sum = 0
    for i in range(dim):
        running_sum += x[i]
        total += running_sum**2
    return total

def f5_quartic_noise(x):
    """Quartic with Noise: Sum(i * x^4 + rand)"""
    dim = len(x)
    indices = np.arange(1, dim + 1)
    return np.sum(indices * (x**4)) + np.random.rand()

def f6_rosenbrock(x):
    """Rosenbrock"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# ==========================================
# 2. Multimodal Benchmark Functions (Table 3)
# ==========================================

def u_func(x, a, k, m):
    """Helper for Penalized functions"""
    r = np.zeros_like(x)
    mask1 = x > a
    mask2 = x < -a
    r[mask1] = k * (x[mask1] - a)**m
    r[mask2] = k * (-x[mask2] - a)**m
    return r

def f7_penalized_2(x):
    """Penalized 2"""
    dim = len(x)
    a, k, m = 5, 100, 4

    term1 = 0.1 * (
        (np.sin(3 * np.pi * x[0]))**2 +
        np.sum(((x[:-1] - 1)**2) * (1 + (np.sin(3 * np.pi * x[1:] + 1))**2)) +
        ((x[-1] - 1)**2) * (1 + (np.sin(2 * np.pi * x[-1]))**2)
    )
    term2 = np.sum(u_func(x, a, k, m))
    return term1 + term2

def f8_penalized_1(x):
    """Penalized 1"""
    dim = len(x)
    y = 1 + (x + 1) / 4
    a, k, m = 10, 100, 4

    term1 = (np.pi / dim) * (
        10 * (np.sin(np.pi * y[0]))**2 +
        np.sum(((y[:-1] - 1)**2) * (1 + 10 * (np.sin(np.pi * y[1:] + 1))**2)) +
        (y[-1] - 1)**2
    )
    term2 = np.sum(u_func(x, a, k, m))
    return term1 + term2

def f9_griewank(x):
    """Griewank"""
    dim = len(x)
    indices = np.arange(1, dim + 1)
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod(np.cos(x / np.sqrt(indices)))
    return sum_sq - prod_cos + 1

def f10_rastrigin(x):
    """Rastrigin"""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def f11_ackley(x):
    """Ackley"""
    dim = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))

    term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim))
    term2 = -np.exp(sum_cos / dim)
    return term1 + term2 + 20 + np.e

def f12_salomon(x):
    """Salomon"""
    norm_x = np.sqrt(np.sum(x**2))
    return 1 - np.cos(2 * np.pi * norm_x) + 0.1 * norm_x

def f13_xin_she_yang(x):
    """Xin-She Yang"""
    # Assuming Xin-She Yang 2 or similar standard variant based on typical benchmarks
    # f(x) = sum(|x| * exp(-sum(sin(x^2))))
    # Paper formula description might vary, using standard interpretation
    return np.sum(np.abs(x)) * np.exp(-np.sum(np.sin(x**2)))

# ==========================================
# 3. CEC 2021 Style Functions (Table 4)
# Note: Full CEC functions require complex shifts/rotation matrices.
# These are the Base definitions.
# ==========================================

def f14_zakharov(x):
    """Zakharov Function"""
    dim = len(x)
    indices = np.arange(1, dim + 1) # 1 to D (0.5*i*x)
    # Standard formula uses 0.5 * i * x
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * indices * x)
    return sum1 + sum2**2 + sum2**4

def f15_schaffer_f6(x):
    """Expanded Schaffer's F6"""
    # f(x,y) = 0.5 + (sin^2(sqrt(x^2+y^2)) - 0.5) / (1 + 0.001(x^2+y^2))^2
    # Expanded means f(x1,x2) + f(x2,x3) ...
    val = 0
    dim = len(x)
    for i in range(dim - 1):
        sq = x[i]**2 + x[i+1]**2
        sin_sq = np.sin(np.sqrt(sq))**2
        val += 0.5 + (sin_sq - 0.5) / (1 + 0.001 * sq)**2
    # Wrap around for last element if needed, typically x_D and x_1
    sq = x[-1]**2 + x[0]**2
    sin_sq = np.sin(np.sqrt(sq))**2
    val += 0.5 + (sin_sq - 0.5) / (1 + 0.001 * sq)**2
    return val

def f16_levy(x):
    """Levy Function"""
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0]))**2
    term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1))**2))
    return term1 + term2 + term3

def f17_hybrid_proxy(x):
    """Proxy for Hybrid Function (Mix of Sphere and Rastrigin)"""
    # CEC Hybrids are specific weightings, this is a placeholder behavior
    half = len(x) // 2
    return f2_sphere(x[:half]) + f10_rastrigin(x[half:])

def f18_composition_proxy(x):
    """Proxy for Composition Function (Weighted mix of Ackley and Griewank)"""
    return 0.5 * f11_ackley(x) + 0.5 * f9_griewank(x)

FUNCTIONS = [
    ("F1 Schwefel 2.21", f1_schwefel_2_21, [-100, 100]),
    ("F2 Sphere", f2_sphere, [-100, 100]),
    ("F3 Schwefel 2.22", f3_schwefel_2_22, [-10, 10]),
    ("F4 Schwefel 1.2", f4_schwefel_1_2, [-100, 100]),
    ("F5 Quartic Noise", f5_quartic_noise, [-1.28, 1.28]),
    ("F6 Rosenbrock", f6_rosenbrock, [-30, 30]),
    ("F7 Penalized 2", f7_penalized_2, [-50, 50]),
    ("F8 Penalized 1", f8_penalized_1, [-50, 50]),
    ("F9 Griewank", f9_griewank, [-600, 600]),
    ("F10 Rastrigin", f10_rastrigin, [-5.12, 5.12]),
    ("F11 Ackley", f11_ackley, [-32, 32]),
    ("F12 Salomon", f12_salomon, [-20, 20]),
    ("F13 Xin-She Yang", f13_xin_she_yang, [-5, 5]),
    ("F14 Zakharov", f14_zakharov, [-10, 10]),
    ("F15 Schaffer F6", f15_schaffer_f6, [-100, 100]),
    ("F16 Levy", f16_levy, [-10, 10]),
    ("F17 Hybrid (Proxy)", f17_hybrid_proxy, [-100, 100]),
    ("F18 Composition (Proxy)", f18_composition_proxy, [-100, 100]),
]

def run_benchmark():
    print(f"{'Function':<25} | {'Best Fitness':<15} | {'Status'}")
    print("-" * 55)

    dim = 30
    pop_size = 30
    max_iter = 1000
    times = 30

    list_val = {bm['name']: [] for bm in FUNCTIONS}
    # results = {bm['name']: [] for bm in FUNCTIONS}

    for name, func, bounds in FUNCTIONS:
        for _ in range(times):
            optimizer = vasf_pso(pop_size, dim, bounds, max_iter, func)
            _, best_fit, history = optimizer.optimize()

            # results[name].append(history)
            list_val[name].append(best_fit)

        print(f"{name:<25} | {best_fit:15.6e} | Done")
        # results[name] = history

    for name, scores in list_val.items():
        print(f"name is: {name}")
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        min_val = np.max(scores)
        max_val = np.min(scores)

        print(f"Mean values: {mean_val:.4e}")
        print(f"Std values: {std_val:.4e}")
        print(f"Min values: {min_val:.4e}")
        print(f"Max values: {max_val:.4e}")

    # Plot specific examples (e.g., Unimodal vs Multimodal)
    # plt.figure(figsize=(10, 6))
    # plt.plot(results["F2 Sphere"], label="Sphere (Unimodal)")
    # plt.plot(results["F10 Rastrigin"], label="Rastrigin (Multimodal)")
    # plt.plot(results["F11 Ackley"], label="Ackley (Multimodal)")
    # plt.yscale('log')
    # plt.xlabel("Iterations")
    # plt.ylabel("Fitness (Log Scale)")
    # plt.title("VASF-PSO Convergence on Benchmarks")
    # plt.legend()
    # plt.grid(True)
    # plt.show()