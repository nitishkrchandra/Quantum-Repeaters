import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, FixedLocator


def normalize_state(a, b, c, d):
    """ Normalize the input coefficients so that they sum to 1. """
    S = a + b + c + d
    return a / S, b / S, c / S, d / S

def generate_biased_random_state(seed=None, bias=0.9):
    """Generates a Bell-diagonal state biased towards |Φ⁺⟩."""
    if seed is not None:
        np.random.seed(seed)
    # Generate small deviations
    perturbations = np.random.rand(4) * (1 - bias)
    a = bias + perturbations[0]
    b, c, d = perturbations[1], perturbations[2], perturbations[3]
    return normalize_state(a, b, c, d)

def get_purification(a1, b1, c1, d1, a2, b2, c2, d2, epsilon_G, xi):
    """
    Performs entanglement purification based on given input parameters.

    Inputs:
    - a1, b1, c1, d1: Initial density matrix coefficients for first pair
    - a2, b2, c2, d2: Initial density matrix coefficients for second pair
    - epsilon_G: Gate infidelity
    - xi: Measurement infidelity

    Returns:
    - P: Success probability of purification
    - a, b, c, d: New purified density matrix coefficients
    """
    P = ((1 - epsilon_G) ** 2) * ((xi**2 + (1 - xi)**2) * ((a1 + d1) * (a2 + d2) + (b1 + c1) * (c2 + b2))
        + 2 * xi * (1 - xi) * ((a1 + d1) * (b2 + c2) + (b1 + c1) * (a2 + d2))) + 0.5 * (1 - (1 - epsilon_G) ** 2)

    a = (1 / P) * ((1 - epsilon_G) ** 2 * ((xi**2 + (1 - xi)**2) * (a1 * a2 + d1 * d2) + 2 * xi * (1 - xi) * 
        (a1 * c2 + d1 * b2)) + (1/8) * (1 - (1 - epsilon_G) ** 2))
    
    b = (1 / P) * ((1 - epsilon_G) ** 2 * ((xi**2 + (1 - xi)**2) * (a1 * d2 + d1 * a2) + 2 * xi * (1 - xi) * 
        (a1 * b2 + d1 * c2)) + (1/8) * (1 - (1 - epsilon_G) ** 2))
    
    c = (1 / P) * ((1 - epsilon_G) ** 2 * ((xi**2 + (1 - xi)**2) * (b1 * b2 + c1 * c2) + 2 * xi * (1 - xi) * 
        (b1 * d2 + c1 * a2)) + (1/8) * (1 - (1 - epsilon_G) ** 2))
    
    d = (1 / P) * ((1 - epsilon_G) ** 2 * ((xi**2 + (1 - xi)**2) * (b1 * c2 + c1 * b2) + 2 * xi * (1 - xi) * 
        (b1 * a2 + c1 * d2)) + (1/8) * (1 - (1 - epsilon_G) ** 2))

    return P, a, b, c, d

def run_entanglement_swapping(a1, b1, c1, d1, a2, b2, c2, d2, epsilon_G, xi):
    """
    Performs entanglement swapping based on given input parameters.

    Inputs:
    - a1, b1, c1, d1: Initial density matrix coefficients for first pair
    - a2, b2, c2, d2: Initial density matrix coefficients for second pair
    - epsilon_G: Gate infidelity
    - xi: Measurement infidelity

    Returns:
    - a, b, c, d: New swapped density matrix coefficients
    """
    # Common terms
    factor = (1 - epsilon_G)
    term_1 = (1 - xi) ** 2
    term_2 = xi * (1 - xi)
    term_3 = xi ** 2

    # Compute new Bell-diagonal coefficients
    a = factor * (
        term_1 * (a1 * a2 + b1 * b2 + c1 * c2 + d1 * d2) +
        term_2 * ((a1 + d1) * (b2 + c2) + (b1 + c1) * (a2 + d2)) +
        term_3 * (a1 * d2 + d1 * a2 + b1 * c2 + c1 * b2)
    ) + epsilon_G / 4

    b = factor * (
        term_1 * (a1 * b2 + b1 * a2 + c1 * d2 + d1 * c2) +
        term_2 * ((a1 + d1) * (a2 + d2) + (b1 + c1) * (b2 + c2)) +
        term_3 * (a1 * c2 + c1 * a2 + b1 * d2 + d1 * b2)
    ) + epsilon_G / 4

    c = factor * (
        term_1 * (a1 * c2 + c1 * a2 + b1 * d2 + d1 * b2) +
        term_2 * ((a1 + d1) * (a2 + d2) + (b1 + c1) * (b2 + c2)) +
        term_3 * (a1 * b2 + b1 * a2 + c1 * d2 + d1 * c2)
    ) + epsilon_G / 4

    d = factor * (
        term_1 * (a1 * d2 + d1 * a2 + b1 * c2 + c1 * b2) +
        term_2 * ((a1 + d1) * (b2 + c2) + (b1 + c1) * (a2 + d2)) +
        term_3 * (a1 * a2 + b1 * b2 + c1 * c2 + d1 * d2)
    ) + epsilon_G / 4

    return a, b, c, d

def run_multiple_purification(a1, b1, c1, d1, a2, b2, c2, d2, epsilon_G, xi, m, protocol):
    """
    Run m rounds of purification, updating the state each time.
    
    Parameters:
    - a1...d1: coefficients for the initial main state
    - a2...d2: coefficients for the second input state
              - Protocol 1: identical to a1...d1
              - Protocol 2: fixed auxiliary state
    - epsilon_G: gate infidelity
    - xi: measurement infidelity
    - m: number of purification rounds
    - protocol: 1 (Deutsch) or 2 (Dür)
    
    Returns:
    - P: final purification success probability
    - a, b, c, d: purified Bell-diagonal coefficients
    """

    if protocol not in [1, 2]:
        raise ValueError("Invalid protocol. Use 1 for Deutsch or 2 for Dür.")

    # Initialize with the first purification step
    P, a, b, c, d = get_purification(a1, b1, c1, d1, a2, b2, c2, d2, epsilon_G, xi)

    # Additional rounds
    for _ in range(1, m):
        if protocol == 1:
            # Deutsch: Use updated state for both inputs
            P, a, b, c, d = get_purification(a, b, c, d, a, b, c, d, epsilon_G, xi)
        elif protocol == 2:
            # Dür: Main state updated, auxiliary state fixed (a2...d2)
            P, a, b, c, d = get_purification(a, b, c, d, a2, b2, c2, d2, epsilon_G, xi)

    return P, a, b, c, d

def P_purify(i, j, N, M_arrow, epsilon_G, xi, seed, protocol, seed2=100):
    """
    Compute the success probability of the i-th purification round at j-th nesting level
    based on the Deutsch (protocol=1) or Dür (protocol=2) purification protocol, including entanglement swapping.

    Parameters:
    - i: Current purification round (0-indexed)
    - j: Nesting level (0-indexed, j in [0, N])
    - N: Total number of nesting levels
    - M_arrow: List of purification rounds per level (length N+1)
    - epsilon_G: Gate infidelity
    - xi: Measurement infidelity
    - seed: Random seed for main state generation
    - seed2: Random seed for auxiliary state generation (used in Dür protocol)
    - protocol: 1 for Deutsch, 2 for Dür

    Returns:
    - P: Success probability after purification and swapping up to level j
    """

    if not (0 <= j <= N):
        raise ValueError(f"Invalid nesting level j={j}; must be in range [0, {N}]")
    if not (0 <= i <= M_arrow[j]):
        raise ValueError(f"Invalid purification round i={i} for level j={j}; exceeds M_arrow[j]={M_arrow[j]}")

    if protocol == 1:
        # Deutsch protocol: identical input states
        rho1 = generate_biased_random_state(seed=seed)
        a, b, c, d = rho1

        for level in range(j):
            P, a, b, c, d = run_multiple_purification(a, b, c, d, a, b, c, d,
                                                      epsilon_G=epsilon_G, xi=xi,
                                                      m=M_arrow[level], protocol=1)
            a, b, c, d = run_entanglement_swapping(a, b, c, d, a, b, c, d,
                                                   epsilon_G, xi)

        P, _, _, _, _ = run_multiple_purification(a, b, c, d, a, b, c, d,
                                                  epsilon_G=epsilon_G, xi=xi,
                                                  m=i, protocol=1)

    elif protocol == 2:
        if seed2 == seed:
            seed2 += 1  # Ensure different random states
        # Dür protocol: main pair + fixed auxiliary state
        rho1 = generate_biased_random_state(seed=seed)
        rho2 = generate_biased_random_state(seed=seed2)

        a1, b1, c1, d1 = rho1
        a2, b2, c2, d2 = rho2

        for level in range(j):
            P, a1, b1, c1, d1 = run_multiple_purification(a1, b1, c1, d1,
                                                          a2, b2, c2, d2,
                                                          epsilon_G=epsilon_G, xi=xi,
                                                          m=M_arrow[level], protocol=2)
            a1, b1, c1, d1 = run_entanglement_swapping(a1, b1, c1, d1,
                                                       a2, b2, c2, d2,
                                                       epsilon_G, xi)

        P, _, _, _, _ = run_multiple_purification(a1, b1, c1, d1,
                                                  a2, b2, c2, d2,
                                                  epsilon_G=epsilon_G, xi=xi,
                                                  m=i, protocol=2)

    else:
        raise ValueError("Invalid protocol. Use 1 for Deutsch or 2 for Dür.")

    return P

def get_final_state(i, j, N, M_arrow, epsilon_G, xi, seed, protocol, seed2=100):
    """
    Return the final Bell-diagonal coefficients (a, b, c, d) of the entangled state
    after purification and entanglement swapping up to level j for the given protocol.

    Parameters:
    - i: Purification round at level j
    - j: Nesting level
    - N: Total number of nesting levels
    - M_arrow: List of purification rounds per level (length N+1)
    - epsilon_G: Gate infidelity
    - xi: Measurement infidelity
    - seed: Random seed for main state generation
    - protocol: 1 (Deutsch), 2 (Dür)
    - seed2: Random seed for auxiliary state generation (Dür only)

    Returns:
    - a, b, c, d: Final Bell-diagonal coefficients at level j
    """
    if not (0 <= j <= N):
        raise ValueError(f"Invalid nesting level j={j}; must be in range [0, {N}]")
    if not (0 <= i <= M_arrow[j]):
        raise ValueError(f"Invalid purification round i={i} for level j={j}; exceeds M_arrow[j]={M_arrow[j]}")

    if protocol == 1:
        # Deutsch protocol: identical inputs
        rho1 = generate_biased_random_state(seed=seed)
        a, b, c, d = rho1

        for level in range(j):
            _, a, b, c, d = run_multiple_purification(a, b, c, d, a, b, c, d,
                                                      epsilon_G=epsilon_G, xi=xi,
                                                      m=M_arrow[level], protocol=1)
            a, b, c, d = run_entanglement_swapping(a, b, c, d, a, b, c, d,
                                                   epsilon_G, xi)

        _, a, b, c, d = run_multiple_purification(a, b, c, d, a, b, c, d,
                                                  epsilon_G=epsilon_G, xi=xi,
                                                  m=i, protocol=1)

    elif protocol == 2:
        # Dür protocol: main pair + fixed auxiliary
        if seed2 == seed:
            seed2 += 1

        rho1 = generate_biased_random_state(seed=seed)
        rho2 = generate_biased_random_state(seed=seed2)

        a1, b1, c1, d1 = rho1
        a2, b2, c2, d2 = rho2

        for level in range(j):
            _, a1, b1, c1, d1 = run_multiple_purification(a1, b1, c1, d1,
                                                          a2, b2, c2, d2,
                                                          epsilon_G=epsilon_G, xi=xi,
                                                          m=M_arrow[level], protocol=2)
            a1, b1, c1, d1 = run_entanglement_swapping(a1, b1, c1, d1,
                                                       a2, b2, c2, d2,
                                                       epsilon_G, xi)

        _, a, b, c, d = run_multiple_purification(a1, b1, c1, d1,
                                                  a2, b2, c2, d2,
                                                  epsilon_G=epsilon_G, xi=xi,
                                                  m=i, protocol=2)

    else:
        raise ValueError("Invalid protocol. Use 1 for Deutsch or 2 for Dür.")

    return a, b, c, d

def generate_M_arrow(N, max_rounds=5):
    """
    Generate a reasonable M_arrow list of purification rounds per level
    for a given nesting level N.

    Parameters:
    - N: Number of nesting levels
    - max_rounds: Maximum purification rounds per level (default: 5)

    Returns:
    - M_arrow: List of length N+1 with purification rounds per level
    """
    np.random.seed(N + 42)  # deterministic randomness based on N
    M_arrow = [np.random.randint(0, max_rounds + 1) for _ in range(N + 1)]
    return M_arrow

def compute_QBER(a, b, c, d):
    QZ = b + d  # Phase error
    QX = c + d  # Bit error
    return (QX + QZ) / 2  # Average QBER

def compute_T(N, M_arrow, epsilon_G, xi, seed, eta_c, L_att, t0, T0, L0, protocol=1, seed2=100):
    """
    Compute total temporal cost for Deutsch (protocol=1) or Dür (protocol=2) entanglement purification.

    Parameters:
    - N: number of nesting levels
    - M_arrow: list of purification rounds per level (length N+1)
    - epsilon_G: gate infidelity
    - xi: measurement infidelity
    - seed: random seed for main state generation
    - eta_c: coupling efficiency
    - L_att: attenuation length (in km)
    - t0: gate time (s)
    - T0: fundamental clock cycle (s)
    - L0: elementary link length (km)
    - protocol: 1 (Deutsch) or 2 (Dür)
    - seed2: optional auxiliary seed for Dür protocol

    Returns:
    - T_total: total time cost in seconds
    """

    P0 = 0.5 * eta_c**2 * np.exp(-L0 / L_att)

    A_list = []
    B_list = []

    for i in range(N + 1):
        M_i = M_arrow[i]

        # Compute A[i]
        if protocol == 1:
            # Deutsch protocol
            product_term = np.prod([
                1 / P_purify(i=M_i - x, j=i, N=N, M_arrow=M_arrow,
                             epsilon_G=epsilon_G, xi=xi, seed=seed, protocol=1)
                for x in range(M_i)
            ])
            sum_term = sum([
                (3/2)**y * np.prod([
                    1 / P_purify(i=M_i - x, j=i, N=N, M_arrow=M_arrow,
                                 epsilon_G=epsilon_G, xi=xi, seed=seed, protocol=1)
                    for x in range(y + 1)
                ])
                for y in range(M_i)
            ])
            A_i = (3/2)**M_i * product_term
            B_i = ((t0 / T0) + 2**i) * sum_term

        elif protocol == 2:
            # Dür protocol
            product_term = np.prod([
                1 / P_purify(i=M_i - x, j=i, N=N, M_arrow=M_arrow,
                             epsilon_G=epsilon_G, xi=xi, seed=seed, seed2=seed2, protocol=2)
                for x in range(M_i)
            ])
            sum_term = sum([
                np.prod([
                    1 / P_purify(i=M_i - x, j=i, N=N, M_arrow=M_arrow,
                                 epsilon_G=epsilon_G, xi=xi, seed=seed, seed2=seed2, protocol=2)
                    for x in range(y + 1)
                ])
                for y in range(M_i)
            ])
            A_i = product_term + sum_term
            B_i = ((t0 / T0) + 2**i) * sum_term

        else:
            raise ValueError("Invalid protocol. Use 1 for Deutsch or 2 for Dür.")

        A_list.append(A_i)
        B_list.append(B_i)

    # Common to both protocols
    prod_A = np.prod(A_list[1:N+1])  # product A_1 to A_N

    first_term = (3/2)**N * prod_A * ((1 / P0) * A_list[0] + B_list[0])
    second_term = sum([
        (3/2)**(N - y) * B_list[y] * np.prod(A_list[y+1:N+1])
        for y in range(1, N + 1)
    ])
    third_term = (t0 / T0) * sum([
        (3/2)**(N - y) * np.prod(A_list[y:N+1])
        for y in range(1, N + 1)
    ])

    return T0 * (first_term + second_term + third_term)

def key_rate(qber, T):
    qber_safe = np.clip(qber, 1e-10, 1 - 1e-10)
    h_Q = -qber_safe * math.log2(qber_safe) - (1 - qber_safe) * math.log2(1 - qber_safe)
    r_secure = max(1 - 2 * h_Q, 0)
    return r_secure / max(T, 1e-10)

def calculate_cost_coefficient_1G(num_points, L_att, eta_min, L_tot, epsilon_G, xi, t0, seed, N, M_arrow, plot=True):
    """
    Calculates cost efficiency (C / L_tot) for first-generation repeaters,
    compares Deutsch (protocol=1) and Dür (protocol=2), selects optimal.

    Returns:
    - optimal_costs: list of min(cost_D, cost_Dr) / L_tot per eta_c
    - etas: corresponding eta_c values
    - protocol_labels: list of 'Deutsch' or 'Dür' indicating better protocol
    """
    eta_vals = np.linspace(eta_min, 1.0, num_points)
    L0 = L_tot / (2 ** N)
    T0 = L0 / 2e5  # c = 2e5 km/s

    i = M_arrow[N]
    j = N
    niter = 30

    optimal_costs = []
    etas = []
    protocol_labels = []

    for eta_c in eta_vals:
        # Deutsch
        total_keyrate_D = 0
        for k in range(niter):
            a, b, c_val, d = get_final_state(i, j, N, M_arrow,
                                             epsilon_G=epsilon_G, xi=xi,
                                             seed=seed + k, protocol=1)
            qber = compute_QBER(a, b, c_val, d)
            T_deu = compute_T(N, M_arrow, epsilon_G, xi, seed + k,
                              eta_c, L_att, t0, T0, L0, protocol=1)
            total_keyrate_D += key_rate(qber, T_deu)
        R_D = total_keyrate_D / niter
        Z_D = 2 ** sum(M_arrow)
        C_D = 2 ** (N + 1) * Z_D / R_D if R_D > 0 else np.inf
        Ceff_D = C_D / L_tot

        # Dür
        total_keyrate_Dr = 0
        for k in range(niter):
            a, b, c_val, d = get_final_state(i, j, N, M_arrow,
                                             epsilon_G=epsilon_G, xi=xi,
                                             seed=seed + k, protocol=2, seed2=123 + k)
            qber = compute_QBER(a, b, c_val, d)
            T_dur = compute_T(N, M_arrow, epsilon_G, xi, seed + k,
                              eta_c, L_att, t0, T0, L0, protocol=2, seed2=123 + k)
            total_keyrate_Dr += key_rate(qber, T_dur)
        R_Dr = total_keyrate_Dr / niter
        Z_Dr = N + 2 - sum(1 for m in M_arrow if m == 0)
        C_Dr = 2 ** (N + 1) * Z_Dr / R_Dr if R_Dr > 0 else np.inf
        Ceff_Dr = C_Dr / L_tot

        # Compare
        if Ceff_D < Ceff_Dr:
            optimal_costs.append(Ceff_D)
            protocol_labels.append("Deutsch")
        else:
            optimal_costs.append(Ceff_Dr)
            protocol_labels.append("Dür")

        etas.append(eta_c)

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(etas, optimal_costs, marker='o', label="Optimal (Deutsch vs. Dür)")

        plt.xscale("log")
        plt.yscale("log")

        plt.gca().xaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1.0]))
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.gca().tick_params(axis='x', which='major', labelsize=12)

        plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        plt.gca().tick_params(axis='y', which='major', labelsize=12)

        plt.xlabel("Coupling efficiency $\\eta_c$", fontsize=14)
        plt.ylabel("Cost Efficiency $C / L_{tot}$", fontsize=14)
        plt.title("Cost Efficiency vs Coupling Efficiency (1G Repeater)", fontsize=15)

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    return optimal_costs, etas, protocol_labels
