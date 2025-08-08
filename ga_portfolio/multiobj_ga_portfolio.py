import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Individual:
    weights: np.ndarray
    # We will minimize both objectives for NSGA-II: (risk, -return)
    # Keep also the natural values for reporting/plotting
    risk: float
    expected_return: float
    domination_rank: int = 0
    crowding_distance: float = 0.0


# -------------------------------
# Data generation / evaluation
# -------------------------------

def generate_synthetic_data(num_assets: int, seed: int = 42,
                             mean_return_low: float = 0.02,
                             mean_return_high: float = 0.20,
                             base_vol_low: float = 0.10,
                             base_vol_high: float = 0.40,
                             correlation: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic expected returns and covariance matrix.

    Returns:
        mu: shape (N,)
        cov: shape (N, N)
    """
    rng = np.random.default_rng(seed)

    # Expected returns
    mu = rng.uniform(mean_return_low, mean_return_high, size=num_assets)

    # Create random volatilities
    vols = rng.uniform(base_vol_low, base_vol_high, size=num_assets)

    # Random correlation matrix with base correlation
    # Using a factor model approach: cov = B B^T + D
    num_factors = min(5, max(1, num_assets // 5))
    B = rng.normal(0, 1.0, size=(num_assets, num_factors))
    # Scale factor loadings so that induced correlations sit around given correlation
    B = B * math.sqrt(correlation)

    # Idiosyncratic variances to hit target vols
    # Start with identity, then rescale diagonal so that diag(cov) ~= vols^2
    D = np.diag(np.maximum(1e-6, vols ** 2)) * (1 - correlation)
    raw_cov = B @ B.T + D

    # Rescale to ensure diagonal equals vols^2 exactly
    scale = np.sqrt(np.diag(raw_cov))
    scale[scale == 0] = 1.0
    S_inv = np.diag(1.0 / scale)
    cov = S_inv @ raw_cov @ S_inv
    cov = (np.diag(vols) @ cov @ np.diag(vols))

    return mu, cov


def evaluate_individual(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> Tuple[float, float]:
    # Portfolio risk as standard deviation
    variance = float(weights.T @ cov @ weights)
    variance = max(variance, 0.0)
    risk = math.sqrt(variance)
    exp_ret = float(weights @ mu)
    return risk, exp_ret


# -------------------------------
# GA operators and NSGA-II helpers
# -------------------------------

def initialize_population(pop_size: int, num_assets: int, rng: np.random.Generator,
                          max_weight: float | None = None) -> List[np.ndarray]:
    # Draw from Dirichlet to satisfy non-negativity and sum-to-one
    alpha = np.ones(num_assets)
    pop = rng.dirichlet(alpha, size=pop_size)
    if max_weight is not None:
        pop = np.array([repair_weights(ind, max_weight=max_weight, rng=rng) for ind in pop])
    return [np.array(ind, dtype=float) for ind in pop]


def repair_weights(weights: np.ndarray, max_weight: float | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    w = weights.copy()
    # Enforce non-negativity
    w[w < 0] = 0.0
    s = w.sum()
    if s <= 0:
        # Fallback to random Dirichlet
        if rng is None:
            rng = np.random.default_rng()
        w = rng.dirichlet(np.ones_like(w))
    else:
        w = w / s

    if max_weight is not None:
        # Iteratively cap and renormalize until all constraints satisfied
        max_iters = 50
        for _ in range(max_iters):
            over = w > max_weight
            if not np.any(over):
                break
            excess = (w[over] - max_weight).sum()
            w[over] = max_weight
            under = ~over
            if np.any(under):
                redistribute = w[under]
                total_under = redistribute.sum()
                if total_under > 0:
                    w[under] = redistribute + (redistribute / total_under) * excess
                else:
                    # If nothing under cap, spread evenly among under set (shouldn't happen)
                    w[under] = w[under] + excess / max(1, under.sum())
            # Renormalize minor numeric drift
            w[w < 0] = 0.0
            s = w.sum()
            if s > 0:
                w = w / s
        # Final small numerical cleanup
        w[w < 0] = 0.0
        s = w.sum()
        if s > 0:
            w = w / s
        else:
            if rng is None:
                rng = np.random.default_rng()
            w = rng.dirichlet(np.ones_like(w))
    return w


def blend_crossover(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator,
                    alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    # BLX-alpha crossover generalization: sample each gene between extended interval
    low = np.minimum(parent1, parent2)
    high = np.maximum(parent1, parent2)
    diff = high - low
    lower_bound = low - alpha * diff
    upper_bound = high + alpha * diff
    child1 = rng.uniform(lower_bound, upper_bound)
    child2 = rng.uniform(lower_bound, upper_bound)
    return child1, child2


def gaussian_mutation(weights: np.ndarray, rng: np.random.Generator,
                      sigma: float = 0.05, mutation_rate: float = 0.3) -> np.ndarray:
    w = weights.copy()
    mask = rng.random(w.shape) < mutation_rate
    noise = rng.normal(0.0, sigma, size=w.shape)
    w = np.where(mask, w + noise, w)
    return w


def fast_non_dominated_sort(objs: List[Tuple[float, float]]) -> List[List[int]]:
    # objs: list of (risk, neg_return) both are to be minimized
    num = len(objs)
    S = [set() for _ in range(num)]
    n = [0] * num
    fronts: List[List[int]] = [[]]

    def dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
        return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

    for p in range(num):
        for q in range(num):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].add(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()  # remove last empty
    return fronts


def compute_crowding_distance(front_indices: List[int], objs: List[Tuple[float, float]]) -> dict:
    distance = {idx: 0.0 for idx in front_indices}
    if len(front_indices) == 0:
        return distance
    # For each objective
    for m in range(2):
        # Sort indices by objective m
        sorted_idx = sorted(front_indices, key=lambda i: objs[i][m])
        distance[sorted_idx[0]] = float('inf')
        distance[sorted_idx[-1]] = float('inf')
        values = [objs[i][m] for i in sorted_idx]
        min_v, max_v = values[0], values[-1]
        if max_v == min_v:
            # All identical; skip to avoid division by zero
            continue
        for j in range(1, len(sorted_idx) - 1):
            prev_v = objs[sorted_idx[j - 1]][m]
            next_v = objs[sorted_idx[j + 1]][m]
            distance[sorted_idx[j]] += (next_v - prev_v) / (max_v - min_v)
    return distance


def tournament_select(candidates: List[Individual], k: int, rng: np.random.Generator) -> Individual:
    a, b = rng.choice(candidates, size=2, replace=False)
    # Prefer lower rank, then higher crowding distance
    if a.domination_rank < b.domination_rank:
        return a
    if b.domination_rank < a.domination_rank:
        return b
    if a.crowding_distance > b.crowding_distance:
        return a
    if b.crowding_distance > a.crowding_distance:
        return b
    return a if rng.random() < 0.5 else b


# -------------------------------
# NSGA-II main loop
# -------------------------------

def nsga2_optimize(mu: np.ndarray,
                   cov: np.ndarray,
                   pop_size: int = 200,
                   generations: int = 200,
                   crossover_prob: float = 0.9,
                   mutation_prob: float = 0.9,
                   mutation_sigma: float = 0.05,
                   mutation_rate: float = 0.3,
                   seed: int = 42,
                   max_weight: float | None = None) -> Tuple[List[Individual], List[Individual]]:
    rng = np.random.default_rng(seed)
    num_assets = len(mu)
    population_weights = initialize_population(pop_size, num_assets, rng, max_weight=max_weight)

    def make_individual(w: np.ndarray) -> Individual:
        w_fixed = repair_weights(w, max_weight=max_weight, rng=rng)
        risk, er = evaluate_individual(w_fixed, mu, cov)
        return Individual(weights=w_fixed, risk=risk, expected_return=er)

    population = [make_individual(w) for w in population_weights]

    for gen in range(generations):
        # Evaluate ranks and crowding distances
        objs = [(ind.risk, -ind.expected_return) for ind in population]
        fronts = fast_non_dominated_sort(objs)
        for rank, front in enumerate(fronts):
            cd = compute_crowding_distance(front, objs)
            for idx in front:
                population[idx].domination_rank = rank
                population[idx].crowding_distance = cd[idx]

        # Create offspring via selection, crossover, mutation
        mating_pool: List[Individual] = []
        while len(mating_pool) < pop_size:
            selected = tournament_select(population, k=2, rng=rng)
            mating_pool.append(selected)

        offspring: List[Individual] = []
        for i in range(0, pop_size, 2):
            p1 = mating_pool[i].weights
            p2 = mating_pool[min(i + 1, pop_size - 1)].weights
            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < crossover_prob:
                c1, c2 = blend_crossover(p1, p2, rng, alpha=0.2)
            if rng.random() < mutation_prob:
                c1 = gaussian_mutation(c1, rng, sigma=mutation_sigma, mutation_rate=mutation_rate)
            if rng.random() < mutation_prob:
                c2 = gaussian_mutation(c2, rng, sigma=mutation_sigma, mutation_rate=mutation_rate)
            offspring.append(make_individual(c1))
            if len(offspring) < pop_size:
                offspring.append(make_individual(c2))

        # Combine and select next generation (elitist)
        combined = population + offspring
        objs_all = [(ind.risk, -ind.expected_return) for ind in combined]
        fronts_all = fast_non_dominated_sort(objs_all)

        new_population: List[Individual] = []
        for front in fronts_all:
            if len(new_population) + len(front) <= pop_size:
                # Add whole front
                cd = compute_crowding_distance(front, objs_all)
                for idx in front:
                    ind = combined[idx]
                    ind.domination_rank = len(new_population)  # not exact, but not used later
                    ind.crowding_distance = cd[idx]
                    new_population.append(ind)
            else:
                # Fill remaining with top crowding distances
                cd = compute_crowding_distance(front, objs_all)
                sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                remaining = pop_size - len(new_population)
                for idx in sorted_front[:remaining]:
                    ind = combined[idx]
                    ind.domination_rank = len(new_population)
                    ind.crowding_distance = cd[idx]
                    new_population.append(ind)
                break
        population = new_population

    # Final fronts for output
    objs = [(ind.risk, -ind.expected_return) for ind in population]
    fronts = fast_non_dominated_sort(objs)
    pareto_front = [population[i] for i in fronts[0]]

    return population, pareto_front


# -------------------------------
# Visualization
# -------------------------------

def plot_pareto_front(population: List[Individual], pareto_front: List[Individual], out_path: str | None = None):
    risks = [ind.risk for ind in population]
    returns = [ind.expected_return for ind in population]

    pf_risks = [ind.risk for ind in pareto_front]
    pf_returns = [ind.expected_return for ind in pareto_front]

    plt.figure(figsize=(8, 6))
    plt.scatter(risks, returns, c="#cccccc", s=12, label="Population")
    plt.scatter(pf_risks, pf_returns, c="#d62728", s=22, label="Pareto Front")
    plt.xlabel("Risk (Std Dev)")
    plt.ylabel("Expected Return")
    plt.title("NSGA-II Portfolio Optimization: Pareto Front")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()
    plt.close()


# -------------------------------
# Utilities
# -------------------------------

def select_knee_point(pareto_front: List[Individual]) -> Individual:
    # Normalize risks and returns, pick point maximizing distance to line between extremes
    risks = np.array([ind.risk for ind in pareto_front])
    rets = np.array([ind.expected_return for ind in pareto_front])

    # Sort by risk asc
    order = np.argsort(risks)
    risks = risks[order]
    rets = rets[order]

    # Endpoints
    p1 = np.array([risks[0], rets[0]])
    p2 = np.array([risks[-1], rets[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return pareto_front[order[0]]
    line_unit = line_vec / line_len

    max_dist = -1.0
    max_idx = 0
    for i in range(len(risks)):
        p = np.array([risks[i], rets[i]])
        # Perpendicular distance from point to line
        proj_len = np.dot(p - p1, line_unit)
        proj_point = p1 + proj_len * line_unit
        dist = np.linalg.norm(p - proj_point)
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    return pareto_front[order[max_idx]]


# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="NSGA-II for portfolio optimization (min risk, max return)")
    parser.add_argument("--assets", type=int, default=30, help="Number of synthetic assets")
    parser.add_argument("--pop_size", type=int, default=200, help="Population size")
    parser.add_argument("--generations", type=int, default=200, help="Number of generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--crossover_prob", type=float, default=0.9, help="Crossover probability")
    parser.add_argument("--mutation_prob", type=float, default=0.9, help="Mutation probability")
    parser.add_argument("--mutation_sigma", type=float, default=0.05, help="Gaussian mutation sigma")
    parser.add_argument("--mutation_rate", type=float, default=0.3, help="Per-gene mutation probability")
    parser.add_argument("--max_weight", type=float, default=0.2, help="Maximum weight per asset (e.g., 0.2). Use 1.0 to disable practical cap.")
    parser.add_argument("--out", type=str, default="/workspace/ga_portfolio/output/pareto_front.png", help="Path to save Pareto front plot")

    args = parser.parse_args()

    if args.max_weight >= 1.0:
        max_weight = None
    else:
        max_weight = args.max_weight

    mu, cov = generate_synthetic_data(args.assets, seed=args.seed)

    population, pareto_front = nsga2_optimize(
        mu=mu,
        cov=cov,
        pop_size=args.pop_size,
        generations=args.generations,
        crossover_prob=args.crossover_prob,
        mutation_prob=args.mutation_prob,
        mutation_sigma=args.mutation_sigma,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        max_weight=max_weight,
    )

    # Visualization
    plot_pareto_front(population, pareto_front, out_path=args.out)

    # Report a few representative solutions
    best_return = max(pareto_front, key=lambda ind: ind.expected_return)
    lowest_risk = min(pareto_front, key=lambda ind: ind.risk)
    knee = select_knee_point(pareto_front)

    def summarize(ind: Individual, name: str):
        top_weights_idx = np.argsort(ind.weights)[-5:][::-1]
        top_str = ", ".join([f"a{idx}:{ind.weights[idx]:.3f}" for idx in top_weights_idx])
        print(f"{name}: risk={ind.risk:.4f}, return={ind.expected_return:.4f}, top_weights=[{top_str}]")

    print("Saved Pareto front to:", args.out)
    summarize(lowest_risk, "Lowest Risk on Pareto")
    summarize(best_return, "Highest Return on Pareto")
    summarize(knee, "Knee Point on Pareto")


if __name__ == "__main__":
    main()