"""Helper methods for OR-Library portfolio batch experiments.

This module intentionally keeps dependencies to the Python standard library
so the batch runner can execute in a minimal environment.
"""

import csv
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


EPS = 1e-12


class PortfolioProblem(object):
    """Container for a constrained portfolio optimization instance."""

    def __init__(
        self,
        mu: Sequence[float],
        covariance: Sequence[Sequence[float]],
        k: int,
        lower_bounds: Optional[Sequence[float]] = None,
        upper_bounds: Optional[Sequence[float]] = None,
    ) -> None:
        self.mu = [float(x) for x in mu]
        self.cov = [[float(v) for v in row] for row in covariance]
        self.n = len(self.mu)
        self.k = int(k)

        if self.n == 0:
            raise ValueError("mu must be non-empty")
        if self.k <= 0 or self.k > self.n:
            raise ValueError("k must satisfy 1 <= k <= n")
        if len(self.cov) != self.n or any(len(row) != self.n for row in self.cov):
            raise ValueError("covariance must be n x n")

        if lower_bounds is None:
            self.lower = [0.0] * self.n
        else:
            self.lower = [float(x) for x in lower_bounds]
        if upper_bounds is None:
            self.upper = [1.0] * self.n
        else:
            self.upper = [float(x) for x in upper_bounds]

        if len(self.lower) != self.n or len(self.upper) != self.n:
            raise ValueError("bounds must be length n")
        for i in range(self.n):
            if self.lower[i] < 0.0:
                raise ValueError("lower bounds must be nonnegative")
            if self.upper[i] <= 0.0:
                raise ValueError("upper bounds must be positive")
            if self.lower[i] > self.upper[i]:
                raise ValueError("lower bound cannot exceed upper bound")

        # A sufficient global feasibility check under uniform k-active selection.
        sorted_lower = sorted(self.lower)
        sorted_upper = sorted(self.upper, reverse=True)
        min_possible = sum(sorted_lower[: self.k])
        max_possible = sum(sorted_upper[: self.k])
        if min_possible - 1.0 > 1e-9 or 1.0 - max_possible > 1e-9:
            raise ValueError(
                "No feasible k-asset portfolio satisfies sum(w)=1 with given bounds"
            )


def dot(x: Sequence[float], y: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def portfolio_return(problem: PortfolioProblem, w: Sequence[float]) -> float:
    return dot(problem.mu, w)


def portfolio_risk(problem: PortfolioProblem, w: Sequence[float]) -> float:
    active = [i for i, wi in enumerate(w) if wi > EPS]
    total = 0.0
    cov = problem.cov
    for i in active:
        wi = w[i]
        row = cov[i]
        for j in active:
            total += wi * w[j] * row[j]
    return total


def risk_gradient(problem: PortfolioProblem, w: Sequence[float]) -> List[float]:
    # grad(w^T Sigma w) = 2 Sigma w
    n = problem.n
    grad = [0.0] * n
    active = [i for i, wi in enumerate(w) if wi > EPS]
    for j in active:
        wj = w[j]
        for i in range(n):
            grad[i] += 2.0 * problem.cov[i][j] * wj
    return grad


def is_active_set_feasible(problem: PortfolioProblem, active: Sequence[int]) -> bool:
    lower_sum = sum(problem.lower[i] for i in active)
    upper_sum = sum(problem.upper[i] for i in active)
    return lower_sum <= 1.0 + 1e-9 and upper_sum >= 1.0 - 1e-9


def _pick_active_set_from_scores(
    problem: PortfolioProblem, scores: Sequence[float], rng: random.Random
) -> List[int]:
    n = problem.n
    k = problem.k
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    candidate = ranked[:k]
    if is_active_set_feasible(problem, candidate):
        return candidate

    # Randomized fallback if top-k violates bounds.
    all_idx = list(range(n))
    for _ in range(600):
        cand = rng.sample(all_idx, k)
        if is_active_set_feasible(problem, cand):
            return cand

    # Deterministic fallback for smaller dimensions.
    if n <= 24:
        from itertools import combinations

        for comb in combinations(all_idx, k):
            if is_active_set_feasible(problem, comb):
                return list(comb)

    raise RuntimeError("Could not find feasible active set for current bounds and k")


def _project_to_bounded_simplex(
    y: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    total: float = 1.0,
    max_iter: int = 120,
) -> List[float]:
    """Projection onto {x | sum x = total, lower <= x <= upper}."""
    if len(y) != len(lower) or len(y) != len(upper):
        raise ValueError("projection vectors must have the same length")

    lsum = sum(lower)
    usum = sum(upper)
    if lsum - total > 1e-9 or total - usum > 1e-9:
        raise ValueError("infeasible projection constraints")

    lo = min(lower[i] - y[i] for i in range(len(y)))
    hi = max(upper[i] - y[i] for i in range(len(y)))

    out = [0.0] * len(y)
    for _ in range(max_iter):
        theta = 0.5 * (lo + hi)
        s = 0.0
        for i in range(len(y)):
            xi = y[i] + theta
            if xi < lower[i]:
                xi = lower[i]
            elif xi > upper[i]:
                xi = upper[i]
            out[i] = xi
            s += xi

        if abs(s - total) <= 1e-11:
            return out[:]
        if s > total:
            hi = theta
        else:
            lo = theta

    # Final normalization pass.
    s = sum(out)
    if abs(s - total) > 1e-8:
        diff = total - s
        for _ in range(3):
            if abs(diff) <= 1e-10:
                break
            room = []
            for i in range(len(out)):
                up_room = upper[i] - out[i]
                low_room = out[i] - lower[i]
                room.append((up_room, low_room))
            if diff > 0:
                idx = max(range(len(out)), key=lambda i: room[i][0])
                delta = min(diff, room[idx][0])
                out[idx] += delta
                diff -= delta
            else:
                idx = max(range(len(out)), key=lambda i: room[i][1])
                delta = min(-diff, room[idx][1])
                out[idx] -= delta
                diff += delta
    return out


def repair_weights(
    problem: PortfolioProblem,
    raw_w: Sequence[float],
    rng: random.Random,
    preferred_active: Optional[Sequence[int]] = None,
) -> List[float]:
    """Map arbitrary vector to a feasible k-sparse portfolio under bounds."""
    n = problem.n
    if len(raw_w) != n:
        raise ValueError("raw_w has wrong dimension")

    if preferred_active is None:
        active = _pick_active_set_from_scores(problem, raw_w, rng)
    else:
        active = list(preferred_active)
        if len(active) != problem.k or not is_active_set_feasible(problem, active):
            active = _pick_active_set_from_scores(problem, raw_w, rng)

    y = []
    lb = []
    ub = []
    for i in active:
        y.append(max(0.0, float(raw_w[i])))
        lb.append(problem.lower[i])
        ub.append(problem.upper[i])

    projected = _project_to_bounded_simplex(y=y, lower=lb, upper=ub, total=1.0)

    w = [0.0] * n
    for pos, idx in enumerate(active):
        w[idx] = projected[pos]
    return w


def random_feasible_weights(problem: PortfolioProblem, rng: random.Random) -> List[float]:
    scores = [rng.random() for _ in range(problem.n)]
    active = _pick_active_set_from_scores(problem, scores, rng)
    y = [rng.random() for _ in range(problem.k)]
    lb = [problem.lower[i] for i in active]
    ub = [problem.upper[i] for i in active]
    projected = _project_to_bounded_simplex(y=y, lower=lb, upper=ub, total=1.0)
    w = [0.0] * problem.n
    for j, i in enumerate(active):
        w[i] = projected[j]
    return w


def compute_objective_ranges(
    problem: PortfolioProblem, rng: random.Random, samples: int = 240
) -> Dict[str, float]:
    risks = []
    rets = []
    for _ in range(samples):
        w = random_feasible_weights(problem, rng)
        risks.append(portfolio_risk(problem, w))
        rets.append(portfolio_return(problem, w))
    rmin = min(risks)
    rmax = max(risks)
    pmin = min(rets)
    pmax = max(rets)
    return {
        "risk_min": rmin,
        "risk_max": rmax,
        "ret_min": pmin,
        "ret_max": pmax,
        "risk_scale": max(rmax - rmin, 1e-9),
        "ret_scale": max(pmax - pmin, 1e-9),
    }


def scalarized_objective(
    alpha: float,
    risk: float,
    ret: float,
    ranges: Dict[str, float],
) -> float:
    risk_n = (risk - ranges["risk_min"]) / ranges["risk_scale"]
    ret_n = (ret - ranges["ret_min"]) / ranges["ret_scale"]
    return alpha * risk_n - (1.0 - alpha) * ret_n


def optimize_weighted_sum_single(
    problem: PortfolioProblem,
    alpha: float,
    ranges: Dict[str, float],
    rng: random.Random,
    restarts: int = 12,
    max_iters: int = 240,
    init_step: float = 0.25,
) -> Dict[str, Any]:
    """Solve one weighted-sum scalarization with projected gradient restarts."""
    best = None
    best_obj = float("inf")
    ret_scale = ranges["ret_scale"]
    risk_scale = ranges["risk_scale"]

    for _ in range(restarts):
        w = random_feasible_weights(problem, rng)
        step = init_step
        local_best = w[:]
        local_best_obj = float("inf")

        for _iter in range(max_iters):
            risk = portfolio_risk(problem, w)
            ret = portfolio_return(problem, w)
            obj = scalarized_objective(alpha, risk, ret, ranges)
            if obj < local_best_obj:
                local_best_obj = obj
                local_best = w[:]

            grad_risk = risk_gradient(problem, w)
            grad = []
            for i in range(problem.n):
                g = alpha * (grad_risk[i] / risk_scale) - (1.0 - alpha) * (
                    problem.mu[i] / ret_scale
                )
                grad.append(g)

            proposal = [w[i] - step * grad[i] for i in range(problem.n)]
            w_new = repair_weights(problem, proposal, rng)
            new_risk = portfolio_risk(problem, w_new)
            new_ret = portfolio_return(problem, w_new)
            new_obj = scalarized_objective(alpha, new_risk, new_ret, ranges)

            if new_obj <= obj - 1e-9:
                w = w_new
                step = min(step * 1.08, 2.0)
            else:
                step *= 0.5
                if step < 1e-5:
                    break

        final_risk = portfolio_risk(problem, local_best)
        final_ret = portfolio_return(problem, local_best)
        final_obj = scalarized_objective(alpha, final_risk, final_ret, ranges)
        if final_obj < best_obj:
            best_obj = final_obj
            best = {
                "weights": local_best,
                "risk": final_risk,
                "return": final_ret,
                "method": "weighted_sum",
                "alpha": alpha,
            }

    return best


def run_weighted_sum_front(
    problem: PortfolioProblem,
    points: int = 21,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    ranges = compute_objective_ranges(problem, rng, samples=220)
    alphas = [i / float(max(1, points - 1)) for i in range(points)]
    sols = []
    for alpha in alphas:
        sols.append(optimize_weighted_sum_single(problem, alpha, ranges, rng))
    return pareto_filter(sols)


def optimize_epsilon_single(
    problem: PortfolioProblem,
    epsilon_risk: float,
    ranges: Dict[str, float],
    rng: random.Random,
    restarts: int = 14,
    max_iters: int = 260,
    init_step: float = 0.18,
    penalty_start: float = 30.0,
) -> Dict[str, Any]:
    """Maximize return under risk <= epsilon, using adaptive penalty descent."""
    ret_scale = ranges["ret_scale"]
    risk_scale = ranges["risk_scale"]

    best_feasible = None
    best_feasible_ret = -float("inf")
    best_infeasible = None
    best_infeasible_pen = float("inf")

    for _ in range(restarts):
        w = random_feasible_weights(problem, rng)
        penalty = penalty_start

        for _outer in range(5):
            step = init_step
            for _inner in range(max_iters):
                risk = portfolio_risk(problem, w)
                ret = portfolio_return(problem, w)
                violation = max(0.0, risk - epsilon_risk)
                pen_obj = -(ret / ret_scale) + penalty * (violation / risk_scale) ** 2

                grad_risk = risk_gradient(problem, w)
                grad = []
                coeff = 0.0
                if violation > 0.0:
                    coeff = penalty * 2.0 * violation / (risk_scale * risk_scale)
                for i in range(problem.n):
                    g = -(problem.mu[i] / ret_scale) + coeff * grad_risk[i]
                    grad.append(g)

                proposal = [w[i] - step * grad[i] for i in range(problem.n)]
                w_new = repair_weights(problem, proposal, rng)
                new_risk = portfolio_risk(problem, w_new)
                new_ret = portfolio_return(problem, w_new)
                new_violation = max(0.0, new_risk - epsilon_risk)
                new_pen_obj = -(new_ret / ret_scale) + penalty * (
                    new_violation / risk_scale
                ) ** 2

                if new_pen_obj <= pen_obj - 1e-9:
                    w = w_new
                    step = min(step * 1.06, 1.0)
                else:
                    step *= 0.5
                    if step < 1e-5:
                        break

            end_risk = portfolio_risk(problem, w)
            if end_risk <= epsilon_risk * 1.0005:
                break
            penalty *= 5.0

        final_risk = portfolio_risk(problem, w)
        final_ret = portfolio_return(problem, w)
        violation = max(0.0, final_risk - epsilon_risk)
        penalty_score = violation * violation - final_ret
        if violation <= epsilon_risk * 5e-4 + 1e-7:
            if final_ret > best_feasible_ret:
                best_feasible_ret = final_ret
                best_feasible = {
                    "weights": w[:],
                    "risk": final_risk,
                    "return": final_ret,
                    "method": "epsilon_constraint",
                    "epsilon_risk": epsilon_risk,
                }
        elif penalty_score < best_infeasible_pen:
            best_infeasible_pen = penalty_score
            best_infeasible = {
                "weights": w[:],
                "risk": final_risk,
                "return": final_ret,
                "method": "epsilon_constraint",
                "epsilon_risk": epsilon_risk,
                "risk_violation": violation,
            }

    return best_feasible if best_feasible is not None else best_infeasible


def run_epsilon_constraint_front(
    problem: PortfolioProblem,
    points: int = 21,
    seed: int = 1,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    ranges = compute_objective_ranges(problem, rng, samples=260)
    low = ranges["risk_min"]
    high = ranges["risk_max"]
    eps_values = [low + (high - low) * i / float(max(1, points - 1)) for i in range(points)]
    sols = []
    for eps_risk in eps_values:
        sols.append(optimize_epsilon_single(problem, eps_risk, ranges, rng))
    return pareto_filter(sols)


def dominates(a: Dict[str, Any], b: Dict[str, Any], tol: float = 1e-10) -> bool:
    # minimize risk, maximize return
    no_worse = (a["risk"] <= b["risk"] + tol) and (a["return"] >= b["return"] - tol)
    strictly_better = (a["risk"] < b["risk"] - tol) or (a["return"] > b["return"] + tol)
    return no_worse and strictly_better


def pareto_filter(solutions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = [s for s in solutions if s is not None]
    out = []
    for i, s in enumerate(valid):
        dominated = False
        for j, t in enumerate(valid):
            if i == j:
                continue
            if dominates(t, s):
                dominated = True
                break
        if not dominated:
            out.append(s)

    # Remove near-duplicates for cleaner reports.
    out.sort(key=lambda x: (x["risk"], -x["return"]))
    unique = []
    seen = set()
    for s in out:
        key = (round(s["risk"], 10), round(s["return"], 10))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def _fast_non_dominated_sort(points: Sequence[Tuple[float, float]]) -> List[List[int]]:
    # points are in minimization form (f1, f2)
    n = len(points)
    dominates_list = [[] for _ in range(n)]
    dominated_count = [0] * n
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            pp = points[p]
            qq = points[q]
            if (pp[0] <= qq[0] and pp[1] <= qq[1]) and (pp[0] < qq[0] or pp[1] < qq[1]):
                dominates_list[p].append(q)
            elif (qq[0] <= pp[0] and qq[1] <= pp[1]) and (
                qq[0] < pp[0] or qq[1] < pp[1]
            ):
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def _crowding_distance(
    front_indices: Sequence[int], points: Sequence[Tuple[float, float]]
) -> Dict[int, float]:
    if not front_indices:
        return {}
    dist = {idx: 0.0 for idx in front_indices}
    for m in (0, 1):
        order = sorted(front_indices, key=lambda i: points[i][m])
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")
        fmin = points[order[0]][m]
        fmax = points[order[-1]][m]
        denom = max(fmax - fmin, 1e-12)
        for i in range(1, len(order) - 1):
            prev_v = points[order[i - 1]][m]
            next_v = points[order[i + 1]][m]
            dist[order[i]] += (next_v - prev_v) / denom
    return dist


def _tournament_pick(
    pop: Sequence[List[float]],
    ranks: Dict[int, int],
    crowding: Dict[int, float],
    rng: random.Random,
) -> List[float]:
    i = rng.randrange(len(pop))
    j = rng.randrange(len(pop))
    ri = ranks[i]
    rj = ranks[j]
    if ri < rj:
        return pop[i]
    if rj < ri:
        return pop[j]
    ci = crowding.get(i, 0.0)
    cj = crowding.get(j, 0.0)
    if ci > cj:
        return pop[i]
    if cj > ci:
        return pop[j]
    return pop[i] if rng.random() < 0.5 else pop[j]


def _crossover_and_mutate(
    p1: Sequence[float],
    p2: Sequence[float],
    mutation_rate: float,
    mutation_sigma: float,
    rng: random.Random,
) -> List[float]:
    child = [0.0] * len(p1)
    for i in range(len(p1)):
        a = rng.uniform(-0.15, 1.15)
        child[i] = a * p1[i] + (1.0 - a) * p2[i]
    for i in range(len(child)):
        if rng.random() < mutation_rate:
            child[i] += rng.gauss(0.0, mutation_sigma)
    return child


def run_nsga2_front(
    problem: PortfolioProblem,
    generations: int = 120,
    population_size: int = 80,
    seed: int = 7,
    mutation_rate: float = 0.08,
    mutation_sigma: float = 0.04,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pop = [random_feasible_weights(problem, rng) for _ in range(population_size)]

    for _gen in range(generations):
        # Evaluate current population.
        eval_points = []
        for w in pop:
            risk = portfolio_risk(problem, w)
            ret = portfolio_return(problem, w)
            eval_points.append((risk, -ret))  # minimization form

        fronts = _fast_non_dominated_sort(eval_points)
        ranks = {}
        crowding = {}
        for r, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = r
            crowding.update(_crowding_distance(front, eval_points))

        offspring = []
        while len(offspring) < population_size:
            p1 = _tournament_pick(pop, ranks, crowding, rng)
            p2 = _tournament_pick(pop, ranks, crowding, rng)
            child_raw = _crossover_and_mutate(
                p1, p2, mutation_rate=mutation_rate, mutation_sigma=mutation_sigma, rng=rng
            )
            child = repair_weights(problem, child_raw, rng)
            offspring.append(child)

        combined = pop + offspring
        comb_eval = []
        for w in combined:
            comb_eval.append((portfolio_risk(problem, w), -portfolio_return(problem, w)))

        comb_fronts = _fast_non_dominated_sort(comb_eval)
        new_pop = []
        for front in comb_fronts:
            if len(new_pop) + len(front) <= population_size:
                for idx in front:
                    new_pop.append(combined[idx])
            else:
                cd = _crowding_distance(front, comb_eval)
                ordered = sorted(front, key=lambda idx: cd[idx], reverse=True)
                needed = population_size - len(new_pop)
                for idx in ordered[:needed]:
                    new_pop.append(combined[idx])
                break
        pop = new_pop

    sols = []
    for w in pop:
        sols.append(
            {
                "weights": w[:],
                "risk": portfolio_risk(problem, w),
                "return": portfolio_return(problem, w),
                "method": "nsga2",
            }
        )
    return pareto_filter(sols)


def save_solutions_csv(path: str, solutions: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "risk", "return", "meta", "weights"])
        for s in solutions:
            method = s.get("method", "")
            risk = s.get("risk", "")
            ret = s.get("return", "")
            meta = []
            if "alpha" in s:
                meta.append("alpha={:.6f}".format(s["alpha"]))
            if "epsilon_risk" in s:
                meta.append("epsilon_risk={:.6f}".format(s["epsilon_risk"]))
            if "risk_violation" in s:
                meta.append("risk_violation={:.6e}".format(s["risk_violation"]))
            weights = ",".join("{:.10f}".format(x) for x in s["weights"])
            writer.writerow([method, risk, ret, ";".join(meta), weights])


def save_pareto_points_csv(path: str, solutions: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["risk", "return", "method"])
        for s in solutions:
            writer.writerow([s["risk"], s["return"], s.get("method", "")])


def _maybe_float(x: str) -> Optional[float]:
    try:
        return float(x.strip())
    except Exception:
        return None


def load_mu_csv(path: str) -> List[float]:
    mu = []
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("mu CSV is empty: {}".format(path))

    # Accept single-column numeric files or 2+ column files with numeric entry.
    for row in rows:
        vals = [_maybe_float(cell) for cell in row]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        mu.append(vals[-1])
    if not mu:
        raise ValueError("could not parse numeric mu values from {}".format(path))
    return mu


def load_covariance_csv(path: str) -> List[List[float]]:
    cov = []
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("covariance CSV is empty: {}".format(path))

    for row in rows:
        vals = [_maybe_float(cell) for cell in row]
        vals = [v for v in vals if v is not None]
        if vals:
            cov.append(vals)
    n = len(cov)
    if n == 0:
        raise ValueError("could not parse covariance values from {}".format(path))
    if any(len(row) != n for row in cov):
        raise ValueError("covariance CSV must be square after numeric parsing")
    return cov


def load_problem_from_csv(
    mu_csv: str,
    covariance_csv: str,
    k: int,
    lower: float = 0.001,
    upper: float = 1.0,
) -> PortfolioProblem:
    mu = load_mu_csv(mu_csv)
    cov = load_covariance_csv(covariance_csv)
    if len(mu) != len(cov):
        raise ValueError("mu length ({}) != covariance size ({})".format(len(mu), len(cov)))
    lower_bounds = [float(lower)] * len(mu)
    upper_bounds = [float(upper)] * len(mu)
    return PortfolioProblem(mu=mu, covariance=cov, k=k, lower_bounds=lower_bounds, upper_bounds=upper_bounds)


def load_orlib_portfolio_file(path: str) -> Dict[str, Any]:
    """Parse OR-Library portfolio file (port1.txt ... port5.txt).

    Format (whitespace-delimited):
      N
      (mu_i, sigma_i) for i=1..N
      (i, j, corr_ij) for all pairs (typically i<=j, including diagonals)
    """
    # Accept UTF-8 files with or without BOM.
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()
    if not text:
        raise ValueError("OR-Library file is empty: {}".format(path))

    toks = text.split()
    if len(toks) < 1:
        raise ValueError("OR-Library file has no tokens: {}".format(path))

    try:
        n = int(float(toks[0]))
    except Exception:
        raise ValueError("Could not parse N from first token in {}".format(path))

    need = 1 + 2 * n
    if len(toks) < need:
        raise ValueError(
            "OR-Library file too short for N={} (need at least {}, got {})".format(
                n, need, len(toks)
            )
        )

    mu = []
    sigma = []
    p = 1
    for _ in range(n):
        mu.append(float(toks[p]))
        sigma.append(float(toks[p + 1]))
        p += 2

    corr = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        corr[i][i] = 1.0

    triples = (len(toks) - p) // 3
    if triples <= 0:
        raise ValueError("No correlation entries found in {}".format(path))

    for _ in range(triples):
        i_raw = int(float(toks[p]))
        j_raw = int(float(toks[p + 1]))
        c = float(toks[p + 2])
        p += 3
        i = i_raw - 1
        j = j_raw - 1
        if i < 0 or i >= n or j < 0 or j >= n:
            continue
        corr[i][j] = c
        corr[j][i] = c

    # Fill potentially missing entries conservatively with 0 correlation.
    for i in range(n):
        corr[i][i] = 1.0
        for j in range(i + 1, n):
            cij = corr[i][j]
            cji = corr[j][i]
            if abs(cij) < 1e-15 and abs(cji) > 1e-15:
                corr[i][j] = cji
            elif abs(cji) < 1e-15 and abs(cij) > 1e-15:
                corr[j][i] = cij

    cov = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cov[i][j] = corr[i][j] * sigma[i] * sigma[j]

    return {"n": n, "mu": mu, "sigma": sigma, "corr": corr, "cov": cov}


def load_problem_from_orlib(
    orlib_port_file: str,
    k: int,
    lower: float = 0.001,
    upper: float = 1.0,
) -> PortfolioProblem:
    parsed = load_orlib_portfolio_file(orlib_port_file)
    n = parsed["n"]
    lower_bounds = [float(lower)] * n
    upper_bounds = [float(upper)] * n
    return PortfolioProblem(
        mu=parsed["mu"],
        covariance=parsed["cov"],
        k=k,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


def generate_synthetic_problem(
    n_assets: int = 30,
    k: int = 8,
    seed: int = 42,
    lower: float = 0.01,
    upper: float = 0.35,
) -> PortfolioProblem:
    """Generate a positive-semidefinite covariance matrix and realistic returns."""
    rng = random.Random(seed)
    # Expected returns: roughly [-2%, 5%] in arbitrary time unit.
    mu = [rng.uniform(-0.02, 0.05) for _ in range(n_assets)]

    # Factor-model-style covariance: A A^T + diagonal noise.
    factors = 4
    A = []
    for _ in range(n_assets):
        row = [rng.uniform(-0.08, 0.12) for _ in range(factors)]
        A.append(row)

    cov = [[0.0 for _ in range(n_assets)] for _ in range(n_assets)]
    for i in range(n_assets):
        for j in range(n_assets):
            s = 0.0
            for f in range(factors):
                s += A[i][f] * A[j][f]
            if i == j:
                s += rng.uniform(0.0005, 0.004)
            cov[i][j] = s

    lower_bounds = [float(lower)] * n_assets
    upper_bounds = [float(upper)] * n_assets
    return PortfolioProblem(
        mu=mu,
        covariance=cov,
        k=k,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
