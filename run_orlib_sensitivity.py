"""Controlled sensitivity experiments for mu and Sigma perturbations."""

import argparse
import csv
import json
import os
import random
import time

import matplotlib.pyplot as plt

from portfolio_methods import (
    PortfolioProblem,
    load_orlib_portfolio_file,
    pareto_filter,
    run_epsilon_constraint_front,
    run_nsga2_front,
    run_weighted_sum_front,
    save_solutions_csv,
)


def std(values):
    if not values:
        return 0.0
    m = sum(values) / len(values)
    var = sum((x - m) * (x - m) for x in values) / len(values)
    return var ** 0.5


def diag_of(mat):
    return [mat[i][i] for i in range(len(mat))]


def perturb_mu_noise(mu, level, seed):
    rng = random.Random(seed)
    s = max(std(mu), 1e-12)
    return [m + level * s * rng.gauss(0.0, 1.0) for m in mu]


def perturb_sigma_scale(cov, factor):
    n = len(cov)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = cov[i][j] * factor
    return out


def perturb_sigma_shrink_corr(cov, shrink):
    # Sigma' = (1-shrink)*Sigma + shrink*Diag(Sigma), PSD-preserving.
    n = len(cov)
    d = diag_of(cov)
    out = [[0.0] * n for _ in range(n)]
    a = max(0.0, min(1.0, shrink))
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i][j] = (1.0 - a) * cov[i][j] + a * d[i]
            else:
                out[i][j] = (1.0 - a) * cov[i][j]
    return out


def make_scenarios(mu, cov, seed):
    return {
        "base": (mu, cov, "No perturbation"),
        "mu_noise_10": (perturb_mu_noise(mu, 0.10, seed + 101), cov, "mu + 10% std noise"),
        "mu_noise_25": (perturb_mu_noise(mu, 0.25, seed + 202), cov, "mu + 25% std noise"),
        "sigma_scale_080": (mu, perturb_sigma_scale(cov, 0.80), "Sigma * 0.80"),
        "sigma_scale_120": (mu, perturb_sigma_scale(cov, 1.20), "Sigma * 1.20"),
        "sigma_shrink_30": (
            mu,
            perturb_sigma_shrink_corr(cov, 0.30),
            "Sigma shrink-corr 30%",
        ),
    }


def save_summary_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "k",
                "scenario",
                "scenario_desc",
                "method",
                "pareto_points",
                "min_risk",
                "max_return",
                "elapsed_sec",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["dataset"],
                    r["k"],
                    r["scenario"],
                    r["scenario_desc"],
                    r["method"],
                    r["pareto_points"],
                    r["min_risk"],
                    r["max_return"],
                    r["elapsed_sec"],
                ]
            )


def summarize(method, sols, elapsed_sec, dataset, k, scenario, scenario_desc):
    if not sols:
        return {
            "dataset": dataset,
            "k": k,
            "scenario": scenario,
            "scenario_desc": scenario_desc,
            "method": method,
            "pareto_points": 0,
            "min_risk": "",
            "max_return": "",
            "elapsed_sec": elapsed_sec,
        }
    return {
        "dataset": dataset,
        "k": k,
        "scenario": scenario,
        "scenario_desc": scenario_desc,
        "method": method,
        "pareto_points": len(sols),
        "min_risk": min(s["risk"] for s in sols),
        "max_return": max(s["return"] for s in sols),
        "elapsed_sec": elapsed_sec,
    }


def plot_sensitivity(summary_rows, out_dir):
    scenarios = []
    for r in summary_rows:
        if r["scenario"] not in scenarios:
            scenarios.append(r["scenario"])
    scenario_order = scenarios
    datasets = sorted(set(r["dataset"] for r in summary_rows))

    # Combined min risk
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for i, ds in enumerate(datasets):
        ax = axes[i]
        vals = []
        for sc in scenario_order:
            xs = [
                r
                for r in summary_rows
                if r["dataset"] == ds
                and r["method"] == "combined_pareto"
                and r["scenario"] == sc
            ]
            vals.append(xs[0]["min_risk"] if xs and xs[0]["min_risk"] != "" else None)
        x = list(range(len(scenario_order)))
        y = [float(v) if v is not None else float("nan") for v in vals]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#d62728")
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_order, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Combined Min Risk")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig08_sensitivity_min_risk.png"), dpi=260, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig08_sensitivity_min_risk.pdf"), bbox_inches="tight")
    plt.close(fig)

    # Combined max return
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 4), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for i, ds in enumerate(datasets):
        ax = axes[i]
        vals = []
        for sc in scenario_order:
            xs = [
                r
                for r in summary_rows
                if r["dataset"] == ds
                and r["method"] == "combined_pareto"
                and r["scenario"] == sc
            ]
            vals.append(xs[0]["max_return"] if xs and xs[0]["max_return"] != "" else None)
        x = list(range(len(scenario_order)))
        y = [float(v) if v is not None else float("nan") for v in vals]
        ax.plot(x, y, marker="o", linewidth=2.0, color="#1f77b4")
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_order, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Combined Max Return")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig09_sensitivity_max_return.png"), dpi=260, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig09_sensitivity_max_return.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run mu/Sigma sensitivity experiments.")
    parser.add_argument("--orlib-dir", type=str, default="data/orlib")
    parser.add_argument("--datasets", type=str, default="port1,port4")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--lower", type=float, default=0.001)
    parser.add_argument("--upper", type=float, default=1.0)
    parser.add_argument("--weighted-points", type=int, default=13)
    parser.add_argument("--epsilon-points", type=int, default=13)
    parser.add_argument("--nsga2-generations", type=int, default=40)
    parser.add_argument("--nsga2-population", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_orlib_batch_final/sensitivity_analysis",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    summary_rows = []
    run_meta = []
    all_t0 = time.time()

    for ds in datasets:
        port_file = os.path.join(args.orlib_dir, ds + ".txt")
        parsed = load_orlib_portfolio_file(port_file)
        base_mu = parsed["mu"]
        base_cov = parsed["cov"]
        n = parsed["n"]
        scenarios = make_scenarios(base_mu, base_cov, args.seed + len(ds))

        for sc_name, (mu, cov, sc_desc) in scenarios.items():
            run_dir = os.path.join(args.output_dir, "{}_k{}_{}".format(ds, args.k, sc_name))
            os.makedirs(run_dir, exist_ok=True)
            problem = PortfolioProblem(
                mu=mu,
                covariance=cov,
                k=args.k,
                lower_bounds=[args.lower] * n,
                upper_bounds=[args.upper] * n,
            )

            t0 = time.time()
            weighted = run_weighted_sum_front(
                problem=problem,
                points=args.weighted_points,
                seed=args.seed + 11,
            )
            tw = time.time() - t0

            t1 = time.time()
            epsilon = run_epsilon_constraint_front(
                problem=problem,
                points=args.epsilon_points,
                seed=args.seed + 23,
            )
            te = time.time() - t1

            t2 = time.time()
            nsga2 = run_nsga2_front(
                problem=problem,
                generations=args.nsga2_generations,
                population_size=args.nsga2_population,
                seed=args.seed + 37,
            )
            tn = time.time() - t2

            combined = pareto_filter(weighted + epsilon + nsga2)
            tc = tw + te + tn

            save_solutions_csv(os.path.join(run_dir, "weighted_sum.csv"), weighted)
            save_solutions_csv(os.path.join(run_dir, "epsilon_constraint.csv"), epsilon)
            save_solutions_csv(os.path.join(run_dir, "nsga2.csv"), nsga2)
            save_solutions_csv(os.path.join(run_dir, "all_pareto.csv"), combined)

            summary_rows.append(summarize("weighted_sum", weighted, tw, ds, args.k, sc_name, sc_desc))
            summary_rows.append(
                summarize("epsilon_constraint", epsilon, te, ds, args.k, sc_name, sc_desc)
            )
            summary_rows.append(summarize("nsga2", nsga2, tn, ds, args.k, sc_name, sc_desc))
            summary_rows.append(
                summarize("combined_pareto", combined, tc, ds, args.k, sc_name, sc_desc)
            )

            run_meta.append(
                {
                    "dataset": ds,
                    "scenario": sc_name,
                    "scenario_desc": sc_desc,
                    "k": args.k,
                    "elapsed_weighted_sec": tw,
                    "elapsed_epsilon_sec": te,
                    "elapsed_nsga2_sec": tn,
                    "elapsed_total_sec": tc,
                    "run_dir": os.path.abspath(run_dir),
                }
            )
            print("Done {} {} k={}".format(ds, sc_name, args.k))

    save_summary_csv(os.path.join(args.output_dir, "sensitivity_summary.csv"), summary_rows)
    with open(os.path.join(args.output_dir, "sensitivity_log.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "elapsed_total_sec": time.time() - all_t0,
                "config": vars(args),
                "runs": run_meta,
            },
            f,
            indent=2,
        )

    plot_sensitivity(summary_rows, args.output_dir)
    print("Sensitivity analysis saved in: {}".format(os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main()

