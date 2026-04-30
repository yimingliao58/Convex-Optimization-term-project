"""Batch experiments on OR-Library portfolio datasets."""

import argparse
import csv
import json
import os
import time
from typing import Dict, List

from portfolio_methods import (
    load_orlib_portfolio_file,
    load_problem_from_orlib,
    pareto_filter,
    run_epsilon_constraint_front,
    run_nsga2_front,
    run_weighted_sum_front,
    save_pareto_points_csv,
    save_solutions_csv,
)


def parse_int_list(text: str) -> List[int]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def save_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "n_assets",
                "k",
                "method",
                "pareto_points",
                "min_risk",
                "max_return",
                "elapsed_sec",
                "output_subdir",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["dataset"],
                    r["n_assets"],
                    r["k"],
                    r["method"],
                    r["pareto_points"],
                    r["min_risk"],
                    r["max_return"],
                    r["elapsed_sec"],
                    r["output_subdir"],
                ]
            )


def summarize_solution_set(
    dataset: str,
    n_assets: int,
    k: int,
    method: str,
    sols: List[Dict[str, object]],
    elapsed_sec: float,
    output_subdir: str,
) -> Dict[str, object]:
    if not sols:
        return {
            "dataset": dataset,
            "n_assets": n_assets,
            "k": k,
            "method": method,
            "pareto_points": 0,
            "min_risk": "",
            "max_return": "",
            "elapsed_sec": elapsed_sec,
            "output_subdir": output_subdir,
        }
    return {
        "dataset": dataset,
        "n_assets": n_assets,
        "k": k,
        "method": method,
        "pareto_points": len(sols),
        "min_risk": min(s["risk"] for s in sols),
        "max_return": max(s["return"] for s in sols),
        "elapsed_sec": elapsed_sec,
        "output_subdir": output_subdir,
    }


def run_one(
    out_dir: str,
    port_file: str,
    k: int,
    lower: float,
    upper: float,
    weighted_points: int,
    epsilon_points: int,
    seed: int,
    do_nsga2: bool,
    nsga2_generations: int,
    nsga2_population: int,
):
    parsed = load_orlib_portfolio_file(port_file)
    n_assets = int(parsed["n"])
    dataset = os.path.splitext(os.path.basename(port_file))[0]
    subdir = "{}_k{}".format(dataset, k)
    target = os.path.join(out_dir, subdir)
    os.makedirs(target, exist_ok=True)

    problem = load_problem_from_orlib(port_file, k=k, lower=lower, upper=upper)

    t0 = time.time()
    weighted = run_weighted_sum_front(
        problem=problem, points=weighted_points, seed=seed + 11 + k
    )
    t_weighted = time.time() - t0

    t1 = time.time()
    epsilon = run_epsilon_constraint_front(
        problem=problem, points=epsilon_points, seed=seed + 23 + k
    )
    t_epsilon = time.time() - t1

    nsga2 = []
    t_nsga2 = 0.0
    if do_nsga2:
        t2 = time.time()
        nsga2 = run_nsga2_front(
            problem=problem,
            generations=nsga2_generations,
            population_size=nsga2_population,
            seed=seed + 37 + k,
        )
        t_nsga2 = time.time() - t2

    combined = pareto_filter(weighted + epsilon + nsga2)

    save_solutions_csv(os.path.join(target, "weighted_sum.csv"), weighted)
    save_solutions_csv(os.path.join(target, "epsilon_constraint.csv"), epsilon)
    if nsga2:
        save_solutions_csv(os.path.join(target, "nsga2.csv"), nsga2)
    save_solutions_csv(os.path.join(target, "all_pareto.csv"), combined)
    save_pareto_points_csv(os.path.join(target, "pareto_points.csv"), combined)

    meta = {
        "dataset": dataset,
        "port_file": os.path.abspath(port_file),
        "n_assets": n_assets,
        "k": k,
        "lower": lower,
        "upper": upper,
        "weighted_points": weighted_points,
        "epsilon_points": epsilon_points,
        "run_nsga2": do_nsga2,
        "nsga2_generations": nsga2_generations,
        "nsga2_population": nsga2_population,
        "seed": seed,
        "elapsed_weighted_sec": t_weighted,
        "elapsed_epsilon_sec": t_epsilon,
        "elapsed_nsga2_sec": t_nsga2,
        "elapsed_total_sec": t_weighted + t_epsilon + t_nsga2,
    }
    with open(os.path.join(target, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    rows = []
    rows.append(
        summarize_solution_set(
            dataset, n_assets, k, "weighted_sum", weighted, t_weighted, subdir
        )
    )
    rows.append(
        summarize_solution_set(
            dataset, n_assets, k, "epsilon_constraint", epsilon, t_epsilon, subdir
        )
    )
    if do_nsga2:
        rows.append(
            summarize_solution_set(dataset, n_assets, k, "nsga2", nsga2, t_nsga2, subdir)
        )
    rows.append(
        summarize_solution_set(
            dataset,
            n_assets,
            k,
            "combined_pareto",
            combined,
            t_weighted + t_epsilon + t_nsga2,
            subdir,
        )
    )
    return rows, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OR-Library batch experiments.")
    parser.add_argument(
        "--orlib-dir",
        type=str,
        default="data/orlib",
        help="Directory containing port1..port5 files",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="port1,port2,port3,port4,port5",
        help="Comma-separated dataset names without extension",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="2,4,8,16",
        help="Comma-separated K values",
    )
    parser.add_argument("--lower", type=float, default=0.001)
    parser.add_argument("--upper", type=float, default=1.0)
    parser.add_argument("--weighted-points", type=int, default=21)
    parser.add_argument("--epsilon-points", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-nsga2",
        action="store_true",
        help="Run NSGA-II for selected K values",
    )
    parser.add_argument(
        "--nsga2-k-values",
        type=str,
        default="8",
        help="Comma-separated K values where NSGA-II should run",
    )
    parser.add_argument("--nsga2-generations", type=int, default=60)
    parser.add_argument("--nsga2-population", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default="results_orlib_batch")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    k_values = parse_int_list(args.k_values)
    nsga2_k_values = set(parse_int_list(args.nsga2_k_values))
    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    run_log = []
    t_all = time.time()

    for ds in datasets:
        port_file = os.path.join(args.orlib_dir, ds + ".txt")
        if not os.path.exists(port_file):
            raise FileNotFoundError("Missing dataset file: {}".format(port_file))

        parsed = load_orlib_portfolio_file(port_file)
        n_assets = int(parsed["n"])
        allowed_k = [k for k in k_values if 1 <= k <= n_assets]
        print("Dataset {} (n={}): K={}".format(ds, n_assets, allowed_k))

        for k in allowed_k:
            if (k * args.lower) > 1.0 + 1e-12 or (k * args.upper) < 1.0 - 1e-12:
                print(
                    "  -> Skipping k={} due to infeasible bounds (lower={}, upper={})".format(
                        k, args.lower, args.upper
                    )
                )
                continue

            do_nsga2 = bool(args.run_nsga2 and (k in nsga2_k_values))
            print("  -> Running k={} (nsga2={})".format(k, do_nsga2))
            rows, meta = run_one(
                out_dir=args.output_dir,
                port_file=port_file,
                k=k,
                lower=args.lower,
                upper=args.upper,
                weighted_points=args.weighted_points,
                epsilon_points=args.epsilon_points,
                seed=args.seed,
                do_nsga2=do_nsga2,
                nsga2_generations=args.nsga2_generations,
                nsga2_population=args.nsga2_population,
            )
            summary_rows.extend(rows)
            run_log.append(meta)

    elapsed = time.time() - t_all
    save_summary_csv(os.path.join(args.output_dir, "summary.csv"), summary_rows)
    with open(os.path.join(args.output_dir, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "elapsed_total_sec": elapsed,
                "config": vars(args),
                "runs": run_log,
            },
            f,
            indent=2,
        )

    print(
        "Saved summary: {}".format(
            os.path.abspath(os.path.join(args.output_dir, "summary.csv"))
        )
    )
    print("Batch elapsed sec: {:.2f}".format(elapsed))


if __name__ == "__main__":
    main()
