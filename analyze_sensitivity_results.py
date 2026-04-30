"""Build comparison tables and plots from sensitivity_summary.csv."""

import argparse
import csv
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def _to_int(text: str) -> int:
    return int(text.strip())


def _to_float(text: str) -> float:
    return float(text.strip())


def _safe_pct(delta: float, base: float) -> float:
    if abs(base) <= 1e-15:
        return math.nan
    return 100.0 * delta / base


def _mean(vals: List[float]) -> float:
    keep = [x for x in vals if x == x]
    if not keep:
        return math.nan
    return sum(keep) / float(len(keep))


def load_summary(path: str) -> List[Dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        needed = {
            "dataset",
            "k",
            "scenario",
            "scenario_desc",
            "method",
            "pareto_points",
            "min_risk",
            "max_return",
            "elapsed_sec",
        }
        missing = [c for c in needed if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Missing columns in summary CSV: {}".format(missing))

        for r in reader:
            rows.append(
                {
                    "dataset": r["dataset"],
                    "k": _to_int(r["k"]),
                    "scenario": r["scenario"],
                    "scenario_desc": r["scenario_desc"],
                    "method": r["method"],
                    "pareto_points": _to_int(r["pareto_points"]),
                    "min_risk": _to_float(r["min_risk"]) if r["min_risk"] else math.nan,
                    "max_return": _to_float(r["max_return"]) if r["max_return"] else math.nan,
                    "elapsed_sec": _to_float(r["elapsed_sec"]) if r["elapsed_sec"] else math.nan,
                }
            )
    return rows


def build_comparison_rows(
    rows: List[Dict[str, object]], base_scenario: str
) -> List[Dict[str, object]]:
    by_key = {}
    for r in rows:
        key = (r["dataset"], r["k"], r["method"], r["scenario"])
        by_key[key] = r

    out = []
    for r in rows:
        if r["scenario"] == base_scenario:
            continue

        base_key = (r["dataset"], r["k"], r["method"], base_scenario)
        base = by_key.get(base_key)
        if base is None:
            continue

        d_risk = r["min_risk"] - base["min_risk"]
        d_ret = r["max_return"] - base["max_return"]
        d_pts = r["pareto_points"] - base["pareto_points"]
        d_time = r["elapsed_sec"] - base["elapsed_sec"]

        out.append(
            {
                "dataset": r["dataset"],
                "k": r["k"],
                "method": r["method"],
                "scenario": r["scenario"],
                "scenario_desc": r["scenario_desc"],
                "base_min_risk": base["min_risk"],
                "scenario_min_risk": r["min_risk"],
                "delta_min_risk": d_risk,
                "delta_min_risk_pct": _safe_pct(d_risk, base["min_risk"]),
                "base_max_return": base["max_return"],
                "scenario_max_return": r["max_return"],
                "delta_max_return": d_ret,
                "delta_max_return_pct": _safe_pct(d_ret, base["max_return"]),
                "base_pareto_points": base["pareto_points"],
                "scenario_pareto_points": r["pareto_points"],
                "delta_pareto_points": d_pts,
                "base_elapsed_sec": base["elapsed_sec"],
                "scenario_elapsed_sec": r["elapsed_sec"],
                "delta_elapsed_sec": d_time,
            }
        )
    return out


def write_comparison_csv(path: str, rows: List[Dict[str, object]]) -> None:
    cols = [
        "dataset",
        "k",
        "method",
        "scenario",
        "scenario_desc",
        "base_min_risk",
        "scenario_min_risk",
        "delta_min_risk",
        "delta_min_risk_pct",
        "base_max_return",
        "scenario_max_return",
        "delta_max_return",
        "delta_max_return_pct",
        "base_pareto_points",
        "scenario_pareto_points",
        "delta_pareto_points",
        "base_elapsed_sec",
        "scenario_elapsed_sec",
        "delta_elapsed_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_method_aggregate_csv(path: str, rows: List[Dict[str, object]]) -> None:
    keys = {}
    for r in rows:
        key = (r["method"], r["scenario"])
        keys.setdefault(key, []).append(r)

    out = []
    for (method, scenario), group in sorted(keys.items()):
        out.append(
            {
                "method": method,
                "scenario": scenario,
                "n": len(group),
                "mean_delta_min_risk_pct": _mean([g["delta_min_risk_pct"] for g in group]),
                "mean_delta_max_return_pct": _mean([g["delta_max_return_pct"] for g in group]),
                "mean_delta_pareto_points": _mean([float(g["delta_pareto_points"]) for g in group]),
                "mean_delta_elapsed_sec": _mean([g["delta_elapsed_sec"] for g in group]),
            }
        )

    cols = [
        "method",
        "scenario",
        "n",
        "mean_delta_min_risk_pct",
        "mean_delta_max_return_pct",
        "mean_delta_pareto_points",
        "mean_delta_elapsed_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow(r)


def _plot_heatmap(
    datasets: List[str],
    scenarios: List[str],
    matrix: List[List[float]],
    title: str,
    cbar_label: str,
    out_png: str,
    out_pdf: str,
) -> None:
    fig_w = max(7.0, 1.4 * len(scenarios))
    fig_h = max(3.2, 1.0 + 1.0 * len(datasets))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_title(title)

    for i in range(len(datasets)):
        for j in range(len(scenarios)):
            v = matrix[i][j]
            txt = "nan" if v != v else "{:.2f}".format(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_combined_heatmaps(
    cmp_rows: List[Dict[str, object]],
    out_dir: str,
    method: str = "combined_pareto",
) -> None:
    rows = [r for r in cmp_rows if r["method"] == method]
    if not rows:
        return

    datasets = sorted({r["dataset"] for r in rows})
    scenarios = sorted({r["scenario"] for r in rows})
    risk_map = {(r["dataset"], r["scenario"]): r["delta_min_risk_pct"] for r in rows}
    ret_map = {(r["dataset"], r["scenario"]): r["delta_max_return_pct"] for r in rows}

    risk_m = []
    ret_m = []
    for ds in datasets:
        rr = []
        pp = []
        for sc in scenarios:
            rr.append(risk_map.get((ds, sc), math.nan))
            pp.append(ret_map.get((ds, sc), math.nan))
        risk_m.append(rr)
        ret_m.append(pp)

    _plot_heatmap(
        datasets=datasets,
        scenarios=scenarios,
        matrix=risk_m,
        title="Sensitivity vs Base: Min Risk Change (%) [{}]".format(method),
        cbar_label="Delta Min Risk (%)",
        out_png=os.path.join(out_dir, "fig10_sensitivity_delta_min_risk_pct_heatmap.png"),
        out_pdf=os.path.join(out_dir, "fig10_sensitivity_delta_min_risk_pct_heatmap.pdf"),
    )
    _plot_heatmap(
        datasets=datasets,
        scenarios=scenarios,
        matrix=ret_m,
        title="Sensitivity vs Base: Max Return Change (%) [{}]".format(method),
        cbar_label="Delta Max Return (%)",
        out_png=os.path.join(out_dir, "fig11_sensitivity_delta_max_return_pct_heatmap.png"),
        out_pdf=os.path.join(out_dir, "fig11_sensitivity_delta_max_return_pct_heatmap.pdf"),
    )


def plot_method_bars(cmp_rows: List[Dict[str, object]], out_dir: str) -> None:
    methods = sorted({r["method"] for r in cmp_rows})
    scenarios = sorted({r["scenario"] for r in cmp_rows})
    if not methods or not scenarios:
        return

    risk_by = {}
    ret_by = {}
    for m in methods:
        risk_by[m] = []
        ret_by[m] = []
        for sc in scenarios:
            group = [r for r in cmp_rows if r["method"] == m and r["scenario"] == sc]
            risk_by[m].append(_mean([g["delta_min_risk_pct"] for g in group]))
            ret_by[m].append(_mean([g["delta_max_return_pct"] for g in group]))

    x = list(range(len(scenarios)))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(scenarios)), 4.5))
    for idx, m in enumerate(methods):
        xs = [v - 0.4 + width * (idx + 0.5) for v in x]
        ys = [0.0 if y != y else y for y in risk_by[m]]
        ax.bar(xs, ys, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel("Mean Delta Min Risk (%)")
    ax.set_title("Method Sensitivity vs Base: Min Risk")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig12_sensitivity_method_delta_min_risk_pct.png"), dpi=260, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig12_sensitivity_method_delta_min_risk_pct.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(scenarios)), 4.5))
    for idx, m in enumerate(methods):
        xs = [v - 0.4 + width * (idx + 0.5) for v in x]
        ys = [0.0 if y != y else y for y in ret_by[m]]
        ax.bar(xs, ys, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel("Mean Delta Max Return (%)")
    ax.set_title("Method Sensitivity vs Base: Max Return")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig13_sensitivity_method_delta_max_return_pct.png"), dpi=260, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "fig13_sensitivity_method_delta_max_return_pct.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison tables and plots from sensitivity summary."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results_orlib_batch_final/sensitivity_analysis/sensitivity_summary.csv",
        help="Path to sensitivity_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Output folder for tables/plots. Default: summary file folder.",
    )
    parser.add_argument(
        "--base-scenario",
        type=str,
        default="base",
        help="Scenario name used as baseline for deltas",
    )
    args = parser.parse_args()

    summary_path = args.summary_csv
    if not os.path.isfile(summary_path):
        raise FileNotFoundError("Missing summary CSV: {}".format(summary_path))

    out_dir = args.output_dir.strip()
    if not out_dir:
        out_dir = os.path.dirname(os.path.abspath(summary_path))
    os.makedirs(out_dir, exist_ok=True)

    rows = load_summary(summary_path)
    cmp_rows = build_comparison_rows(rows, base_scenario=args.base_scenario)

    write_comparison_csv(os.path.join(out_dir, "sensitivity_comparison_vs_base.csv"), cmp_rows)
    write_method_aggregate_csv(
        os.path.join(out_dir, "sensitivity_method_aggregate_vs_base.csv"),
        cmp_rows,
    )
    plot_combined_heatmaps(cmp_rows, out_dir, method="combined_pareto")
    plot_method_bars(cmp_rows, out_dir)

    print("Saved comparison tables and plots to: {}".format(os.path.abspath(out_dir)))
    print("Comparison rows: {}".format(len(cmp_rows)))


if __name__ == "__main__":
    main()
