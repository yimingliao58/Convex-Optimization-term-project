import argparse
import csv
import os
import re

import matplotlib.pyplot as plt


METHOD_ORDER = ["weighted_sum", "epsilon_constraint", "nsga2"]
METHOD_COLORS = {
    "weighted_sum": "#1f77b4",
    "epsilon_constraint": "#ff7f0e",
    "nsga2": "#2ca02c",
}


def parse_run_dir_name(name):
    m = re.match(r"^(port\d+)_k(\d+)$", name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def parse_weights(weights_str):
    vals = []
    for tok in weights_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals


def load_method_solutions(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "method": r.get("method", ""),
                    "risk": float(r["risk"]),
                    "return": float(r["return"]),
                    "weights": parse_weights(r["weights"]),
                }
            )
    return rows


def pick_tradeoff_points(solutions):
    if not solutions:
        return {}
    min_risk = min(solutions, key=lambda x: x["risk"])
    max_ret = max(solutions, key=lambda x: x["return"])

    rmin = min(s["risk"] for s in solutions)
    rmax = max(s["risk"] for s in solutions)
    pmin = min(s["return"] for s in solutions)
    pmax = max(s["return"] for s in solutions)
    rs = max(rmax - rmin, 1e-12)
    ps = max(pmax - pmin, 1e-12)

    def balanced_score(s):
        # Minimize distance to ideal (low risk, high return).
        rn = (s["risk"] - rmin) / rs
        pn = (pmax - s["return"]) / ps
        return rn * rn + pn * pn

    balanced = min(solutions, key=balanced_score)
    return {
        "min_risk": min_risk,
        "balanced": balanced,
        "max_return": max_ret,
    }


def composition_metrics(weights, eps=1e-10):
    active = [w for w in weights if w > eps]
    if not active:
        return {
            "active_count": 0,
            "hhi": 0.0,
            "effective_n": 0.0,
            "top1_share": 0.0,
            "top3_share": 0.0,
        }
    sorted_w = sorted(active, reverse=True)
    hhi = sum(w * w for w in active)
    return {
        "active_count": len(active),
        "hhi": hhi,
        "effective_n": (1.0 / hhi) if hhi > 0 else 0.0,
        "top1_share": sorted_w[0],
        "top3_share": sum(sorted_w[:3]),
    }


def save_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[h] for h in header])


def avg(values):
    return sum(values) / len(values) if values else 0.0


def plot_hhi_tradeoff(rows, out_png, out_pdf):
    tradeoffs = ["min_risk", "balanced", "max_return"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = list(range(len(tradeoffs)))
    width = 0.22

    for i, m in enumerate(METHOD_ORDER):
        ys = []
        for t in tradeoffs:
            vals = [r["hhi"] for r in rows if r["method"] == m and r["tradeoff"] == t]
            ys.append(avg(vals))
        xpos = [v + (i - 1) * width for v in x]
        ax.bar(
            xpos,
            ys,
            width=width,
            label=m,
            color=METHOD_COLORS[m],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tradeoffs)
    ax.set_ylabel("Average HHI (Higher = More Concentrated)")
    ax.set_title("Portfolio Concentration by Trade-Off Region")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_effective_n_by_k(rows, out_png, out_pdf):
    ks = sorted(set(r["k"] for r in rows))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    for m in METHOD_ORDER:
        ys = []
        for k in ks:
            vals = [
                r["effective_n"]
                for r in rows
                if r["method"] == m and r["tradeoff"] == "balanced" and r["k"] == k
            ]
            ys.append(avg(vals))
        ax.plot(
            ks,
            ys,
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[m],
            label=m,
        )
    ax.set_xlabel("K")
    ax.set_ylabel("Effective Number of Assets (1/HHI)")
    ax.set_title("Diversification vs Cardinality (Balanced Point)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze composition from final batch results.")
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="results_orlib_batch_final",
        help="Batch results root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_orlib_batch_final/composition_analysis",
        help="Output analysis directory",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for name in sorted(os.listdir(args.batch_dir)):
        run_dir = os.path.join(args.batch_dir, name)
        if not os.path.isdir(run_dir):
            continue
        ds, k = parse_run_dir_name(name)
        if not ds:
            continue

        for method in METHOD_ORDER:
            csv_path = os.path.join(run_dir, "{}.csv".format(method))
            sols = load_method_solutions(csv_path)
            if not sols:
                continue
            picks = pick_tradeoff_points(sols)
            for tradeoff, sol in picks.items():
                cm = composition_metrics(sol["weights"])
                rows.append(
                    {
                        "dataset": ds,
                        "k": k,
                        "method": method,
                        "tradeoff": tradeoff,
                        "risk": sol["risk"],
                        "return": sol["return"],
                        "active_count": cm["active_count"],
                        "hhi": cm["hhi"],
                        "effective_n": cm["effective_n"],
                        "top1_share": cm["top1_share"],
                        "top3_share": cm["top3_share"],
                    }
                )

    csv_header = [
        "dataset",
        "k",
        "method",
        "tradeoff",
        "risk",
        "return",
        "active_count",
        "hhi",
        "effective_n",
        "top1_share",
        "top3_share",
    ]
    save_csv(
        os.path.join(args.output_dir, "composition_metrics.csv"),
        rows=rows,
        header=csv_header,
    )

    fig_png = os.path.join(args.output_dir, "fig06_hhi_tradeoff.png")
    fig_pdf = os.path.join(args.output_dir, "fig06_hhi_tradeoff.pdf")
    plot_hhi_tradeoff(rows, fig_png, fig_pdf)

    fig_png2 = os.path.join(args.output_dir, "fig07_effective_n_by_k.png")
    fig_pdf2 = os.path.join(args.output_dir, "fig07_effective_n_by_k.pdf")
    plot_effective_n_by_k(rows, fig_png2, fig_pdf2)

    print("Saved composition metrics and plots in: {}".format(os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main()

