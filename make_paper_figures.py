import argparse
import csv
import math
import os

import matplotlib.pyplot as plt


COLORS = {
    "weighted_sum": "#1f77b4",
    "epsilon_constraint": "#ff7f0e",
    "nsga2": "#2ca02c",
    "combined_pareto": "#d62728",
}


def read_summary(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out = dict(r)
            out["n_assets"] = int(r["n_assets"])
            out["k"] = int(r["k"])
            out["pareto_points"] = int(r["pareto_points"]) if r["pareto_points"] else 0
            out["min_risk"] = float(r["min_risk"]) if r["min_risk"] else None
            out["max_return"] = float(r["max_return"]) if r["max_return"] else None
            out["elapsed_sec"] = float(r["elapsed_sec"]) if r["elapsed_sec"] else 0.0
            rows.append(out)
    return rows


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_png_pdf(fig, out_dir, stem, dpi=260):
    png = os.path.join(out_dir, stem + ".png")
    pdf = os.path.join(out_dir, stem + ".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_grid(n_items, max_cols=3, base_w=13.8, base_h=7.6):
    cols = min(max_cols, max(1, n_items))
    rows = int(math.ceil(float(max(1, n_items)) / float(cols)))
    fig_w = base_w if cols == 3 else max(6.0, 4.6 * cols)
    fig_h = base_h if rows == 2 else max(3.8, 3.6 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [list(axes)]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    return fig, axes, rows, cols


def plot_metric_grid(rows, metric_key, metric_label, out_dir, stem):
    datasets = sorted(set(r["dataset"] for r in rows))
    fig, axes, n_rows, n_cols = build_grid(len(datasets), max_cols=3, base_w=13.8, base_h=7.6)
    method_order = ["epsilon_constraint", "nsga2", "weighted_sum", "combined_pareto"]
    style = {
        "weighted_sum": dict(ls="-", lw=2.1, marker="o", zorder=3),
        "epsilon_constraint": dict(ls="-", lw=1.8, marker="o", zorder=2),
        "nsga2": dict(ls="-", lw=1.8, marker="o", zorder=2),
        "combined_pareto": dict(ls="--", lw=1.5, marker="s", zorder=4, mfc="none"),
    }

    for idx, ds in enumerate(datasets):
        ax = axes[idx // n_cols][idx % n_cols]
        ds_rows = [r for r in rows if r["dataset"] == ds]
        for m in method_order:
            pts = [r for r in ds_rows if r["method"] == m and r[metric_key] is not None]
            if not pts:
                continue
            pts.sort(key=lambda x: x["k"])
            x = [p["k"] for p in pts]
            y = [p[metric_key] for p in pts]
            st = style[m]
            ax.plot(
                x,
                y,
                color=COLORS[m],
                linestyle=st.get("ls", "-"),
                linewidth=st.get("lw", 1.8),
                marker=st.get("marker", "o"),
                markersize=5,
                markerfacecolor=st.get("mfc", None),
                markeredgewidth=1.0,
                label=m,
                zorder=st.get("zorder", 3),
            )
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("K")
        ax.set_ylabel(metric_label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    for idx in range(len(datasets), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.tight_layout()
    save_png_pdf(fig, out_dir, stem)


def plot_runtime(rows, out_dir, stem="fig03_runtime_by_method"):
    methods = ["weighted_sum", "epsilon_constraint", "nsga2"]
    datasets = sorted(set(r["dataset"] for r in rows))
    fig, axes = plt.subplots(1, len(datasets), figsize=(14, 3.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for i, ds in enumerate(datasets):
        ax = axes[i]
        ds_rows = [r for r in rows if r["dataset"] == ds]
        avg = []
        for m in methods:
            ms = [r["elapsed_sec"] for r in ds_rows if r["method"] == m]
            avg.append(sum(ms) / len(ms) if ms else 0.0)
        ax.bar(methods, avg, color=[COLORS[m] for m in methods], alpha=0.9)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Method")
        if i == 0:
            ax.set_ylabel("Average Runtime (sec)")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_png_pdf(fig, out_dir, stem)


def read_pareto_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "risk": float(r["risk"]),
                    "return": float(r["return"]),
                    "method": r.get("method", ""),
                }
            )
    return rows


def unique_dataset_k_pairs(rows):
    return sorted(set((r["dataset"], r["k"]) for r in rows), key=lambda x: (x[0], x[1]))


def plot_pareto_single(batch_dir, dataset, k, out_dir, stem):
    run_dir = os.path.join(batch_dir, "{}_k{}".format(dataset, k))
    csv_path = os.path.join(run_dir, "all_pareto.csv")
    if not os.path.exists(csv_path):
        return False

    pts = read_pareto_csv(csv_path)
    if not pts:
        return False

    methods = sorted(set(p["method"] for p in pts))
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    for m in methods:
        sub = [p for p in pts if p["method"] == m]
        ax.scatter(
            [p["return"] for p in sub],
            [p["risk"] for p in sub],
            s=18,
            alpha=0.80,
            color=COLORS.get(m, None),
            label=m,
        )
    ax.set_title("{}  (K={})".format(dataset, k), fontsize=11)
    ax.set_xlabel("Profit (Return)")
    ax.set_ylabel("Risk")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_png_pdf(fig, out_dir, stem)
    return True



def main():
    parser = argparse.ArgumentParser(description="Create publication-ready figure package.")
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="results_orlib_batch_final",
        help="Directory with summary.csv and per-run folders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_orlib_batch_final/paper_figures",
        help="Output figure directory",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    summary_csv = os.path.join(args.batch_dir, "summary.csv")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError("Missing summary CSV: {}".format(summary_csv))
    rows = read_summary(summary_csv)

    plot_metric_grid(
        rows=rows,
        metric_key="min_risk",
        metric_label="Minimum Risk",
        out_dir=args.output_dir,
        stem="fig01_min_risk_vs_k",
    )
    plot_metric_grid(
        rows=rows,
        metric_key="max_return",
        metric_label="Maximum Return",
        out_dir=args.output_dir,
        stem="fig02_max_return_vs_k",
    )
    plot_runtime(rows=rows, out_dir=args.output_dir, stem="fig03_runtime_by_method")

    saved = 0
    for dataset, k in unique_dataset_k_pairs(rows):
        stem = "pareto_{}_k{}".format(dataset, k)
        if plot_pareto_single(
            batch_dir=args.batch_dir,
            dataset=dataset,
            k=k,
            out_dir=args.output_dir,
            stem=stem,
        ):
            saved += 1
    print("Saved single Pareto figures: {}".format(saved))
    print("Paper figures saved to: {}".format(os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main()
