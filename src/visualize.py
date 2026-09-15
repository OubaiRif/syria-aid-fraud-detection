"""
Findings-walkthrough visualizations for the README/portfolio.

Reads from the same non-ground-truth outputs the rest of the pipeline
produces (data/rule_evaluation.csv, data/ml_comparison.csv, and the DB
directly for the vendor night-share chart) -- no new analysis happens
here, this just visualizes results Chunks 6/7 already computed.

Usage:
    python src/visualize.py --db data/aid_program.db --outdir notebooks/figures
"""

import argparse
import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd

METHOD_COLORS = {
    "RuleEnsemble": "#6b7280",
    "IsolationForest_behavioral": "#60a5fa",
    "IsolationForest_behavioral+rules": "#2563eb",
    "XGBoost_behavioral": "#fca5a5",
    "XGBoost_behavioral+rules": "#dc2626",
}


def fig1_capacity_sweep(ml_df, outdir):
    """Finding 4: precision/recall vs review capacity, all methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for (model, features), group in ml_df.groupby(["model", "features"]):
        key = f"{model}_{features}" if model != "RuleEnsemble" else "RuleEnsemble"
        label = f"{model} ({features})" if model != "RuleEnsemble" else "Rule ensemble"
        color = METHOD_COLORS.get(key, "#999999")
        g = group.sort_values("k_full_population_equivalent")
        axes[0].plot(g["k_full_population_equivalent"], g["precision"], marker="o", label=label, color=color)
        axes[1].plot(g["k_full_population_equivalent"], g["recall"], marker="o", label=label, color=color)

    axes[0].set_title("Precision vs. review capacity")
    axes[1].set_title("Recall vs. review capacity")
    for ax in axes:
        ax.set_xlabel("Review capacity (cases)")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Precision")
    axes[1].set_ylabel("Recall")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[0].axvline(500, color="black", linestyle=":", alpha=0.5)
    axes[1].axvline(500, color="black", linestyle=":", alpha=0.5)
    axes[1].text(520, 0.05, "capacity=500\n(methods converge)", fontsize=8, alpha=0.7)

    fig.suptitle("Finding 4: model quality only pays off once review capacity exists", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_capacity_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig2_pr_auc(ml_df, outdir):
    summary = ml_df[["model", "features", "pr_auc"]].drop_duplicates().reset_index(drop=True)
    summary["label"] = summary.apply(
        lambda r: "Rule ensemble" if r["model"] == "RuleEnsemble" else f"{r['model']}\n({r['features']})", axis=1
    )
    summary = summary.sort_values("pr_auc")
    colors = [METHOD_COLORS.get(f"{r.model}_{r.features}" if r.model != "RuleEnsemble" else "RuleEnsemble", "#999999")
              for r in summary.itertuples()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(summary["label"], summary["pr_auc"], color=colors)
    for bar, val in zip(bars, summary["pr_auc"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("PR-AUC (average precision)")
    ax.set_title("Ranking quality by method\n(rules alone rank weakly; rules-as-features consistently improve both models)")
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_pr_auc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig3_rule_landscape(rule_df, outdir):
    sub = rule_df[rule_df["vs"] == "target_typology"].copy()
    sub = sub[sub["entity_type"] == "beneficiary"]  # keep the plot to comparable per-record rules

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in sub.iterrows():
        ax.scatter(row["precision"], row["recall"], s=120)
        ax.annotate(row["rule_id"], (row["precision"], row["recall"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=10)
    ax.set_xlabel("Precision (vs. target typology)")
    ax.set_ylabel("Recall (vs. target typology)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.5)
    ax.axvline(0.5, color="gray", linewidth=0.5)
    ax.grid(alpha=0.3)
    ax.set_title("Beneficiary-level rule precision/recall landscape\n(against each rule's own target typology)")
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_rule_landscape.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig4_typology_coverage(conn, outdir):
    cur = conn.cursor()
    cur.execute("SELECT rule_id, entity_type, entity_id FROM rule_flags WHERE rule_id != 'R6_naive'")
    flags = pd.DataFrame(cur.fetchall(), columns=["rule_id", "entity_type", "entity_id"])
    cur.execute("SELECT entity_type, fraud_type, entity_id FROM answer_key")
    ak = pd.DataFrame(cur.fetchall(), columns=["entity_type", "fraud_type", "entity_id"])

    rule_targets = {
        "R1": ["ghost", "duplicate"], "R2": ["duplicate"], "R3": ["vendor_collusion"],
        "R4": ["vendor_collusion"], "R5": ["ghost", "registrar_diversion"], "R6": ["cashout_ring"],
        "R7": ["vendor_collusion"], "R8": ["registrar_diversion"], "R9": ["family_gaming"],
        "R10": ["family_gaming"],
    }

    rows = []
    for entity_type in ak["entity_type"].unique():
        for fraud_type in ak[ak["entity_type"] == entity_type]["fraud_type"].unique():
            true_ids = set(ak[(ak["entity_type"] == entity_type) & (ak["fraud_type"] == fraud_type)]["entity_id"])
            targeting = [r for r, t in rule_targets.items() if fraud_type in t]
            union_flagged = set(flags[(flags["rule_id"].isin(targeting)) & (flags["entity_type"] == entity_type)]["entity_id"])
            tp = len(union_flagged & true_ids)
            precision = tp / len(union_flagged) if union_flagged else float("nan")
            recall = tp / len(true_ids) if true_ids else float("nan")
            label = f"{fraud_type}\n({entity_type})"
            rows.append({"label": label, "precision": precision, "recall": recall})

    df = pd.DataFrame(rows).sort_values("recall")
    fig, ax = plt.subplots(figsize=(9, 6))
    y = range(len(df))
    ax.barh([i + 0.2 for i in y], df["recall"], height=0.4, label="Recall", color="#2563eb")
    ax.barh([i - 0.2 for i in y], df["precision"], height=0.4, label="Precision", color="#93c5fd")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Rate")
    ax.set_title("Typology coverage: union of all rules targeting each fraud type")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    path = os.path.join(outdir, "fig4_typology_coverage.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig5_vendor_night_share(conn, outdir):
    cur = conn.cursor()
    cur.execute("SELECT entity_id FROM answer_key WHERE fraud_type='vendor_collusion'")
    colluding = [r[0] for r in cur.fetchall()]
    ghost_vendor, standalone_vendor = colluding[0], colluding[1]

    def night_share(vendor_id):
        cur.execute("SELECT COUNT(*) FROM transactions WHERE vendor_id=?", (vendor_id,))
        total = cur.fetchone()[0]
        cur.execute("""SELECT COUNT(*) FROM transactions
                       WHERE vendor_id=? AND CAST(strftime('%H',timestamp) AS INT) BETWEEN 0 AND 5""", (vendor_id,))
        night = cur.fetchone()[0]
        return 100 * night / total if total else 0.0

    cur.execute("SELECT vendor_id FROM vendors WHERE vendor_id NOT IN (?,?)", (ghost_vendor, standalone_vendor))
    others = [r[0] for r in cur.fetchall()]
    other_shares = [night_share(v) for v in others]
    baseline = sum(other_shares) / len(other_shares)

    labels = [f"Quiet vendor\n({ghost_vendor})", f"Loud vendor\n({standalone_vendor})", "Honest baseline\n(avg of 118)"]
    values = [night_share(ghost_vendor), night_share(standalone_vendor), baseline]
    colors = ["#93c5fd", "#dc2626", "#6b7280"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}%", ha="center", fontsize=10)
    ax.set_ylabel("Night-hour (00:00-05:00) transaction share")
    ax.set_title("Vendor collusion: the loud/quiet asymmetry (DI-006)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(outdir, "fig5_vendor_night_share.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate findings-walkthrough visualizations")
    parser.add_argument("--db", type=str, default="data/aid_program.db")
    parser.add_argument("--outdir", type=str, default="notebooks/figures")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    conn = sqlite3.connect(args.db)
    rule_df = pd.read_csv("data/rule_evaluation.csv")
    ml_df = pd.read_csv("data/ml_comparison.csv")

    paths = [
        fig1_capacity_sweep(ml_df, args.outdir),
        fig2_pr_auc(ml_df, args.outdir),
        fig3_rule_landscape(rule_df, args.outdir),
        fig4_typology_coverage(conn, args.outdir),
        fig5_vendor_night_share(conn, args.outdir),
    ]
    conn.close()

    print(f"Wrote {len(paths)} figures to {args.outdir}/:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
