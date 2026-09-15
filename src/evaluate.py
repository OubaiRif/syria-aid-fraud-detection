"""
Chunk 6 -- Evaluation.

Joins rule_flags (Chunk 5's output) against answer_key (Chunk 4's ground
truth) to compute precision/recall/F1 per rule -- both against the
specific typology(ies) each rule targets and against all fraud combined
-- plus typology coverage (does ANY targeting rule catch this entity?)
and a combined ensemble vote sweep. This is the one script in the
pipeline allowed to read answer_key; that's its job.

Usage:
    python src/evaluate.py --db data/aid_program.db
"""

import argparse
import sqlite3
import time

import pandas as pd

# Which typology(ies) each rule targets, per fraud_typology_spec.md's R1-R10
# preview table (R6_naive shares R6's target -- it's the deliberately weak
# baseline for comparison, not a separate typology).
RULE_TARGETS = {
    "R1": ["ghost", "duplicate"],
    "R2": ["duplicate"],
    "R3": ["vendor_collusion"],
    "R4": ["vendor_collusion"],
    "R5": ["ghost", "registrar_diversion"],
    "R6_naive": ["cashout_ring"],
    "R6": ["cashout_ring"],
    "R7": ["vendor_collusion"],
    "R8": ["registrar_diversion"],
    "R9": ["family_gaming"],
    "R10": ["family_gaming"],
}

# The "real" (non-naive) beneficiary-level rules that make up the combined
# detection system, for ensemble voting. R6_naive is deliberately excluded
# -- it exists to demonstrate a failure mode, not to vote in the ensemble.
ENSEMBLE_BENEFICIARY_RULES = ["R1", "R2", "R5", "R6", "R9", "R10"]
ENSEMBLE_VENDOR_RULES = ["R3", "R4", "R7"]
ENSEMBLE_REGISTRAR_RULES = ["R5", "R8"]


def load_tables(conn):
    rule_flags = pd.read_sql("SELECT * FROM rule_flags", conn)
    answer_key = pd.read_sql("SELECT * FROM answer_key", conn)
    return rule_flags, answer_key


def compute_metrics(flagged_ids, true_ids):
    flagged = set(flagged_ids)
    true = set(true_ids)
    tp = len(flagged & true)
    fp = len(flagged - true)
    fn = len(true - flagged)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def per_rule_metrics(rule_flags, answer_key):
    """For every (rule_id, entity_type) pair actually present in rule_flags,
    compute metrics against (a) the rule's own target typology(ies) and
    (b) all fraud of that entity_type combined."""
    rows = []
    combos = rule_flags[["rule_id", "entity_type"]].drop_duplicates()
    for _, (rule_id, entity_type) in combos.iterrows():
        flagged_ids = rule_flags[
            (rule_flags["rule_id"] == rule_id) & (rule_flags["entity_type"] == entity_type)
        ]["entity_id"].unique()

        ak_this_type = answer_key[answer_key["entity_type"] == entity_type]
        targets = RULE_TARGETS.get(rule_id, [])
        true_target = ak_this_type[ak_this_type["fraud_type"].isin(targets)]["entity_id"].unique()
        true_all = ak_this_type["entity_id"].unique()

        m_target = compute_metrics(flagged_ids, true_target)
        m_all = compute_metrics(flagged_ids, true_all)

        rows.append({"rule_id": rule_id, "entity_type": entity_type,
                      "target_typologies": ",".join(targets), "n_flagged": len(flagged_ids),
                      "vs": "target_typology", **m_target})
        rows.append({"rule_id": rule_id, "entity_type": entity_type,
                      "target_typologies": ",".join(targets), "n_flagged": len(flagged_ids),
                      "vs": "all_fraud", **m_all})
    return pd.DataFrame(rows)


def typology_coverage(rule_flags, answer_key):
    """For each fraud_type, which rules target it, and what recall does
    the UNION of those rules' flags achieve (does ANY targeting rule
    catch this entity)? This is the 'does the detection SYSTEM catch this
    typology' view, as opposed to any single rule's own recall."""
    rows = []
    for entity_type in answer_key["entity_type"].unique():
        for fraud_type in answer_key[answer_key["entity_type"] == entity_type]["fraud_type"].unique():
            true_ids = set(answer_key[
                (answer_key["entity_type"] == entity_type) & (answer_key["fraud_type"] == fraud_type)
            ]["entity_id"])
            targeting_rules = [r for r, t in RULE_TARGETS.items() if fraud_type in t and r != "R6_naive"]
            union_flagged = set(rule_flags[
                (rule_flags["rule_id"].isin(targeting_rules)) & (rule_flags["entity_type"] == entity_type)
            ]["entity_id"])
            m = compute_metrics(union_flagged, true_ids)
            rows.append({
                "fraud_type": fraud_type, "entity_type": entity_type,
                "targeting_rules": ",".join(targeting_rules), "n_true": len(true_ids),
                "union_recall": m["recall"], "union_precision": m["precision"],
            })
    return pd.DataFrame(rows)


def ensemble_sweep(rule_flags, answer_key, rules, entity_type):
    """Vote-count ensemble: flag an entity if >=K of `rules` flagged it,
    for K=1..len(rules). Shows the precision/recall tradeoff of requiring
    corroboration across independent rules vs. trusting any single one."""
    sub = rule_flags[(rule_flags["rule_id"].isin(rules)) & (rule_flags["entity_type"] == entity_type)]
    votes = sub.groupby("entity_id")["rule_id"].nunique()
    true_ids = set(answer_key[answer_key["entity_type"] == entity_type]["entity_id"])

    rows = []
    for k in range(1, len(rules) + 1):
        flagged_ids = votes[votes >= k].index
        m = compute_metrics(flagged_ids, true_ids)
        rows.append({"min_votes": k, "n_flagged": len(flagged_ids), **m})
    return pd.DataFrame(rows)


def print_table(df, cols, formats=None):
    formats = formats or {}
    header = "  " + "  ".join(f"{c:>12s}" if c not in ("rule_id", "fraud_type", "entity_type",
                                                          "target_typologies", "targeting_rules", "vs")
                               else f"{c:<20s}" for c in cols)
    print(header)
    for _, row in df.iterrows():
        parts = []
        for c in cols:
            v = row[c]
            if c in formats:
                parts.append(formats[c](v))
            elif isinstance(v, float):
                parts.append(f"{v:>12.3f}" if pd.notna(v) else f"{'--':>12s}")
            elif c in ("rule_id", "fraud_type", "entity_type", "target_typologies", "targeting_rules", "vs"):
                parts.append(f"{str(v):<20s}")
            else:
                parts.append(f"{v:>12}")
        print("  " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Chunk 6: evaluate rules against answer_key")
    parser.add_argument("--db", type=str, default="data/aid_program.db")
    parser.add_argument("--out-csv", type=str, default="data/rule_evaluation.csv")
    args = parser.parse_args()

    t0 = time.time()
    conn = sqlite3.connect(args.db)
    rule_flags, answer_key = load_tables(conn)
    print(f"Loaded {len(rule_flags):,} rule_flags rows, {len(answer_key):,} answer_key rows")

    print("\n=== Per-rule metrics ===")
    metrics_df = per_rule_metrics(rule_flags, answer_key)
    metrics_df.to_csv(args.out_csv, index=False)
    print(f"(full table written to {args.out_csv})\n")
    for vs_label in ["target_typology", "all_fraud"]:
        print(f"--- vs {vs_label} ---")
        sub = metrics_df[metrics_df["vs"] == vs_label].sort_values(["entity_type", "rule_id"])
        print_table(sub, ["rule_id", "entity_type", "n_flagged", "tp", "fp", "fn", "precision", "recall", "f1"])
        print()

    print("=== Typology coverage (union of all rules targeting each typology) ===")
    coverage_df = typology_coverage(rule_flags, answer_key)
    print_table(coverage_df.sort_values("entity_type"),
                ["fraud_type", "entity_type", "n_true", "targeting_rules", "union_precision", "union_recall"])

    print("\n=== Ensemble vote sweep: beneficiary-level rules "
          f"({', '.join(ENSEMBLE_BENEFICIARY_RULES)}) ===")
    ben_sweep = ensemble_sweep(rule_flags, answer_key, ENSEMBLE_BENEFICIARY_RULES, "beneficiary")
    print_table(ben_sweep, ["min_votes", "n_flagged", "tp", "fp", "fn", "precision", "recall", "f1"])

    print(f"\n=== Ensemble vote sweep: vendor-level rules ({', '.join(ENSEMBLE_VENDOR_RULES)}) ===")
    vendor_sweep = ensemble_sweep(rule_flags, answer_key, ENSEMBLE_VENDOR_RULES, "vendor")
    print_table(vendor_sweep, ["min_votes", "n_flagged", "tp", "fp", "fn", "precision", "recall", "f1"])

    print(f"\n=== Ensemble vote sweep: registrar-level rules ({', '.join(ENSEMBLE_REGISTRAR_RULES)}) ===")
    registrar_sweep = ensemble_sweep(rule_flags, answer_key, ENSEMBLE_REGISTRAR_RULES, "registrar")
    print_table(registrar_sweep, ["min_votes", "n_flagged", "tp", "fp", "fn", "precision", "recall", "f1"])

    conn.close()
    print(f"\nDone in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
