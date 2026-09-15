"""
Chunk 7 -- ML comparison: IsolationForest + XGBoost vs. the rule-based
system (Chunks 5-6).

Scope decision: this chunk models BENEFICIARY-level fraud only. Vendor
collusion has 2 true positives and registrar diversion has 1 -- nowhere
near enough positive examples for train/test ML (XGBoost) or a
meaningful contamination estimate (IsolationForest). Those typologies
stay correctly owned by the relative-threshold rules (R3/R4/R7, R5/R8);
this chunk doesn't pretend ML adds anything there.

Feature design, two tiers:
  (a) "behavioral" -- engineered purely from beneficiaries/vendors/
      registrars/issuances/transactions (never answer_key). Tests
      whether ML from raw data alone can match or beat hand-built rules.
  (b) "behavioral + rules" -- adds each Chunk 5 rule's flag/score as a
      feature, plus a vote-count. This is standard real-world practice
      (rule-engine outputs feeding an ML meta-model), not leakage --
      rule_flags never touched answer_key either. Tests whether ML adds
      value ON TOP OF the rules rather than instead of them.

XGBoost is supervised and trains on answer_key labels (train/test split)
-- that's how real supervised fraud models work, using confirmed past
cases. IsolationForest is unsupervised and never sees labels during
fitting; labels are only used post-hoc to evaluate its anomaly ranking.

Usage:
    python src/model.py --db data/aid_program.db
"""

import argparse
import sqlite3
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import xgboost as xgb

RANDOM_STATE = 42

# Analyst-capacity operating points: how many cases can a review team
# actually work through, independent of how many true positives happen
# to exist. Expressed as counts against the FULL population (49,928).
# Setting K = the true-positive count (an earlier version of this script
# did this) forces precision and recall to be numerically identical by
# construction -- TP/K where K IS the true-positive count means the
# denominator is the same for both, which is a diagnostic convenience for
# comparing models, not an operating point any real review team sits at.
# Real fraud teams talk in terms of "what can we clear this cycle" --
# this reports precision/recall at fixed capacities instead.
CAPACITY_POINTS = [100, 250, 500, 1000, 1500, 2000]


def load_tables(conn):
    beneficiaries = pd.read_sql("SELECT * FROM beneficiaries", conn, parse_dates=["registration_date"])
    vendors = pd.read_sql("SELECT * FROM vendors", conn)
    registrars = pd.read_sql("SELECT * FROM registrars", conn)
    issuances = pd.read_sql("SELECT * FROM issuances", conn, parse_dates=["issue_timestamp"])
    transactions = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["timestamp"])
    rule_flags = pd.read_sql("SELECT * FROM rule_flags", conn)
    answer_key = pd.read_sql("SELECT * FROM answer_key", conn)
    return beneficiaries, vendors, registrars, issuances, transactions, rule_flags, answer_key


# ---------------------------------------------------------------------------
# Feature engineering (tier a: behavioral, no answer_key, no rule flags)
# ---------------------------------------------------------------------------
def build_behavioral_features(beneficiaries, vendors, registrars, issuances, transactions):
    b = beneficiaries.set_index("beneficiary_id").copy()

    # --- attribute-cluster features ---
    b["phone_share_count"] = b.groupby("phone")["family_name"].transform("count") - 1
    b["address_share_count"] = b.groupby("address_id")["family_name"].transform("count") - 1
    same_hh = b.groupby(["address_id", "claimed_household_size"])["family_name"].transform("count") - 1
    b["address_same_hh_size_count"] = same_hh
    b["registrar_day_burst_size"] = b.groupby(
        ["registrar_id", "registration_date"]
    )["family_name"].transform("count")
    registrar_totals = b.groupby("registrar_id")["family_name"].transform("count")
    b["registrar_total_caseload"] = registrar_totals

    # --- transaction-cycle behavioral features ---
    tx = transactions.copy()
    tx["cycle_month"] = tx["timestamp"].dt.strftime("%Y-%m")
    tx["hour"] = tx["timestamp"].dt.hour
    tx["is_night"] = tx["hour"].between(0, 5)

    vendor_district = vendors.set_index("vendor_id")["district"]
    vendor_type = vendors.set_index("vendor_id")["vendor_type"]
    tx["vendor_district"] = tx["vendor_id"].map(vendor_district)
    tx["vendor_type"] = tx["vendor_id"].map(vendor_type)
    tx["ben_district"] = tx["beneficiary_id"].map(b["district"])
    tx["cross_district"] = tx["vendor_district"] != tx["ben_district"]
    tx["is_merchant"] = tx["vendor_type"] == "merchant-wholesale"

    agg = tx.groupby("beneficiary_id").agg(
        n_transactions=("txn_id", "count"),
        n_distinct_vendors=("vendor_id", "nunique"),
        total_amount=("amount", "sum"),
        avg_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        pct_night=("is_night", "mean"),
        pct_cross_district=("cross_district", "mean"),
        pct_merchant_vendor=("is_merchant", "mean"),
    )
    agg["std_amount"] = agg["std_amount"].fillna(0.0)

    # vendor concentration: share of a beneficiary's own transactions at
    # their single most-used vendor (high = always same vendor; ghosts/
    # ring members are highly concentrated by construction)
    top_vendor_share = (
        tx.groupby(["beneficiary_id", "vendor_id"]).size()
        .groupby("beneficiary_id").max() / tx.groupby("beneficiary_id").size()
    )
    agg["top_vendor_share"] = top_vendor_share

    # cycle-level: issuance count, fraction spent, fast-full-cashout rate
    cyc = tx.groupby(["beneficiary_id", "cycle_month"]).agg(
        n_txns=("txn_id", "count"), spent=("amount", "sum"), first_ts=("timestamp", "min")
    ).reset_index()
    cyc = issuances.merge(cyc, on=["beneficiary_id", "cycle_month"], how="left")
    cyc["n_txns"] = cyc["n_txns"].fillna(0)
    cyc["spent"] = cyc["spent"].fillna(0.0)
    cyc["fraction_spent"] = cyc["spent"] / cyc["amount_issued"]
    cyc["hours_to_first"] = (cyc["first_ts"] - cyc["issue_timestamp"]).dt.total_seconds() / 3600
    cyc["is_fast_full"] = (
        (cyc["n_txns"] >= 1) & (cyc["n_txns"] <= 2) &
        (cyc["fraction_spent"] >= 0.90) & (cyc["hours_to_first"] >= 0) & (cyc["hours_to_first"] <= 6)
    )

    cycle_agg = cyc.groupby("beneficiary_id").agg(
        n_issuances=("issuance_id", "count"),
        avg_fraction_spent=("fraction_spent", "mean"),
        pct_cycles_fast_full=("is_fast_full", "mean"),
        pct_cycles_missed=("n_txns", lambda s: (s == 0).mean()),
    )

    features = b[["claimed_household_size", "phone_share_count", "address_share_count",
                  "address_same_hh_size_count", "registrar_day_burst_size", "registrar_total_caseload"]].copy()
    features = features.join(agg, how="left").join(cycle_agg, how="left")
    features = features.fillna(0.0)
    return features


# ---------------------------------------------------------------------------
# Feature engineering (tier b add-on: Chunk 5 rule flags as features)
# ---------------------------------------------------------------------------
def build_rule_features(rule_flags, beneficiary_ids):
    ben_flags = rule_flags[rule_flags["entity_type"] == "beneficiary"]
    rules = sorted(ben_flags["rule_id"].unique())
    idx = pd.Index(beneficiary_ids, name="beneficiary_id")
    out = pd.DataFrame(0.0, index=idx, columns=[f"flag_{r}" for r in rules])
    for r in rules:
        sub = ben_flags[ben_flags["rule_id"] == r].groupby("entity_id")["score"].max()
        out.loc[out.index.intersection(sub.index), f"flag_{r}"] = sub.reindex(out.index.intersection(sub.index)).values
    out["vote_count"] = (out[[f"flag_{r}" for r in rules if r != "R6_naive"]] > 0).sum(axis=1)
    return out


def build_labels(answer_key, beneficiary_ids):
    fraud_ids = set(answer_key[answer_key["entity_type"] == "beneficiary"]["entity_id"])
    return pd.Series([1 if bid in fraud_ids else 0 for bid in beneficiary_ids], index=beneficiary_ids, name="label")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def run_isolation_forest(X, contamination):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(Xs)
    # more negative score_samples = more anomalous; flip sign so higher = more anomalous
    scores = -model.score_samples(Xs)
    return pd.Series(scores, index=X.index, name="if_score")


def run_xgboost(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    test_scores = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index, name="xgb_score")
    return model, X_train, X_test, y_train, y_test, test_scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def capacity_sweep(scores, y_true, population_size, capacity_points=None, scale_to=None):
    """
    Precision/recall at fixed review-capacity operating points, not at
    K=true-positive-count. `capacity_points` are expressed against the
    FULL population (population_size); if this scores series covers a
    smaller slice (e.g. a held-out test set), pass scale_to=len(scores)
    so the same "fraction of caseload" capacity is used consistently --
    e.g. reviewing 500 of 49,928 (1.0%) becomes ~150 of 14,979 test rows.
    """
    capacity_points = capacity_points or CAPACITY_POINTS
    rows = []
    for k_full in capacity_points:
        if scale_to is not None:
            k = max(1, round(k_full * scale_to / population_size))
        else:
            k = k_full
        if k > len(scores):
            continue
        m = metrics_at_topk(scores, y_true, k)
        m["k_full_population_equivalent"] = k_full
        rows.append(m)
    return pd.DataFrame(rows)


def print_capacity_table(df, population_label):
    print(f"  (K against {population_label})")
    cols = ["k_full_population_equivalent", "k", "tp", "fp", "fn", "precision", "recall"]
    header = "  " + "  ".join(f"{c:>14s}" for c in ["capacity", "k_actual", "tp", "fp", "fn", "precision", "recall"])
    print(header)
    for _, row in df.iterrows():
        parts = [f"{int(row['k_full_population_equivalent']):>14d}", f"{int(row['k']):>14d}",
                 f"{int(row['tp']):>14d}", f"{int(row['fp']):>14d}", f"{int(row['fn']):>14d}",
                 f"{row['precision']:>14.3f}" if pd.notna(row['precision']) else f"{'--':>14s}",
                 f"{row['recall']:>14.3f}" if pd.notna(row['recall']) else f"{'--':>14s}"]
        print("  " + "  ".join(parts))


def metrics_at_topk(scores, y_true, k):
    top_ids = scores.sort_values(ascending=False).head(k).index
    flagged = set(top_ids)
    true_ids = set(y_true[y_true == 1].index)
    tp = len(flagged & true_ids)
    fp = len(flagged - true_ids)
    fn = len(true_ids - flagged)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) > 0 else float("nan")
    return {"k": k, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def metrics_at_threshold(scores, y_true, threshold=0.5):
    flagged = set(scores[scores >= threshold].index)
    true_ids = set(y_true[y_true == 1].index)
    tp = len(flagged & true_ids)
    fp = len(flagged - true_ids)
    fn = len(true_ids - flagged)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and (precision + recall) > 0 else float("nan")
    return {"threshold": threshold, "n_flagged": len(flagged), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def identify_hard_duplicates(beneficiaries, answer_key):
    """Duplicates that share NEITHER phone NOR address with anyone (the
    deliberately harder ~30% subset from inject_fraud.py). Retroactive
    identification for ANALYSIS ONLY -- never fed as a feature or label."""
    dup_ids = set(answer_key[answer_key["fraud_type"] == "duplicate"]["entity_id"])
    b = beneficiaries.set_index("beneficiary_id")
    phone_counts = b.groupby("phone").size()
    addr_counts = b.groupby("address_id").size()
    hard = [
        bid for bid in dup_ids
        if bid in b.index and phone_counts.get(b.loc[bid, "phone"], 1) < 2
        and addr_counts.get(b.loc[bid, "address_id"], 1) < 2
    ]
    return set(hard)


def identify_reg07_beneficiaries(beneficiaries, answer_key):
    fraud_ids = set(answer_key[answer_key["entity_type"] == "beneficiary"]["entity_id"])
    reg07_ids = set(beneficiaries[beneficiaries["registrar_id"] == "REG-07"]["beneficiary_id"])
    return fraud_ids & reg07_ids


def recovery_rate(flagged_ids, target_subset):
    if not target_subset:
        return float("nan")
    return len(set(flagged_ids) & target_subset) / len(target_subset)


def print_table(df, cols):
    header = "  " + "  ".join(f"{c:>12s}" for c in cols)
    print(header)
    for _, row in df.iterrows():
        parts = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                parts.append(f"{v:>12.3f}" if pd.notna(v) else f"{'--':>12s}")
            else:
                parts.append(f"{v:>12}")
        print("  " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Chunk 7: IsolationForest + XGBoost comparison")
    parser.add_argument("--db", type=str, default="data/aid_program.db")
    parser.add_argument("--out-csv", type=str, default="data/ml_comparison.csv")
    args = parser.parse_args()

    t0 = time.time()
    conn = sqlite3.connect(args.db)
    beneficiaries, vendors, registrars, issuances, transactions, rule_flags, answer_key = load_tables(conn)
    print(f"Loaded {len(beneficiaries):,} beneficiaries, {len(transactions):,} transactions, "
          f"{len(rule_flags):,} rule_flags, {len(answer_key):,} answer_key rows")

    print("\nEngineering behavioral features (tier a)...")
    feat_behavioral = build_behavioral_features(beneficiaries, vendors, registrars, issuances, transactions)
    print(f"  {feat_behavioral.shape[1]} behavioral features for {feat_behavioral.shape[0]:,} beneficiaries")

    print("Adding rule-flag features (tier b)...")
    feat_rules = build_rule_features(rule_flags, feat_behavioral.index)
    feat_combined = feat_behavioral.join(feat_rules)
    print(f"  {feat_combined.shape[1]} total features with rule flags added")

    labels = build_labels(answer_key, feat_behavioral.index)
    n_fraud = int(labels.sum())
    contamination = n_fraud / len(labels)
    print(f"  {n_fraud:,} true fraud beneficiaries ({100*contamination:.2f}% prevalence)")

    hard_dupes = identify_hard_duplicates(beneficiaries, answer_key)
    reg07_fraud = identify_reg07_beneficiaries(beneficiaries, answer_key)
    print(f"  Hard-duplicate subset (no shared phone/address): {len(hard_dupes):,}")
    print(f"  REG-07 fraud beneficiaries (any typology): {len(reg07_fraud):,}")

    population_size = len(labels)
    illustrative_k = 500  # for the hard-duplicate/REG-07 recovery spot-check below

    # rule ensemble: rank by vote count (continuous score), not a fixed
    # >=2 threshold, so it can be swept at the same capacity points as
    # everything else on equal footing
    ben_flags = rule_flags[(rule_flags["entity_type"] == "beneficiary") & (rule_flags["rule_id"] != "R6_naive")]
    votes = ben_flags.groupby("entity_id")["rule_id"].nunique()
    rule_scores = votes.reindex(feat_behavioral.index).fillna(0.0)

    results = []

    # --- Rule ensemble (vote-count ranking), full population ---
    print("\n=== Rule ensemble (vote-count ranking) -- analyst-capacity sweep ===")
    rule_ap = average_precision_score(labels, rule_scores)
    print(f"  average precision (PR-AUC): {rule_ap:.3f}")
    rule_sweep = capacity_sweep(rule_scores, labels, population_size)
    print_capacity_table(rule_sweep, "49,928 beneficiaries")
    rule_sweep["model"] = "RuleEnsemble"; rule_sweep["features"] = "vote_count"; rule_sweep["pr_auc"] = rule_ap
    results.append(rule_sweep)
    top500_rules = set(rule_scores.sort_values(ascending=False).head(illustrative_k).index)
    print(f"  at capacity 500: hard-duplicate recovery {recovery_rate(top500_rules, hard_dupes):.3f}, "
          f"REG-07-fraud recovery {recovery_rate(top500_rules, reg07_fraud):.3f}")

    # --- IsolationForest, both feature tiers, full population ---
    for tier_name, X in [("behavioral", feat_behavioral), ("behavioral+rules", feat_combined)]:
        print(f"\nFitting IsolationForest ({tier_name})...")
        if_scores = run_isolation_forest(X, contamination)
        ap = average_precision_score(labels, if_scores)
        print(f"  average precision (PR-AUC): {ap:.3f}")
        sweep = capacity_sweep(if_scores, labels, population_size)
        print_capacity_table(sweep, "49,928 beneficiaries")
        sweep["model"] = "IsolationForest"; sweep["features"] = tier_name; sweep["pr_auc"] = ap
        results.append(sweep)
        top500 = set(if_scores.sort_values(ascending=False).head(illustrative_k).index)
        print(f"  at capacity 500: hard-duplicate recovery {recovery_rate(top500, hard_dupes):.3f}, "
              f"REG-07-fraud recovery {recovery_rate(top500, reg07_fraud):.3f}")

    # --- XGBoost, both feature tiers, held-out test set only ---
    # Capacity points are scaled to the test set's ~30% share of the
    # population so "capacity 500" means the same fraction of caseload
    # here as it does for the full-population methods above -- these
    # numbers are honest generalization estimates (held-out data only),
    # not deployment-scale counts.
    for tier_name, X in [("behavioral", feat_behavioral), ("behavioral+rules", feat_combined)]:
        print(f"\nFitting XGBoost ({tier_name})...")
        model, X_train, X_test, y_train, y_test, test_scores = run_xgboost(X, labels)
        ap = average_precision_score(y_test, test_scores)
        m_thresh = metrics_at_threshold(test_scores, y_test, 0.5)
        print(f"  held-out test set ({len(X_test):,} rows, {int(y_test.sum())} true fraud): "
              f"average precision (PR-AUC) {ap:.3f}")
        print(f"  at probability threshold 0.5: precision {m_thresh['precision']:.3f}, "
              f"recall {m_thresh['recall']:.3f}, {m_thresh['n_flagged']:,} flagged")
        sweep = capacity_sweep(test_scores, y_test, population_size, scale_to=len(X_test))
        print_capacity_table(sweep, f"49,928 beneficiaries, scaled to {len(X_test):,}-row test set")
        sweep["model"] = "XGBoost"; sweep["features"] = tier_name; sweep["pr_auc"] = ap
        results.append(sweep)

        top500_scaled_k = max(1, round(500 * len(X_test) / population_size))
        top_test_ids = set(test_scores.sort_values(ascending=False).head(top500_scaled_k).index)
        print(f"  at capacity 500 (~{top500_scaled_k} in test set): "
              f"hard-duplicate recovery {recovery_rate(top_test_ids, hard_dupes & set(X_test.index)):.3f}, "
              f"REG-07-fraud recovery {recovery_rate(top_test_ids, reg07_fraud & set(X_test.index)):.3f}")

        if tier_name == "behavioral":
            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            print("  top 8 feature importances (behavioral-only model):")
            for feat, imp in importances.head(8).items():
                print(f"    {feat:30s} {imp:.3f}")

    print("\n=== Summary: PR-AUC by method (threshold-independent ranking quality) ===")
    summary_rows = []
    for df in results:
        summary_rows.append({"model": df["model"].iloc[0], "features": df["features"].iloc[0],
                              "pr_auc": df["pr_auc"].iloc[0]})
    summary_df = pd.DataFrame(summary_rows).drop_duplicates()
    print_table(summary_df, ["model", "features", "pr_auc"])

    print("\n=== Summary: precision @ capacity=500 across all methods (comparable operating point) ===")
    at_500 = []
    for df in results:
        row = df[df["k_full_population_equivalent"] == 500]
        if not row.empty:
            r = row.iloc[0]
            at_500.append({"model": r["model"], "features": r["features"],
                           "precision": r["precision"], "recall": r["recall"], "tp": r["tp"]})
    print_table(pd.DataFrame(at_500), ["model", "features", "tp", "precision", "recall"])

    full_results_df = pd.concat(results, ignore_index=True)
    full_results_df.to_csv(args.out_csv, index=False)
    print(f"\n(full capacity-sweep table written to {args.out_csv})")

    conn.close()
    print(f"\nDone in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
