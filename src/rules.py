"""
Chunk 5 -- Detection rules R1-R10.

Implements the rule candidates from fraud_typology_spec.md's preview
table, reading ONLY from beneficiaries/vendors/registrars/issuances/
transactions. answer_key is never touched here -- that's Chunk 6's job
(evaluate.py), which joins these rules' output against ground truth to
compute precision/recall. Keeping that boundary real (not just notional)
means these rules are calibrated the way a real analyst would have to:
against the population's own statistics, not against a peek at the
answer.

Where a rule's real measured behavior on this dataset diverges from what
the schema/typology specs or the Chunk 3-4 adjudication predicted, that's
noted in comments and printed at runtime rather than papered over --
several of Chunk 4's own corrections (DI-004 to DI-006) exist precisely
because a first attempt didn't hold up against real numbers, and the same
discipline applies here.

Usage:
    python src/rules.py --db data/aid_program.db
"""

import argparse
import sqlite3
import statistics
import time
from difflib import SequenceMatcher
from itertools import combinations

import numpy as np
import pandas as pd

from names_data import normalize_name, find_canonical, FIRST_NAMES, FAMILY_NAMES

# ---------------------------------------------------------------------------
# Tunable thresholds. All documented inline; Chunk 6 evaluation is expected
# to reveal which of these need adjusting -- that's the point of separating
# "implement the rule" (this file) from "measure how well it worked" (Chunk 6).
# ---------------------------------------------------------------------------
R1_MIN_SHARED_PHONE = 3            # per spec: phone on >=3 registrations

R2_FAMILY_NAME_THRESHOLD = 0.50    # fallback fuzzy family-name similarity
R2_ATTR_FIRST_NAME_FLOOR = 0.30    # fallback fuzzy first-name similarity

R3_SIGMA = 3.0                     # night-share > population mean + R3_SIGMA*std
R4_SIGMA = 3.0                     # volume > population mean + R4_SIGMA*std

R5_SIGMA = 3.0                     # registrar-day registrations > mean + R5_SIGMA*std

R6_MAX_TXNS_FOR_FAST = 2           # "fast full cash-out": <=2 txns that cycle
R6_MIN_FRACTION_SPENT = 0.90       # >=90% of issuance redeemed
R6_MAX_HOURS_TO_FIRST = 6          # first txn within 6h of issuance
R6_COORDINATION_SIGMA = 6.0        # bucket count > vendor's OWN mean+k*std for its
                                    # own date/hour buckets. NOT a fixed count: the
                                    # fixed distribution day (Chunk 3 DI, day 5 every
                                    # cycle) means busy honest vendors organically see
                                    # 100+ distinct beneficiaries in a single hour --
                                    # a global threshold like ">=15 others" is trivially
                                    # satisfied almost everywhere and catches nothing
                                    # useful (verified: 41,513 vendor/hour buckets
                                    # exceed 15 honestly). What actually distinguishes
                                    # ring coordination is a hour that's anomalous
                                    # relative to THAT vendor's own baseline, not
                                    # anomalous in absolute terms.

R7_SIGMA = 3.0                     # vendor cross-district-txn share > mean + R7_SIGMA*std

R8_SIGMA = 2.0                     # registrar flag-rate > mean + R8_SIGMA*std (lower --
                                    # this is a meta-rule aggregating already-scarce flags)

R9_MIN_GROUP_SIZE = 3              # per spec: >=3 registrations, identical claimed size
R9_MAX_PLAUSIBLE_RESIDENTS = 15    # sum of claims beyond this is not a real household

R10_BULK_SHARE_THRESHOLD = 0.5     # majority of an address cluster's redemptions in
                                    # one vendor/day = bulk same-day pattern


def load_tables(conn):
    beneficiaries = pd.read_sql(
        "SELECT * FROM beneficiaries", conn, parse_dates=["registration_date"]
    )
    vendors = pd.read_sql("SELECT * FROM vendors", conn)
    registrars = pd.read_sql("SELECT * FROM registrars", conn)
    issuances = pd.read_sql(
        "SELECT * FROM issuances", conn, parse_dates=["issue_timestamp"]
    )
    transactions = pd.read_sql(
        "SELECT * FROM transactions", conn, parse_dates=["timestamp"]
    )
    return beneficiaries, vendors, registrars, issuances, transactions


def sigma_threshold(values, k):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    std = values.std()  # population std, matches the adjudication's "X sigma above mean"
    return mean, std, mean + k * std


# ---------------------------------------------------------------------------
# R1 -- Phone number on >=3 registrations (targets T1, T2)
# ---------------------------------------------------------------------------
def rule_r1(beneficiaries):
    counts = beneficiaries.groupby("phone")["beneficiary_id"].transform("count")
    flagged = beneficiaries.loc[counts >= R1_MIN_SHARED_PHONE, ["beneficiary_id", "phone"]].copy()
    flagged["rule_id"] = "R1"
    flagged["entity_type"] = "beneficiary"
    flagged["score"] = counts[counts >= R1_MIN_SHARED_PHONE].values
    flagged["detail"] = flagged["phone"].apply(lambda p: f"phone {p} shared by {int((beneficiaries['phone']==p).sum())} registrations")
    return flagged.rename(columns={"beneficiary_id": "entity_id"})[
        ["rule_id", "entity_type", "entity_id", "score", "detail"]
    ]


# ---------------------------------------------------------------------------
# R2 -- Name similarity + shared attribute (targets T2)
# ---------------------------------------------------------------------------
def rule_r2(beneficiaries):
    """
    Requires BOTH: a shared attribute (phone or address) AND a name match
    (canonical-dictionary match preferred, fuzzy first+family similarity
    as fallback for names outside the dictionary).

    This is a hard AND, matching the spec's literal wording ("name
    similarity >= threshold + shared attribute") -- and it has to be,
    because canonical name-matching ALONE is far too weak a signal in
    this dataset: with only 45 first-name and 60 family-name canonicals
    shared by the ENTIRE population (honest included, since Chunk 3 draws
    from the same dictionary), any given canonical pair is shared by
    ~18.5 people on average purely by chance (49,928 / (45*60)). An
    earlier version of this rule treated canonical-match + matching
    household size as sufficient and flagged 81% of all beneficiaries at
    full scale -- that's not a detection rule, that's just describing the
    dictionary's size.

    Consequence for recall: the ~30% of Typology 2 duplicates that share
    NEITHER phone nor address (the deliberately "harder" case per
    fraud_typology_spec.md) are NOT reliably catchable by this rule. The
    spec itself frames that segment as "detectable only by name
    similarity + behavior" -- given this dataset's name-space size, name
    similarity alone doesn't get there; it likely needs the behavioral
    cross-referencing Chunk 7's ML models are positioned for, not a
    single attribute rule. Documenting that gap here rather than
    papering over it with a rule that can't actually deliver it.
    """
    df = beneficiaries[["beneficiary_id", "full_name", "family_name", "phone", "address_id"]].copy()
    df["first_name"] = df["full_name"].str.split().str[0]
    df["first_canonical"] = df["first_name"].apply(lambda n: find_canonical(n, FIRST_NAMES))
    df["family_canonical"] = df["family_name"].apply(lambda n: find_canonical(n, FAMILY_NAMES))
    df["norm_first"] = df["first_name"].apply(normalize_name)
    df["norm_family"] = df["family_name"].apply(normalize_name)

    idx = df.set_index("beneficiary_id")
    hits = {}

    for attr in ("phone", "address_id"):
        groups = df.groupby(attr)["beneficiary_id"].apply(list)
        for ids in groups[groups.apply(len) >= 2]:
            for a, b in combinations(ids, 2):
                a_row, b_row = idx.loc[a], idx.loc[b]

                canonical_match = (
                    pd.notna(a_row["first_canonical"]) and pd.notna(b_row["first_canonical"]) and
                    a_row["first_canonical"] == b_row["first_canonical"] and
                    pd.notna(a_row["family_canonical"]) and pd.notna(b_row["family_canonical"]) and
                    a_row["family_canonical"] == b_row["family_canonical"]
                )
                if canonical_match:
                    score = 0.95
                else:
                    first_ratio = SequenceMatcher(None, a_row["norm_first"], b_row["norm_first"]).ratio()
                    family_ratio = SequenceMatcher(None, a_row["norm_family"], b_row["norm_family"]).ratio()
                    if first_ratio >= R2_ATTR_FIRST_NAME_FLOOR and family_ratio >= R2_FAMILY_NAME_THRESHOLD:
                        score = (first_ratio + family_ratio) / 2
                    else:
                        continue

                hits[a] = max(hits.get(a, 0.0), score)
                hits[b] = max(hits.get(b, 0.0), score)

    rows = [
        {"rule_id": "R2", "entity_type": "beneficiary", "entity_id": eid, "score": score,
         "detail": f"name-similarity + shared attribute duplicate candidate, score {score:.2f}"}
        for eid, score in hits.items()
    ]
    return pd.DataFrame(rows, columns=["rule_id", "entity_type", "entity_id", "score", "detail"])


# ---------------------------------------------------------------------------
# R3 -- Vendor night-transaction share, relative threshold (targets T3)
# ---------------------------------------------------------------------------
def rule_r3(transactions, vendors):
    tx = transactions.copy()
    tx["hour"] = tx["timestamp"].dt.hour
    tx["is_night"] = tx["hour"].between(0, 5)
    per_vendor = tx.groupby("vendor_id").agg(total=("txn_id", "count"), night=("is_night", "sum"))
    per_vendor["night_share"] = per_vendor["night"] / per_vendor["total"]
    per_vendor = per_vendor.reindex(vendors["vendor_id"]).fillna(0.0)

    mean, std, threshold = sigma_threshold(per_vendor["night_share"], R3_SIGMA)
    flagged = per_vendor[per_vendor["night_share"] > threshold].reset_index()
    flagged["rule_id"] = "R3"
    flagged["entity_type"] = "vendor"
    flagged["score"] = flagged["night_share"]
    flagged["detail"] = flagged.apply(
        lambda r: f"night-share {r['night_share']:.1%} vs population mean {mean:.1%} (+{R3_SIGMA}sigma={threshold:.1%})",
        axis=1,
    )
    return flagged.rename(columns={"vendor_id": "entity_id"})[
        ["rule_id", "entity_type", "entity_id", "score", "detail"]
    ]


# ---------------------------------------------------------------------------
# R4 -- Vendor share of total volume, relative threshold (targets T3)
# ---------------------------------------------------------------------------
def rule_r4(transactions, vendors):
    per_vendor = transactions.groupby("vendor_id")["amount"].sum()
    per_vendor = per_vendor.reindex(vendors["vendor_id"]).fillna(0.0)

    mean, std, threshold = sigma_threshold(per_vendor, R4_SIGMA)
    flagged = per_vendor[per_vendor > threshold].reset_index()
    flagged.columns = ["vendor_id", "volume"]
    flagged["rule_id"] = "R4"
    flagged["entity_type"] = "vendor"
    flagged["score"] = flagged["volume"]
    flagged["detail"] = flagged.apply(
        lambda r: f"volume ${r['volume']:,.0f} vs population mean ${mean:,.0f} (+{R4_SIGMA}sigma=${threshold:,.0f})",
        axis=1,
    )
    return flagged.rename(columns={"vendor_id": "entity_id"})[
        ["rule_id", "entity_type", "entity_id", "score", "detail"]
    ]


# ---------------------------------------------------------------------------
# R5 -- Registration burst: >N registrations/registrar/day (targets T1, T4)
# ---------------------------------------------------------------------------
def rule_r5(beneficiaries):
    per_day = beneficiaries.groupby(["registrar_id", "registration_date"]).size()
    mean, std, threshold = sigma_threshold(per_day.values, R5_SIGMA)
    burst_days = per_day[per_day > threshold]

    rows = []
    for (registrar_id, reg_date), count in burst_days.items():
        rows.append({"rule_id": "R5", "entity_type": "registrar", "entity_id": registrar_id,
                      "score": count,
                      "detail": f"{count} registrations on {reg_date.date()} vs mean {mean:.1f} (+{R5_SIGMA}sigma={threshold:.1f})"})
        members = beneficiaries[(beneficiaries["registrar_id"] == registrar_id) &
                                 (beneficiaries["registration_date"] == reg_date)]["beneficiary_id"]
        for bid in members:
            rows.append({"rule_id": "R5", "entity_type": "beneficiary", "entity_id": bid,
                          "score": count,
                          "detail": f"registered in a {count}-registration burst at {registrar_id} on {reg_date.date()}"})
    return pd.DataFrame(rows, columns=["rule_id", "entity_type", "entity_id", "score", "detail"])


# ---------------------------------------------------------------------------
# R6 (revised) / R6-naive -- cash-out velocity (targets T5)
# ---------------------------------------------------------------------------
def _fast_cashout_cycles(issuances, transactions):
    """For every (beneficiary, cycle) pair: txn count, fraction spent, hours
    to first redemption, and (for the fast ones) which vendor/timestamp."""
    tx = transactions.copy()
    tx["cycle_month"] = tx["timestamp"].dt.strftime("%Y-%m")
    agg = tx.groupby(["beneficiary_id", "cycle_month"]).agg(
        n_txns=("txn_id", "count"),
        spent=("amount", "sum"),
        first_ts=("timestamp", "min"),
    ).reset_index()

    merged = issuances.merge(agg, on=["beneficiary_id", "cycle_month"], how="left")
    merged["n_txns"] = merged["n_txns"].fillna(0)
    merged["spent"] = merged["spent"].fillna(0.0)
    merged["fraction_spent"] = merged["spent"] / merged["amount_issued"]
    merged["hours_to_first"] = (merged["first_ts"] - merged["issue_timestamp"]).dt.total_seconds() / 3600

    is_fast = (
        (merged["n_txns"] >= 1) & (merged["n_txns"] <= R6_MAX_TXNS_FOR_FAST) &
        (merged["fraction_spent"] >= R6_MIN_FRACTION_SPENT) &
        (merged["hours_to_first"] >= 0) & (merged["hours_to_first"] <= R6_MAX_HOURS_TO_FIRST)
    )
    return merged[is_fast].copy()


def rule_r6_naive(issuances, transactions):
    """Speed-and-fullness alone -- kept deliberately to demonstrate its
    precision collapse in a cash economy where ~40% of HONEST beneficiaries
    legitimately cash out this way (see fraud_typology_spec.md's explicit
    framing of Typology 5, and the Chunk 3-4 adjudication's note that the
    ring signal must be same-vendor+same-hour, never same-day alone)."""
    fast = _fast_cashout_cycles(issuances, transactions)
    rows = [
        {"rule_id": "R6_naive", "entity_type": "beneficiary", "entity_id": bid, "score": 1.0,
         "detail": "fast full cash-out (speed+fullness only, no locality/coordination check)"}
        for bid in fast["beneficiary_id"].unique()
    ]
    return pd.DataFrame(rows, columns=["rule_id", "entity_type", "entity_id", "score", "detail"])


def rule_r6(issuances, transactions, beneficiaries, vendors):
    fast = _fast_cashout_cycles(issuances, transactions)
    if fast.empty:
        return pd.DataFrame(columns=["rule_id", "entity_type", "entity_id", "score", "detail"])

    tx = transactions.copy()
    tx["cycle_month"] = tx["timestamp"].dt.strftime("%Y-%m")
    fast_txns = tx.merge(fast[["beneficiary_id", "cycle_month"]], on=["beneficiary_id", "cycle_month"], how="inner")

    ben_district = beneficiaries.set_index("beneficiary_id")["district"]
    vendor_district = vendors.set_index("vendor_id")["district"]
    fast_txns["ben_district"] = fast_txns["beneficiary_id"].map(ben_district)
    fast_txns["vendor_district"] = fast_txns["vendor_id"].map(vendor_district)
    fast_txns["non_local"] = fast_txns["ben_district"] != fast_txns["vendor_district"]

    # coordinated timing: an hour that's anomalous RELATIVE TO THAT VENDOR'S
    # OWN baseline, computed across ALL transactions (not just fast-cashout
    # ones) so genuinely coordinated rings show up even though each member's
    # own cycle-level stats look individually fine. A fixed global count
    # doesn't work here -- see R6_COORDINATION_SIGMA comment above.
    fast_txns["txn_date"] = fast_txns["timestamp"].dt.date
    fast_txns["txn_hour"] = fast_txns["timestamp"].dt.hour
    bucket_key = ["vendor_id", "txn_date", "txn_hour"]
    bucket_counts = tx.assign(
        txn_date=tx["timestamp"].dt.date, txn_hour=tx["timestamp"].dt.hour
    ).groupby(bucket_key)["beneficiary_id"].nunique()

    vendor_bucket_stats = bucket_counts.groupby("vendor_id").agg(["mean", "std"])
    vendor_bucket_stats["std"] = vendor_bucket_stats["std"].fillna(0.0)
    vendor_bucket_stats["threshold"] = (
        vendor_bucket_stats["mean"] + R6_COORDINATION_SIGMA * vendor_bucket_stats["std"]
    )

    fast_txns["bucket_count"] = fast_txns.set_index(bucket_key).index.map(bucket_counts)
    fast_txns["bucket_threshold"] = fast_txns["vendor_id"].map(vendor_bucket_stats["threshold"])
    fast_txns["coordinated"] = fast_txns["bucket_count"] > fast_txns["bucket_threshold"]

    flagged = fast_txns[fast_txns["non_local"] | fast_txns["coordinated"]]
    rows = []
    for bid, group in flagged.groupby("beneficiary_id"):
        reasons = []
        if group["non_local"].any():
            reasons.append("non-local vendor")
        if group["coordinated"].any():
            max_bucket = group.loc[group["coordinated"], "bucket_count"].max()
            reasons.append(f"coordinated with >={int(max_bucket)-1} others same vendor/hour")
        rows.append({"rule_id": "R6", "entity_type": "beneficiary", "entity_id": bid, "score": 1.0,
                      "detail": "fast full cash-out + " + " & ".join(reasons)})
    return pd.DataFrame(rows, columns=["rule_id", "entity_type", "entity_id", "score", "detail"])


# ---------------------------------------------------------------------------
# R7 -- Beneficiary-to-vendor distance past nearer options (targets T3)
#
# Simplification: "distance past nearer vendors" is operationalized as
# CROSS-DISTRICT redemption. Every district has vendors of every core type
# (guaranteed since Chunk 3's DI-005 fix), so a beneficiary crossing into
# another district has necessarily bypassed same-district options -- this
# avoids needing raw zone-coordinate distance math across 1.7M transactions
# while staying a faithful, defensible proxy for the same real-world signal.
# ---------------------------------------------------------------------------
def rule_r7(transactions, beneficiaries, vendors):
    ben_district = beneficiaries.set_index("beneficiary_id")["district"]
    vendor_district = vendors.set_index("vendor_id")["district"]
    tx = transactions.copy()
    tx["ben_district"] = tx["beneficiary_id"].map(ben_district)
    tx["vendor_district"] = tx["vendor_id"].map(vendor_district)
    tx["cross_district"] = tx["ben_district"] != tx["vendor_district"]

    per_vendor = tx.groupby("vendor_id").agg(total=("txn_id", "count"), cross=("cross_district", "sum"))
    per_vendor["cross_share"] = per_vendor["cross"] / per_vendor["total"]
    per_vendor = per_vendor.reindex(vendors["vendor_id"]).fillna(0.0)

    mean, std, threshold = sigma_threshold(per_vendor["cross_share"], R7_SIGMA)
    flagged = per_vendor[per_vendor["cross_share"] > threshold].reset_index()
    flagged["rule_id"] = "R7"
    flagged["entity_type"] = "vendor"
    flagged["score"] = flagged["cross_share"]
    flagged["detail"] = flagged.apply(
        lambda r: f"cross-district redemption share {r['cross_share']:.1%} vs mean {mean:.1%} (+{R7_SIGMA}sigma={threshold:.1%})",
        axis=1,
    )
    return flagged.rename(columns={"vendor_id": "entity_id"})[
        ["rule_id", "entity_type", "entity_id", "score", "detail"]
    ]


# ---------------------------------------------------------------------------
# R8 -- Registrar aggregate flag rate (targets T4). Meta-rule: runs after
# the beneficiary-level rules and aggregates by registrar.
# ---------------------------------------------------------------------------
def rule_r8(beneficiaries, beneficiary_flags_df):
    flagged_ids = set(beneficiary_flags_df["entity_id"].unique())
    per_registrar = beneficiaries.groupby("registrar_id")["beneficiary_id"].apply(list)
    rates = {}
    for registrar_id, ids in per_registrar.items():
        n_flagged = sum(1 for i in ids if i in flagged_ids)
        rates[registrar_id] = n_flagged / len(ids)

    mean, std, threshold = sigma_threshold(list(rates.values()), R8_SIGMA)
    rows = []
    for registrar_id, rate in rates.items():
        if rate > threshold:
            rows.append({"rule_id": "R8", "entity_type": "registrar", "entity_id": registrar_id,
                         "score": rate,
                         "detail": f"flag-rate {rate:.1%} vs mean {mean:.1%} (+{R8_SIGMA}sigma={threshold:.1%})"})
    return pd.DataFrame(rows, columns=["rule_id", "entity_type", "entity_id", "score", "detail"])


# ---------------------------------------------------------------------------
# R9 -- Same address/family: >=3 identical claimed household size (targets T2b)
# ---------------------------------------------------------------------------
def rule_r9(beneficiaries):
    rows = []
    grouped = beneficiaries.groupby(["address_id", "claimed_household_size"])["beneficiary_id"].apply(list)
    for (address_id, hh_size), ids in grouped.items():
        if len(ids) >= R9_MIN_GROUP_SIZE and hh_size * len(ids) > R9_MAX_PLAUSIBLE_RESIDENTS:
            for bid in ids:
                rows.append({"rule_id": "R9", "entity_type": "beneficiary", "entity_id": bid,
                             "score": hh_size * len(ids),
                             "detail": f"{len(ids)} registrants at {address_id} all claim household size {hh_size} "
                                       f"(sum {hh_size*len(ids)} > {R9_MAX_PLAUSIBLE_RESIDENTS} plausible residents)"})
    return pd.DataFrame(rows, columns=["rule_id", "entity_type", "entity_id", "score", "detail"])


# ---------------------------------------------------------------------------
# R10 -- Bulk same-day redemption at merchant vendor by address cluster (T2b)
# ---------------------------------------------------------------------------
def rule_r10(beneficiaries, transactions, vendors):
    address_groups = beneficiaries.groupby("address_id")["beneficiary_id"].apply(list)
    address_groups = address_groups[address_groups.apply(len) >= R9_MIN_GROUP_SIZE]
    if address_groups.empty:
        return pd.DataFrame(columns=["rule_id", "entity_type", "entity_id", "score", "detail"])

    merchant_vendor_ids = set(vendors.loc[vendors["vendor_type"] == "merchant-wholesale", "vendor_id"])
    tx = transactions.copy()
    tx["txn_date"] = tx["timestamp"].dt.date
    tx_merchant = tx[tx["vendor_id"].isin(merchant_vendor_ids)]

    rows = []
    for address_id, member_ids in address_groups.items():
        member_set = set(member_ids)
        member_tx = tx_merchant[tx_merchant["beneficiary_id"].isin(member_set)]
        if member_tx.empty:
            continue
        # find (vendor, date) combos hit by a majority of this cluster
        hits = member_tx.groupby(["vendor_id", "txn_date"])["beneficiary_id"].nunique()
        bulk_events = hits[hits / len(member_set) >= R10_BULK_SHARE_THRESHOLD]
        if bulk_events.empty:
            continue
        flagged_members = set()
        for (vendor_id, txn_date), n in bulk_events.items():
            flagged_members |= set(
                member_tx[(member_tx["vendor_id"] == vendor_id) & (member_tx["txn_date"] == txn_date)]["beneficiary_id"]
            )
        for bid in flagged_members:
            rows.append({"rule_id": "R10", "entity_type": "beneficiary", "entity_id": bid,
                         "score": len(bulk_events),
                         "detail": f"part of a same-address cluster ({address_id}, {len(member_set)} registrants) "
                                   f"with {len(bulk_events)} bulk same-day merchant-vendor event(s)"})
    return pd.DataFrame(rows, columns=["rule_id", "entity_type", "entity_id", "score", "detail"])


def write_flags(conn, all_flags_df):
    cur = conn.cursor()
    cur.executescript("DROP TABLE IF EXISTS rule_flags;")
    cur.execute("""
        CREATE TABLE rule_flags (
            rule_id     TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id   TEXT NOT NULL,
            score       REAL,
            detail      TEXT
        )
    """)
    conn.commit()
    rows = list(all_flags_df.itertuples(index=False, name=None))
    cur.executemany("INSERT INTO rule_flags VALUES (?,?,?,?,?)", rows)
    conn.commit()
    cur.execute("CREATE INDEX idx_rule_flags_rule ON rule_flags(rule_id)")
    cur.execute("CREATE INDEX idx_rule_flags_entity ON rule_flags(entity_type, entity_id)")
    conn.commit()


def print_summary(all_flags_df):
    print("\n--- Rule flag counts ---")
    summary = all_flags_df.groupby(["rule_id", "entity_type"])["entity_id"].nunique()
    for (rule_id, entity_type), count in summary.items():
        print(f"  {rule_id:10s} {entity_type:12s} {count:>7,} distinct entities flagged")


def main():
    parser = argparse.ArgumentParser(description="Chunk 5: detection rules R1-R10")
    parser.add_argument("--db", type=str, default="data/aid_program.db")
    args = parser.parse_args()

    t0 = time.time()
    conn = sqlite3.connect(args.db)

    print("Loading tables...")
    beneficiaries, vendors, registrars, issuances, transactions = load_tables(conn)
    print(f"  {len(beneficiaries):,} beneficiaries, {len(vendors)} vendors, {len(registrars)} registrars, "
          f"{len(issuances):,} issuances, {len(transactions):,} transactions")

    all_flags = []

    print("\nRunning R1 (shared phone >=3)...")
    r1 = rule_r1(beneficiaries); all_flags.append(r1)
    print(f"  {r1['entity_id'].nunique():,} flagged")

    print("Running R2 (name similarity + shared attribute)...")
    r2 = rule_r2(beneficiaries); all_flags.append(r2)
    print(f"  {r2['entity_id'].nunique():,} flagged")

    print("Running R3 (vendor night-share, relative)...")
    r3 = rule_r3(transactions, vendors); all_flags.append(r3)
    print(f"  {r3['entity_id'].nunique():,} flagged: {sorted(r3['entity_id'].tolist())}")

    print("Running R4 (vendor volume, relative)...")
    r4 = rule_r4(transactions, vendors); all_flags.append(r4)
    print(f"  {r4['entity_id'].nunique():,} flagged: {sorted(r4['entity_id'].tolist())}")

    print("Running R5 (registration burst)...")
    r5 = rule_r5(beneficiaries); all_flags.append(r5)
    r5_registrars = r5[r5["entity_type"] == "registrar"]["entity_id"].nunique()
    r5_beneficiaries = r5[r5["entity_type"] == "beneficiary"]["entity_id"].nunique()
    print(f"  {r5_registrars} registrar-days over threshold, {r5_beneficiaries:,} beneficiaries flagged")

    print("Running R6-naive (speed+fullness only -- expected to over-flag)...")
    r6n = rule_r6_naive(issuances, transactions); all_flags.append(r6n)
    print(f"  {r6n['entity_id'].nunique():,} flagged "
          f"({100*r6n['entity_id'].nunique()/len(beneficiaries):.1f}% of all beneficiaries)")

    print("Running R6 (revised: fast + non-local OR coordinated)...")
    r6 = rule_r6(issuances, transactions, beneficiaries, vendors); all_flags.append(r6)
    print(f"  {r6['entity_id'].nunique():,} flagged")

    print("Running R7 (cross-district redemption share, relative)...")
    r7 = rule_r7(transactions, beneficiaries, vendors); all_flags.append(r7)
    print(f"  {r7['entity_id'].nunique():,} flagged: {sorted(r7['entity_id'].tolist())}")

    beneficiary_flags_so_far = pd.concat(
        [f[f["entity_type"] == "beneficiary"] for f in all_flags], ignore_index=True
    )
    print("Running R8 (registrar aggregate flag rate, meta-rule)...")
    r8 = rule_r8(beneficiaries, beneficiary_flags_so_far); all_flags.append(r8)
    print(f"  {r8['entity_id'].nunique():,} flagged: {sorted(r8['entity_id'].tolist())}")

    print("Running R9 (identical household-size address cluster)...")
    r9 = rule_r9(beneficiaries); all_flags.append(r9)
    print(f"  {r9['entity_id'].nunique():,} flagged")

    print("Running R10 (bulk same-day merchant-vendor redemption)...")
    r10 = rule_r10(beneficiaries, transactions, vendors); all_flags.append(r10)
    print(f"  {r10['entity_id'].nunique():,} flagged")

    all_flags_df = pd.concat(all_flags, ignore_index=True)
    print(f"\nWriting {len(all_flags_df):,} total flag rows to rule_flags table...")
    write_flags(conn, all_flags_df)

    print_summary(all_flags_df)
    conn.close()

    print(f"\nDone in {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    main()
