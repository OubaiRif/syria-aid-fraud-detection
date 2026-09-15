"""
Chunk 3 -- Honest population generator.

Builds the honest majority of the synthetic humanitarian cash-transfer
dataset per specs/dataset_schema_spec.md and specs/fraud_typology_spec.md.
No fraud is injected here (that's Chunk 4, which extends this DB and
writes the answer_key). This script must run standalone and validate.

Usage:
    python src/generate_population.py --seed 42 --db data/aid_program.db
"""

import argparse
import random
import sqlite3
import time
from datetime import datetime, timedelta

import numpy as np

from names_data import FIRST_NAMES, FAMILY_NAMES
from geo_data import GOVERNORATE_DISTRICT_PAIRS, ALL_DISTRICTS, euclidean_distance

# ---------------------------------------------------------------------------
# Fixed population parameters (from dataset_schema_spec.md)
# ---------------------------------------------------------------------------
N_HONEST_BENEFICIARIES = 48_600
PCT_DISTRIBUTED_SPENDER = 0.60
PCT_FAST_CASHOUTER = 0.40
N_VENDORS = 120
N_REGISTRARS = 12
N_CYCLES = 12
CYCLE_MONTHS = [f"2025-{m:02d}" for m in range(4, 13)] + [f"2026-{m:02d}" for m in range(1, 4)]
REG_WINDOW_START = datetime(2025, 1, 1)
REG_WINDOW_END = datetime(2025, 3, 31)

N_PHONE_SHARE_FAMILIES = 500      # FP trap: extended families sharing 1 phone
N_SPLIT_HOUSEHOLD_FAMILIES = 200  # FP trap: consistent household-size splits

VENDOR_TYPES = ["grocery", "general", "merchant-wholesale", "pharmacy"]
VENDOR_TYPE_WEIGHTS = [0.40, 0.30, 0.10, 0.20]


def rand_name_variant(rng, name_dict):
    """Pick a random canonical name and one random variant of it."""
    canonical = rng.choice(list(name_dict.keys()))
    variant = rng.choice(name_dict[canonical])
    return canonical, variant


def build_schema(conn):
    cur = conn.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS beneficiaries;
        DROP TABLE IF EXISTS vendors;
        DROP TABLE IF EXISTS registrars;
        DROP TABLE IF EXISTS issuances;
        DROP TABLE IF EXISTS transactions;
        DROP TABLE IF EXISTS answer_key;

        CREATE TABLE registrars (
            registrar_id   TEXT PRIMARY KEY,
            office_district TEXT NOT NULL
        );

        CREATE TABLE vendors (
            vendor_id   TEXT PRIMARY KEY,
            vendor_name TEXT NOT NULL,
            vendor_type TEXT NOT NULL,
            governorate TEXT NOT NULL,
            district    TEXT NOT NULL,
            zone_x      REAL NOT NULL,
            zone_y      REAL NOT NULL
        );

        CREATE TABLE beneficiaries (
            beneficiary_id      TEXT PRIMARY KEY,
            full_name           TEXT NOT NULL,
            family_name         TEXT NOT NULL,
            phone               TEXT NOT NULL,
            address_id          TEXT NOT NULL,
            governorate         TEXT NOT NULL,
            district            TEXT NOT NULL,
            zone_x              REAL NOT NULL,
            zone_y              REAL NOT NULL,
            claimed_household_size INTEGER NOT NULL,
            registration_date   DATE NOT NULL,
            registrar_id        TEXT NOT NULL REFERENCES registrars(registrar_id)
        );

        CREATE TABLE issuances (
            issuance_id     TEXT PRIMARY KEY,
            beneficiary_id  TEXT NOT NULL REFERENCES beneficiaries(beneficiary_id),
            cycle_month     TEXT NOT NULL,
            amount_issued   REAL NOT NULL,
            issue_timestamp DATETIME NOT NULL
        );

        CREATE TABLE transactions (
            txn_id          TEXT PRIMARY KEY,
            beneficiary_id  TEXT NOT NULL REFERENCES beneficiaries(beneficiary_id),
            vendor_id       TEXT NOT NULL REFERENCES vendors(vendor_id),
            amount          REAL NOT NULL,
            timestamp       DATETIME NOT NULL
        );

        -- Created empty here; Chunk 4 populates it. Never joined during
        -- detection -- evaluation only.
        CREATE TABLE answer_key (
            entity_type TEXT NOT NULL,
            entity_id   TEXT NOT NULL,
            fraud_type  TEXT NOT NULL
        );
        """
    )
    conn.commit()


def generate_registrars(rng):
    rows = []
    for i in range(1, N_REGISTRARS + 1):
        registrar_id = f"REG-{i:02d}"
        office_district = rng.choice(ALL_DISTRICTS)
        rows.append((registrar_id, office_district))
    return rows


def generate_vendors(rng, np_rng):
    rows = []
    for i in range(1, N_VENDORS + 1):
        vendor_id = f"VEN-{i:03d}"
        vendor_type = np_rng.choice(VENDOR_TYPES, p=VENDOR_TYPE_WEIGHTS)
        governorate, district = rng.choice(GOVERNORATE_DISTRICT_PAIRS)
        zone_x, zone_y = np_rng.uniform(0, 100), np_rng.uniform(0, 100)
        vendor_name = f"{district} {vendor_type.title()} #{i}"
        rows.append([vendor_id, vendor_name, vendor_type, governorate, district,
                     round(float(zone_x), 2), round(float(zone_y), 2)])

    # Guarantee every district has >=1 merchant-wholesale vendor. Without
    # this, a district can end up with zero by chance (10% draw probability
    # x ~12 vendors/district means ~28% chance any given district misses
    # entirely), silently removing Typology 2b's resale channel there.
    districts_with_merchant = {row[4] for row in rows if row[2] == "merchant-wholesale"}
    missing_districts = [d for d in ALL_DISTRICTS if d not in districts_with_merchant]
    for district in missing_districts:
        candidates = [row for row in rows if row[4] == district]
        target = rng.choice(candidates)
        old_type = target[2]
        target[2] = "merchant-wholesale"
        target[1] = f"{district} Merchant-Wholesale #{target[0].split('-')[1]}"
        print(f"  [vendor fix] {district} had no merchant-wholesale vendor; "
              f"reassigned {target[0]} from {old_type}")

    return [tuple(row) for row in rows]


def random_reg_date(rng):
    delta_days = (REG_WINDOW_END - REG_WINDOW_START).days
    return REG_WINDOW_START + timedelta(days=rng.randint(0, delta_days))


def household_size_dist(rng):
    # Skewed toward 3-6, matching typical family sizes; range 1-12.
    weights = [3, 6, 9, 14, 16, 14, 10, 7, 5, 3, 2, 1]  # for sizes 1..12
    sizes = list(range(1, 13))
    return rng.choices(sizes, weights=weights, k=1)[0]


def generate_beneficiaries(rng, np_rng, registrar_ids):
    """Base honest population, before FP-trap grouping is layered on."""
    rows = []
    modes = []
    for i in range(1, N_HONEST_BENEFICIARIES + 1):
        beneficiary_id = f"BEN-{i:06d}"
        _, first_variant = rand_name_variant(rng, FIRST_NAMES)
        _, family_variant = rand_name_variant(rng, FAMILY_NAMES)
        full_name = f"{first_variant} {family_variant}"
        phone = "09" + "".join(str(rng.randint(0, 9)) for _ in range(8))
        address_id = f"ADDR-{i:06d}"
        governorate, district = rng.choice(GOVERNORATE_DISTRICT_PAIRS)
        zone_x, zone_y = float(np_rng.uniform(0, 100)), float(np_rng.uniform(0, 100))
        household_size = household_size_dist(rng)
        reg_date = random_reg_date(rng)
        registrar_id = rng.choice(registrar_ids)
        mode = "distributed" if rng.random() < PCT_DISTRIBUTED_SPENDER else "fast_cashout"

        rows.append({
            "beneficiary_id": beneficiary_id,
            "full_name": full_name,
            "family_name": family_variant,
            "phone": phone,
            "address_id": address_id,
            "governorate": governorate,
            "district": district,
            "zone_x": round(zone_x, 2),
            "zone_y": round(zone_y, 2),
            "claimed_household_size": household_size,
            "registration_date": reg_date.strftime("%Y-%m-%d"),
            "registrar_id": registrar_id,
        })
        modes.append(mode)
    return rows, modes


def apply_fp_traps(rng, beneficiaries):
    """
    Layer the two false-positive traps onto the base honest population,
    in place. These exist specifically to make naive R1/R9 rules over-fire
    on honest people -- see fraud_typology_spec.md 'honest majority'.
    """
    n = len(beneficiaries)
    all_idx = list(range(n))
    rng.shuffle(all_idx)
    cursor = 0

    # --- Trap A: extended families sharing one phone number ---------------
    phone_trap_groups = 0
    for _ in range(N_PHONE_SHARE_FAMILIES):
        gsize = rng.randint(2, 4)
        if cursor + gsize > n:
            break
        idxs = all_idx[cursor:cursor + gsize]
        cursor += gsize
        shared_phone = "09" + "".join(str(rng.randint(0, 9)) for _ in range(8))
        for idx in idxs:
            beneficiaries[idx]["phone"] = shared_phone
        phone_trap_groups += 1

    # --- Trap B: large families, consistent (non-identical) splits --------
    split_trap_groups = 0
    for _ in range(N_SPLIT_HOUSEHOLD_FAMILIES):
        gsize = rng.randint(2, 4)
        if cursor + gsize > n:
            break
        idxs = all_idx[cursor:cursor + gsize]
        cursor += gsize

        shared_address = f"ADDR-FAM-{split_trap_groups:04d}"
        _, family_variant = rand_name_variant(rng, FAMILY_NAMES)
        total_residents = rng.randint(6, 14)

        # split total_residents into gsize positive parts summing to it
        cuts = sorted(rng.sample(range(1, total_residents), gsize - 1)) if gsize > 1 else []
        bounds = [0] + cuts + [total_residents]
        splits = [max(1, bounds[i + 1] - bounds[i]) for i in range(gsize)]

        for idx, split_size in zip(idxs, splits):
            beneficiaries[idx]["address_id"] = shared_address
            beneficiaries[idx]["family_name"] = family_variant
            beneficiaries[idx]["claimed_household_size"] = split_size
            # keep full_name's surname consistent with the shared family name
            first_part = beneficiaries[idx]["full_name"].split(" ")[0]
            beneficiaries[idx]["full_name"] = f"{first_part} {family_variant}"
        split_trap_groups += 1

    return phone_trap_groups, split_trap_groups


def build_vendor_lookup(vendor_rows):
    """district -> list of (vendor_id, vendor_type, x, y); plus flat list of all vendor_ids."""
    by_district = {}
    all_vendor_ids = []
    for vendor_id, _, vendor_type, _, district, x, y in vendor_rows:
        by_district.setdefault(district, []).append((vendor_id, vendor_type, x, y))
        all_vendor_ids.append(vendor_id)
    return by_district, all_vendor_ids


def nearby_vendors_for(district_vendors, x, y, radius=15.0, min_count=3):
    """Returns a list of (vendor_id, vendor_type) tuples, nearest first."""
    dists = [(vid, vtype, euclidean_distance(x, y, vx, vy)) for vid, vtype, vx, vy in district_vendors]
    dists.sort(key=lambda t: t[2])
    within = [(vid, vtype) for vid, vtype, d in dists if d <= radius]
    if len(within) >= min_count:
        return within
    if dists:
        return [(vid, vtype) for vid, vtype, _ in dists[:min_count]]
    return []


def daytime_hour(rng, lo=9, hi=18):
    return rng.randint(lo, hi)


def morning_hour(rng, lo=7, hi=11):
    return rng.randint(lo, hi)


def household_factor(household_size, cap=2.5):
    return min(1 + 0.15 * (household_size - 1), cap)


def weights_to_amounts(target_total, weights, floor=0.5):
    """
    Convert normalized weights into dollar amounts summing to target_total,
    guaranteeing every transaction is at least `floor`. Food is cheap in
    this context (a kilo of tomatoes can be under $1), so the floor is
    deliberately low -- a single day's fresh produce run can legitimately
    cost under a dollar. It still prevents a $0.00 rounding artifact.
    """
    n = len(weights)
    if n == 0:
        return np.array([])
    total_floor = floor * n
    if total_floor >= target_total:
        return np.round(np.full(n, target_total / n), 2)
    remaining = target_total - total_floor
    amounts = floor + remaining * np.asarray(weights)
    return np.round(amounts, 2)


def generate_issuances_and_transactions(rng, np_rng, beneficiaries, modes, vendor_by_district, all_vendor_ids):
    issuance_rows = []
    txn_rows = []
    issuance_counter = 0
    txn_counter = 0

    # Precompute each beneficiary's local vendor pool once (not per cycle).
    # Each pool entry is (vendor_id, vendor_type).
    local_pools = []
    for b in beneficiaries:
        district_vendors = vendor_by_district.get(b["district"], [])
        if not district_vendors:
            # fall back to any vendor if a district happens to have none
            fallback_ids = rng.sample(all_vendor_ids, min(3, len(all_vendor_ids)))
            pool = [(vid, None) for vid in fallback_ids]
        else:
            pool = nearby_vendors_for(district_vendors, b["zone_x"], b["zone_y"])
        local_pools.append(pool)

    for bi, (b, mode, pool) in enumerate(zip(beneficiaries, modes, local_pools)):
        beneficiary_id = b["beneficiary_id"]
        household_size = b["claimed_household_size"]
        base_amount = 50.0 * household_factor(household_size)
        usual_vendor = pool[0][0]

        for cycle in CYCLE_MONTHS:
            # 3% of honest beneficiaries miss a random month
            if rng.random() < 0.03:
                continue

            issuance_counter += 1
            issuance_id = f"ISS-{issuance_counter:07d}"
            year, month = map(int, cycle.split("-"))
            dist_day = datetime(year, month, 5, morning_hour(rng), rng.randint(0, 59))
            amount_issued = round(base_amount * rng.uniform(0.97, 1.03), 2)
            issuance_rows.append((issuance_id, beneficiary_id, cycle, amount_issued,
                                   dist_day.strftime("%Y-%m-%d %H:%M:%S")))

            # occasional distant-vendor trip: market day, 2% of txns handled below per-txn
            if mode == "distributed":
                num_txns = rng.randint(2, 6)
                span_days = rng.randint(3, 10)
                target_total = amount_issued * rng.uniform(0.85, 1.00)

                # Real pattern: frequent, cheap, near-daily/every-other-day grocery
                # runs (hunting the cheapest available fresh food for that day --
                # tomatoes etc. can be well under $1/kg here), plus -- not every
                # cycle, but often -- one larger spend. That larger spend is a
                # BULK STAPLES stock-up (rice, flour, oil, sugar): each item is
                # cheap per unit, but buying a month's quantity at once still
                # adds up to real money, so this -- not medicine, which is also
                # cheap here relative to the US -- is what dominates the budget.
                # This is NOT an even split: most transactions stay small, with
                # at most one dominant outlier.
                big_idx = None
                if num_txns >= 3 and rng.random() < 0.55:
                    big_idx = rng.randrange(num_txns)
                    big_share = rng.uniform(0.55, 0.80)
                    small_n = num_txns - 1
                    small_weights = np_rng.dirichlet(alpha=[1.5] * small_n) * (1 - big_share)
                    weights = np.zeros(num_txns)
                    weights[big_idx] = big_share
                    j = 0
                    for i in range(num_txns):
                        if i == big_idx:
                            continue
                        weights[i] = small_weights[j]
                        j += 1
                else:
                    # No single dominant item this cycle -- still lumpy/uneven
                    # day-to-day (cheapest-available-food hunting varies daily
                    # spend), just without one outlier transaction.
                    weights = np_rng.dirichlet(alpha=[0.4] * num_txns)

                amounts = weights_to_amounts(target_total, weights, floor=0.5)

                for k in range(num_txns):
                    txn_counter += 1
                    txn_id = f"TXN-{txn_counter:08d}"
                    if rng.random() < 0.02:
                        vendor_id = rng.choice(all_vendor_ids)  # distant/market-day trip
                    elif k == big_idx:
                        # bulk-staples stock-up: prefer general/merchant-wholesale
                        # vendors in the local pool when available
                        candidates = [v for v in pool if v[1] in ("general", "merchant-wholesale")]
                        if candidates and rng.random() < 0.7:
                            vendor_id = rng.choice(candidates)[0]
                        else:
                            vendor_id = rng.choice(pool)[0]
                    else:
                        vendor_id = rng.choice(pool)[0]
                    day_offset = rng.randint(1, span_days)
                    ts = dist_day + timedelta(days=day_offset,
                                               hours=daytime_hour(rng) - dist_day.hour,
                                               minutes=rng.randint(0, 59) - dist_day.minute,
                                               seconds=rng.randint(0, 59))
                    txn_rows.append((txn_id, beneficiary_id, vendor_id, float(amounts[k]),
                                      ts.strftime("%Y-%m-%d %H:%M:%S")))
            else:  # fast_cashout
                num_txns = rng.randint(1, 2)
                target_total = amount_issued * rng.uniform(0.95, 1.00)
                weights = np.array([rng.random() for _ in range(num_txns)])
                weights = weights / weights.sum()
                amounts = weights_to_amounts(target_total, weights, floor=0.5)

                for k in range(num_txns):
                    txn_counter += 1
                    txn_id = f"TXN-{txn_counter:08d}"
                    if rng.random() < 0.02:
                        vendor_id = rng.choice(all_vendor_ids)
                    else:
                        vendor_id = usual_vendor
                    ts = dist_day + timedelta(seconds=rng.uniform(1, 6 * 3600))
                    txn_rows.append((txn_id, beneficiary_id, vendor_id, float(amounts[k]),
                                      ts.strftime("%Y-%m-%d %H:%M:%S")))

        # periodic progress ping for long runs
        if (bi + 1) % 10_000 == 0:
            print(f"  ... {bi + 1:,}/{len(beneficiaries):,} beneficiaries processed "
                  f"({issuance_counter:,} issuances, {txn_counter:,} txns so far)")

    return issuance_rows, txn_rows


def insert_batch(conn, sql, rows, batch_size=20_000):
    cur = conn.cursor()
    for i in range(0, len(rows), batch_size):
        cur.executemany(sql, rows[i:i + batch_size])
    conn.commit()


def validate(conn):
    cur = conn.cursor()
    print("\n--- Validation ---")
    checks_ok = True

    cur.execute("""
        SELECT district FROM vendors WHERE vendor_type='merchant-wholesale' GROUP BY district
    """)
    districts_with_merchant = {r[0] for r in cur.fetchall()}
    missing = [d for d in ALL_DISTRICTS if d not in districts_with_merchant]
    print(f"  Districts missing a merchant-wholesale vendor: {missing}  "
          f"[{'OK' if not missing else 'FAIL'}]")
    if missing:
        checks_ok = False

    for table, pk in [
        ("registrars", "registrar_id"),
        ("vendors", "vendor_id"),
        ("beneficiaries", "beneficiary_id"),
        ("issuances", "issuance_id"),
        ("transactions", "txn_id"),
    ]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        total = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM (SELECT {pk} FROM {table} GROUP BY {pk} HAVING COUNT(*) > 1)")
        dupes = cur.fetchone()[0]
        status = "OK" if dupes == 0 else "FAIL"
        if dupes != 0:
            checks_ok = False
        print(f"  {table:15s} rows={total:>10,}  duplicate_pks={dupes}  [{status}]")

    cur.execute("SELECT COUNT(*) FROM answer_key")
    ak_count = cur.fetchone()[0]
    print(f"  {'answer_key':15s} rows={ak_count:>10,}  (expected 0 -- Chunk 3 is honest-only)  "
          f"[{'OK' if ak_count == 0 else 'FAIL'}]")
    if ak_count != 0:
        checks_ok = False

    # a beneficiary should never appear at the exact same timestamp at two vendors
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT beneficiary_id, timestamp, COUNT(DISTINCT vendor_id) c
            FROM transactions GROUP BY beneficiary_id, timestamp HAVING c > 1
        )
    """)
    conflicts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM transactions")
    total_txns = cur.fetchone()[0]
    # A handful of same-second collisions are expected noise at this volume
    # (random second-level jitter over ~1.7M rows) -- not a schema/PK issue.
    # Only fail if it's more than a negligible fraction of all transactions.
    conflict_rate_ok = conflicts <= max(5, int(total_txns * 0.0005))
    print(f"  same-instant multi-vendor txn conflicts: {conflicts} / {total_txns:,}  "
          f"[{'OK' if conflict_rate_ok else 'FAIL'}]")
    if not conflict_rate_ok:
        checks_ok = False

    print(f"\nOverall: {'PASS' if checks_ok else 'FAIL'}")
    return checks_ok


def print_summary(conn, phone_trap_groups, split_trap_groups):
    cur = conn.cursor()
    print("\n--- Summary stats ---")
    cur.execute("SELECT COUNT(*) FROM beneficiaries")
    print(f"  Beneficiaries:        {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM vendors")
    print(f"  Vendors:               {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM registrars")
    print(f"  Registrars:            {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM issuances")
    print(f"  Issuances:             {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM transactions")
    print(f"  Transactions:          {cur.fetchone()[0]:,}")
    cur.execute("SELECT ROUND(AVG(amount),2) FROM transactions")
    print(f"  Avg transaction amt:   ${cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM (SELECT phone FROM beneficiaries GROUP BY phone HAVING COUNT(*) > 1)")
    print(f"  Distinct phones shared by >1 beneficiary: {cur.fetchone()[0]} "
          f"(expected ~{phone_trap_groups} phone-share trap families)")
    cur.execute("SELECT COUNT(*) FROM (SELECT address_id FROM beneficiaries GROUP BY address_id HAVING COUNT(*) > 1)")
    print(f"  Distinct addresses shared by >1 beneficiary: {cur.fetchone()[0]} "
          f"(expected ~{split_trap_groups} split-household trap families)")
    cur.execute("SELECT governorate, COUNT(*) FROM beneficiaries GROUP BY governorate")
    print(f"  Governorate split: {cur.fetchall()}")


def main():
    parser = argparse.ArgumentParser(description="Chunk 3: generate honest population")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=str, default="data/aid_program.db")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    t0 = time.time()
    conn = sqlite3.connect(args.db)
    build_schema(conn)

    print("Generating registrars and vendors...")
    registrar_rows = generate_registrars(rng)
    vendor_rows = generate_vendors(rng, np_rng)
    insert_batch(conn, "INSERT INTO registrars VALUES (?,?)", registrar_rows)
    insert_batch(conn, "INSERT INTO vendors VALUES (?,?,?,?,?,?,?)", vendor_rows)
    registrar_ids = [r[0] for r in registrar_rows]

    print(f"Generating {N_HONEST_BENEFICIARIES:,} honest beneficiaries...")
    beneficiaries, modes = generate_beneficiaries(rng, np_rng, registrar_ids)

    print("Layering false-positive traps (phone-sharing families, split-household families)...")
    phone_trap_groups, split_trap_groups = apply_fp_traps(rng, beneficiaries)
    print(f"  phone-share trap families: {phone_trap_groups}, split-household trap families: {split_trap_groups}")

    beneficiary_rows = [
        (b["beneficiary_id"], b["full_name"], b["family_name"], b["phone"], b["address_id"],
         b["governorate"], b["district"], b["zone_x"], b["zone_y"], b["claimed_household_size"],
         b["registration_date"], b["registrar_id"])
        for b in beneficiaries
    ]
    insert_batch(conn, "INSERT INTO beneficiaries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", beneficiary_rows)

    vendor_by_district, all_vendor_ids = build_vendor_lookup(vendor_rows)

    print("Generating issuances and transactions (this is the slow part)...")
    issuance_rows, txn_rows = generate_issuances_and_transactions(
        rng, np_rng, beneficiaries, modes, vendor_by_district, all_vendor_ids
    )
    print(f"  {len(issuance_rows):,} issuances, {len(txn_rows):,} transactions generated. Inserting...")
    insert_batch(conn, "INSERT INTO issuances VALUES (?,?,?,?,?)", issuance_rows)
    insert_batch(conn, "INSERT INTO transactions VALUES (?,?,?,?,?)", txn_rows)

    print("Creating indexes...")
    conn.executescript(
        """
        CREATE INDEX idx_beneficiaries_phone ON beneficiaries(phone);
        CREATE INDEX idx_beneficiaries_address ON beneficiaries(address_id);
        CREATE INDEX idx_beneficiaries_registrar ON beneficiaries(registrar_id);
        CREATE INDEX idx_issuances_beneficiary ON issuances(beneficiary_id);
        CREATE INDEX idx_transactions_beneficiary ON transactions(beneficiary_id);
        CREATE INDEX idx_transactions_vendor ON transactions(vendor_id);
        """
    )
    conn.commit()

    ok = validate(conn)
    print_summary(conn, phone_trap_groups, split_trap_groups)
    conn.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. DB written to {args.db}")
    if not ok:
        raise SystemExit("Validation FAILED -- see checks above.")


if __name__ == "__main__":
    main()
