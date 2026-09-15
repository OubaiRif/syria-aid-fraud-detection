"""
Chunk 4 -- Fraud injection.

Extends the DB built by generate_population.py (Chunk 3) with the six
fraud typologies from fraud_typology_spec.md, using the injection
parameters resolved in dataset_schema_spec.md. Writes every fraud
entity's fraud_type(s) to answer_key (never joined during detection --
evaluation only). Must be run against a freshly-generated Chunk 3 DB
(answer_key must be empty going in) since it is not idempotent -- running
it twice against the same DB would double-inject.

Usage:
    python src/generate_population.py --seed 42 --db data/aid_program.db
    python src/inject_fraud.py       --seed 42 --db data/aid_program.db
"""

import argparse
import random
import sqlite3
import time
from datetime import datetime, timedelta

import numpy as np

import generate_population as genpop
from names_data import FIRST_NAMES, FAMILY_NAMES, find_canonical
from geo_data import GOVERNORATE_DISTRICT_PAIRS

# ---------------------------------------------------------------------------
# Injection parameters (from dataset_schema_spec.md "Population & injection
# parameters (final)" table)
# ---------------------------------------------------------------------------
N_GHOSTS = 200
N_GHOST_PHONES = 40
N_GHOST_BURSTS = 6
GHOST_REG07_SHARE = 0.60
GHOST_COLLUDING_VENDOR_SHARE = 0.90

N_DUP_GROUPS = 150                 # 150 real (existing honest) people ...
DUP_EXTRA_IDS_CHOICES = [2, 3]     # ... each creating 2-3 EXTRA fraudulent IDs
DUP_EXTRA_IDS_WEIGHTS = [2, 1]     # avg 2.33 * 150 = ~350 extra IDs, matches spec
DUP_SHARE_PHONE_OR_ADDR = 0.70
DUP_REG07_SHARE = 0.25             # elevated vs ~1/12 baseline -- T4 fingerprint

N_FAMILY_GROUPS = 120
FAMILY_SIZE_RANGE = (3, 5)
FAMILY_BULK_REDEMPTION_SHARE = 0.70
FAMILY_CLAIMED_TOTAL_RANGE = (6, 10)

N_RING_MEMBERS = 300
N_RING_ONLY_VENDORS = 3            # + the 2 colluding vendors = 5 sub-rings

REGISTRAR_COMPROMISED = "REG-07"

# Typology 3 standalone over-invoicing: phantom-skim mechanism (per Fable
# adjudication -- reshaping already-fraud-routed transactions can't produce
# a standalone vendor-fabricates-volume scheme; it needs its own fabricated
# transactions against honest beneficiaries' unspent voucher balances).
PHANTOM_SKIM_MIN_REMAINDER = 1.0     # only skim meaningful leftover balance
PHANTOM_SKIM_TARGET_COUNT = 2500     # upper end of the adjudicated 1,500-2,500 range
PHANTOM_SKIM_NIGHT_SHARE = 0.65      # ~60-70% of phantom txns at night
PHANTOM_SKIM_BALANCE_EMPTYING_SHARE = 0.70  # ~70% take ~all of the remainder


class Counters:
    """Continues beneficiary/issuance/txn ID numbering from the existing DB
    so Chunk 4 rows never collide with Chunk 3's."""

    def __init__(self, ben_start, iss_start, txn_start):
        self.ben = ben_start
        self.iss = iss_start
        self.txn = txn_start

    def next_ben(self):
        self.ben += 1
        return f"BEN-{self.ben:06d}"

    def next_iss(self):
        self.iss += 1
        return f"ISS-{self.iss:07d}"

    def next_txn(self):
        self.txn += 1
        return f"TXN-{self.txn:08d}"


def get_counters(conn):
    cur = conn.cursor()
    cur.execute("SELECT beneficiary_id FROM beneficiaries ORDER BY beneficiary_id DESC LIMIT 1")
    ben_max = int(cur.fetchone()[0].split("-")[1])
    cur.execute("SELECT issuance_id FROM issuances ORDER BY issuance_id DESC LIMIT 1")
    iss_max = int(cur.fetchone()[0].split("-")[1])
    cur.execute("SELECT txn_id FROM transactions ORDER BY txn_id DESC LIMIT 1")
    txn_max = int(cur.fetchone()[0].split("-")[1])
    return Counters(ben_max, iss_max, txn_max)


def random_phone(rng):
    return "09" + "".join(str(rng.randint(0, 9)) for _ in range(8))


# ---------------------------------------------------------------------------
# Shared "looks honest" transaction simulator -- mirrors the distributed
# spender / fast cash-outer logic from Chunk 3, factored so Typology 2
# duplicates and Typology 2b's non-bulk cycles can reuse it. This is what
# makes those fraud entities individually indistinguishable from honest
# beneficiaries when viewed in isolation.
# ---------------------------------------------------------------------------
def simulate_one_cycle_transactions(rng, np_rng, counters, beneficiary_id, mode,
                                     dist_day, amount_issued, pool, all_vendor_ids):
    txn_rows = []
    if mode == "distributed":
        num_txns = rng.randint(2, 6)
        span_days = rng.randint(3, 10)
        target_total = amount_issued * rng.uniform(0.85, 1.00)

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
            weights = np_rng.dirichlet(alpha=[0.4] * num_txns)

        amounts = genpop.weights_to_amounts(target_total, weights, floor=0.5)

        for k in range(num_txns):
            txn_id = counters.next_txn()
            if rng.random() < 0.02:
                vendor_id = rng.choice(all_vendor_ids)
            elif k == big_idx:
                candidates = [v for v in pool if v[1] in ("general", "merchant-wholesale")]
                vendor_id = rng.choice(candidates)[0] if candidates and rng.random() < 0.7 else rng.choice(pool)[0]
            else:
                vendor_id = rng.choice(pool)[0]
            day_offset = rng.randint(1, span_days)
            ts = dist_day + timedelta(days=day_offset,
                                       hours=genpop.daytime_hour(rng) - dist_day.hour,
                                       minutes=rng.randint(0, 59) - dist_day.minute,
                                       seconds=rng.randint(0, 59))
            txn_rows.append((txn_id, beneficiary_id, vendor_id, float(amounts[k]),
                              ts.strftime("%Y-%m-%d %H:%M:%S")))
    else:  # fast_cashout
        num_txns = rng.randint(1, 2)
        target_total = amount_issued * rng.uniform(0.95, 1.00)
        weights = np.array([rng.random() for _ in range(num_txns)])
        weights = weights / weights.sum()
        amounts = genpop.weights_to_amounts(target_total, weights, floor=0.5)
        usual_vendor = pool[0][0]

        for k in range(num_txns):
            txn_id = counters.next_txn()
            vendor_id = rng.choice(all_vendor_ids) if rng.random() < 0.02 else usual_vendor
            ts = dist_day + timedelta(seconds=rng.uniform(1, 6 * 3600))
            txn_rows.append((txn_id, beneficiary_id, vendor_id, float(amounts[k]),
                              ts.strftime("%Y-%m-%d %H:%M:%S")))
    return txn_rows


def simulate_all_cycles(rng, np_rng, counters, beneficiary_id, mode, household_size,
                         pool, all_vendor_ids, skip_prob=0.03):
    issuance_rows = []
    txn_rows = []
    base_amount = 50.0 * genpop.household_factor(household_size)
    for cycle in genpop.CYCLE_MONTHS:
        if rng.random() < skip_prob:
            continue
        year, month = map(int, cycle.split("-"))
        dist_day = datetime(year, month, 5, genpop.morning_hour(rng), rng.randint(0, 59))
        amount_issued = round(base_amount * rng.uniform(0.97, 1.03), 2)
        issuance_id = counters.next_iss()
        issuance_rows.append((issuance_id, beneficiary_id, cycle, amount_issued,
                               dist_day.strftime("%Y-%m-%d %H:%M:%S")))
        txn_rows.extend(simulate_one_cycle_transactions(
            rng, np_rng, counters, beneficiary_id, mode, dist_day, amount_issued, pool, all_vendor_ids
        ))
    return issuance_rows, txn_rows


def local_pool_for(district, zone_x, zone_y, vendor_by_district, all_vendor_ids, rng):
    district_vendors = vendor_by_district.get(district, [])
    if not district_vendors:
        return [(v, None) for v in rng.sample(all_vendor_ids, min(3, len(all_vendor_ids)))]
    return genpop.nearby_vendors_for(district_vendors, zone_x, zone_y)


# ---------------------------------------------------------------------------
# Typology 1: Ghost Beneficiaries
# ---------------------------------------------------------------------------
def inject_ghosts(rng, np_rng, counters, registrar_ids, colluding_vendors, all_vendor_ids):
    beneficiary_rows, issuance_rows, txn_rows, answer_rows = [], [], [], []

    # ~40 shared phones, each anchoring one implausibly-clustered fake address
    phones = [random_phone(rng) for _ in range(N_GHOST_PHONES)]
    phone_geo = {}
    for idx, ph in enumerate(phones):
        governorate, district = rng.choice(GOVERNORATE_DISTRICT_PAIRS)
        zone_x, zone_y = float(np_rng.uniform(0, 100)), float(np_rng.uniform(0, 100))
        phone_geo[ph] = (governorate, district, round(zone_x, 2), round(zone_y, 2),
                          f"ADDR-GHOST-{idx:03d}")

    # 6 registration bursts, one shared burst_date each
    burst_sizes = [N_GHOSTS // N_GHOST_BURSTS] * N_GHOST_BURSTS
    for i in range(N_GHOSTS - sum(burst_sizes)):
        burst_sizes[i] += 1
    other_registrars = [r for r in registrar_ids if r != REGISTRAR_COMPROMISED]

    for burst_size in burst_sizes:
        burst_date = genpop.random_reg_date(rng)
        template_hh_size = genpop.household_size_dist(rng)  # REG-07 template-uniformity signal

        for _ in range(burst_size):
            ben_id = counters.next_ben()
            phone = rng.choice(phones)
            governorate, district, zone_x, zone_y, address_id = phone_geo[phone]
            _, first_variant = genpop.rand_name_variant(rng, FIRST_NAMES)
            _, family_variant = genpop.rand_name_variant(rng, FAMILY_NAMES)
            full_name = f"{first_variant} {family_variant}"

            is_reg07 = rng.random() < GHOST_REG07_SHARE
            registrar_id = REGISTRAR_COMPROMISED if is_reg07 else rng.choice(other_registrars)
            household_size = template_hh_size if is_reg07 else genpop.household_size_dist(rng)

            beneficiary_rows.append((ben_id, full_name, family_variant, phone, address_id,
                                      governorate, district, zone_x, zone_y, household_size,
                                      burst_date.strftime("%Y-%m-%d"), registrar_id))
            answer_rows.append(("beneficiary", ben_id, "ghost"))
            if is_reg07:
                answer_rows.append(("beneficiary", ben_id, "registrar_diversion"))

            # fixed personal cash-out vendor + fixed time-of-day: the
            # "unnaturally regular, no household-like variation" fingerprint.
            # Ghosts are dedicated to colluding_vendors[0] specifically (the
            # "quiet" ghost-cash-out vendor) -- colluding_vendors[1] is the
            # "loud" standalone over-invoicing vendor, built up separately
            # via the phantom-skim mechanism (see inject_vendor_phantom_skim).
            ghost_vendor = colluding_vendors[0]
            uses_colluding = rng.random() < GHOST_COLLUDING_VENDOR_SHARE
            fixed_vendor = ghost_vendor if uses_colluding else rng.choice(all_vendor_ids)
            fixed_hour = rng.randint(9, 17)
            fixed_minute = rng.randint(0, 59)
            base_amount = 50.0 * genpop.household_factor(household_size)

            for cycle in genpop.CYCLE_MONTHS:
                year, month = map(int, cycle.split("-"))
                amount_issued = round(base_amount * rng.uniform(0.98, 1.02), 2)
                issuance_id = counters.next_iss()
                dist_ts = datetime(year, month, 5, genpop.morning_hour(rng), rng.randint(0, 59))
                issuance_rows.append((issuance_id, ben_id, cycle, amount_issued,
                                       dist_ts.strftime("%Y-%m-%d %H:%M:%S")))

                txn_id = counters.next_txn()
                ts = datetime(year, month, 5, fixed_hour, fixed_minute, rng.randint(0, 59))
                amount = amount_issued
                if fixed_vendor == ghost_vendor:
                    # T3 fingerprint on the quiet vendor: diluted by design --
                    # this vendor's aggregate signal stays low (~2% night
                    # share), only detectable via ghost-beneficiary linkage
                    # (R1/R5), not vendor-aggregate rules (R3/R4).
                    if rng.random() >= 0.80:
                        amount = round(amount_issued * rng.uniform(0.90, 0.99), 2)
                    if rng.random() < 0.15:
                        ts = ts.replace(hour=rng.randint(0, 4))
                txn_rows.append((txn_id, ben_id, fixed_vendor, amount, ts.strftime("%Y-%m-%d %H:%M:%S")))

    return beneficiary_rows, issuance_rows, txn_rows, answer_rows


# ---------------------------------------------------------------------------
# Typology 2: Duplicate Registration
# ---------------------------------------------------------------------------
def inject_duplicates(rng, np_rng, counters, conn, registrar_ids, vendor_by_district, all_vendor_ids):
    cur = conn.cursor()
    cur.execute("""SELECT beneficiary_id, full_name, family_name, phone, address_id,
                          governorate, district, zone_x, zone_y, claimed_household_size,
                          registration_date, registrar_id
                   FROM beneficiaries""")
    all_honest = cur.fetchall()
    anchors = rng.sample(all_honest, N_DUP_GROUPS)
    other_registrars = [r for r in registrar_ids if r != REGISTRAR_COMPROMISED]

    beneficiary_rows, issuance_rows, txn_rows, answer_rows = [], [], [], []

    for anchor in anchors:
        (a_id, a_full, a_family, a_phone, a_addr, a_gov, a_dist, a_x, a_y,
         a_hh, _a_regdate, a_registrar) = anchor

        answer_rows.append(("beneficiary", a_id, "duplicate"))
        if a_registrar == REGISTRAR_COMPROMISED:
            answer_rows.append(("beneficiary", a_id, "registrar_diversion"))

        a_first = a_full.split(" ")[0]
        first_canonical = find_canonical(a_first, FIRST_NAMES)
        family_canonical = find_canonical(a_family, FAMILY_NAMES)

        shares_attr = rng.random() < DUP_SHARE_PHONE_OR_ADDR
        share_mode = rng.choice(["phone", "address"]) if shares_attr else None
        n_extra = rng.choices(DUP_EXTRA_IDS_CHOICES, weights=DUP_EXTRA_IDS_WEIGHTS)[0]

        for _ in range(n_extra):
            ben_id = counters.next_ben()

            if first_canonical:
                variants = [v for v in FIRST_NAMES[first_canonical] if v != a_first]
                first_variant = rng.choice(variants) if variants else a_first
            else:
                first_variant = a_first
            if family_canonical:
                variants = [v for v in FAMILY_NAMES[family_canonical] if v != a_family]
                family_variant = rng.choice(variants) if variants else a_family
            else:
                family_variant = a_family
            full_name = f"{first_variant} {family_variant}"

            if share_mode == "phone":
                phone = a_phone
                address_id = f"ADDR-{ben_id.split('-')[1]}"
                governorate, district = rng.choice(GOVERNORATE_DISTRICT_PAIRS)
                zone_x, zone_y = round(float(np_rng.uniform(0, 100)), 2), round(float(np_rng.uniform(0, 100)), 2)
            elif share_mode == "address":
                phone = random_phone(rng)
                address_id = a_addr
                governorate, district, zone_x, zone_y = a_gov, a_dist, a_x, a_y
            else:
                phone = random_phone(rng)
                address_id = f"ADDR-{ben_id.split('-')[1]}"
                governorate, district = rng.choice(GOVERNORATE_DISTRICT_PAIRS)
                zone_x, zone_y = round(float(np_rng.uniform(0, 100)), 2), round(float(np_rng.uniform(0, 100)), 2)

            household_size = a_hh  # near-identical household composition
            reg_date = genpop.random_reg_date(rng)
            is_reg07 = rng.random() < DUP_REG07_SHARE
            registrar_id = REGISTRAR_COMPROMISED if is_reg07 else rng.choice(other_registrars)

            beneficiary_rows.append((ben_id, full_name, family_variant, phone, address_id,
                                      governorate, district, zone_x, zone_y, household_size,
                                      reg_date.strftime("%Y-%m-%d"), registrar_id))
            answer_rows.append(("beneficiary", ben_id, "duplicate"))
            if is_reg07:
                answer_rows.append(("beneficiary", ben_id, "registrar_diversion"))

            pool = local_pool_for(district, zone_x, zone_y, vendor_by_district, all_vendor_ids, rng)
            mode = "distributed" if rng.random() < 0.6 else "fast_cashout"
            iss, txns = simulate_all_cycles(rng, np_rng, counters, ben_id, mode, household_size,
                                             pool, all_vendor_ids)
            issuance_rows.extend(iss)
            txn_rows.extend(txns)

    return beneficiary_rows, issuance_rows, txn_rows, answer_rows


# ---------------------------------------------------------------------------
# Typology 2b: Intra-Family Eligibility Gaming
# ---------------------------------------------------------------------------
def inject_family_gaming(rng, np_rng, counters, registrar_ids, vendor_by_district, all_vendor_ids):
    beneficiary_rows, issuance_rows, txn_rows, answer_rows = [], [], [], []

    for fam_i in range(N_FAMILY_GROUPS):
        gsize = rng.randint(*FAMILY_SIZE_RANGE)
        governorate, district = rng.choice(GOVERNORATE_DISTRICT_PAIRS)
        zone_x, zone_y = round(float(np_rng.uniform(0, 100)), 2), round(float(np_rng.uniform(0, 100)), 2)
        address_id = f"ADDR-FAMGAME-{fam_i:04d}"
        _, family_variant = genpop.rand_name_variant(rng, FAMILY_NAMES)
        # every member claims this SAME full size -- arithmetically impossible,
        # unlike Chunk 3's honest split-household FP trap
        claimed_total = rng.randint(*FAMILY_CLAIMED_TOTAL_RANGE)

        pool = local_pool_for(district, zone_x, zone_y, vendor_by_district, all_vendor_ids, rng)
        merchant_candidates = [v[0] for v in pool if v[1] == "merchant-wholesale"]
        family_vendors = merchant_candidates[:2] if merchant_candidates else [pool[0][0]]

        member_ids = []
        member_modes = {}
        for _ in range(gsize):
            ben_id = counters.next_ben()
            _, first_variant = genpop.rand_name_variant(rng, FIRST_NAMES)
            full_name = f"{first_variant} {family_variant}"
            reg_date = genpop.random_reg_date(rng)
            registrar_id = rng.choice(registrar_ids)

            beneficiary_rows.append((ben_id, full_name, family_variant, random_phone(rng), address_id,
                                      governorate, district, zone_x, zone_y, claimed_total,
                                      reg_date.strftime("%Y-%m-%d"), registrar_id))
            answer_rows.append(("beneficiary", ben_id, "family_gaming"))
            member_ids.append(ben_id)
            member_modes[ben_id] = "distributed" if rng.random() < 0.6 else "fast_cashout"

        base_amount = 50.0 * genpop.household_factor(claimed_total)

        for cycle in genpop.CYCLE_MONTHS:
            year, month = map(int, cycle.split("-"))
            dist_day = datetime(year, month, 5, genpop.morning_hour(rng), rng.randint(0, 59))
            bulk_cycle = rng.random() < FAMILY_BULK_REDEMPTION_SHARE

            if bulk_cycle:
                # signature fingerprint: whole family redeems in bulk, same
                # day, at a shared merchant-type vendor (resale pattern)
                bulk_day_offset = rng.randint(0, 5)
                bulk_vendor = rng.choice(family_vendors)
                bulk_hour = rng.randint(10, 16)
                for mid in member_ids:
                    amount_issued = round(base_amount * rng.uniform(0.97, 1.03), 2)
                    issuance_id = counters.next_iss()
                    issuance_rows.append((issuance_id, mid, cycle, amount_issued,
                                           dist_day.strftime("%Y-%m-%d %H:%M:%S")))
                    txn_id = counters.next_txn()
                    ts = dist_day + timedelta(days=bulk_day_offset,
                                               hours=bulk_hour - dist_day.hour,
                                               minutes=rng.randint(0, 29) - dist_day.minute,
                                               seconds=rng.randint(0, 59))
                    amount = round(amount_issued * rng.uniform(0.95, 1.0), 2)
                    txn_rows.append((txn_id, mid, bulk_vendor, amount, ts.strftime("%Y-%m-%d %H:%M:%S")))
            else:
                # the other ~30%: individually-clean, ordinary household spending
                for mid in member_ids:
                    amount_issued = round(base_amount * rng.uniform(0.97, 1.03), 2)
                    issuance_id = counters.next_iss()
                    issuance_rows.append((issuance_id, mid, cycle, amount_issued,
                                           dist_day.strftime("%Y-%m-%d %H:%M:%S")))
                    txn_rows.extend(simulate_one_cycle_transactions(
                        rng, np_rng, counters, mid, member_modes[mid], dist_day, amount_issued,
                        pool, all_vendor_ids
                    ))

    return beneficiary_rows, issuance_rows, txn_rows, answer_rows


# ---------------------------------------------------------------------------
# Typology 5: Cash-Out Ring (cash-economy calibrated -- see spec)
# ---------------------------------------------------------------------------
def inject_cashout_ring(rng, np_rng, counters, registrar_ids, colluding_vendors, ring_only_vendors):
    beneficiary_rows, issuance_rows, txn_rows, answer_rows = [], [], [], []

    half = N_RING_MEMBERS // 2
    per_colluding = half // 2
    subgroups = [
        (colluding_vendors[0], per_colluding),
        (colluding_vendors[1], half - per_colluding),
    ]
    remaining = N_RING_MEMBERS - half
    per_ring_only = remaining // len(ring_only_vendors)
    counts = [per_ring_only] * len(ring_only_vendors)
    for i in range(remaining - sum(counts)):
        counts[i] += 1
    subgroups.extend(zip(ring_only_vendors, counts))

    for vendor_id, count in subgroups:
        ring_hour = rng.randint(10, 16)  # shared synchronized cash-out hour
        is_colluding_group = vendor_id in colluding_vendors

        for _ in range(count):
            ben_id = counters.next_ben()
            _, first_variant = genpop.rand_name_variant(rng, FIRST_NAMES)
            _, family_variant = genpop.rand_name_variant(rng, FAMILY_NAMES)
            full_name = f"{first_variant} {family_variant}"
            governorate, district = rng.choice(GOVERNORATE_DISTRICT_PAIRS)
            zone_x, zone_y = round(float(np_rng.uniform(0, 100)), 2), round(float(np_rng.uniform(0, 100)), 2)
            household_size = genpop.household_size_dist(rng)
            reg_date = genpop.random_reg_date(rng)
            registrar_id = rng.choice(registrar_ids)

            beneficiary_rows.append((ben_id, full_name, family_variant, random_phone(rng),
                                      f"ADDR-{ben_id.split('-')[1]}", governorate, district,
                                      zone_x, zone_y, household_size, reg_date.strftime("%Y-%m-%d"),
                                      registrar_id))
            answer_rows.append(("beneficiary", ben_id, "cashout_ring"))

            base_amount = 50.0 * genpop.household_factor(household_size)
            for cycle in genpop.CYCLE_MONTHS:
                if rng.random() < 0.05:  # small participation noise
                    continue
                year, month = map(int, cycle.split("-"))
                amount_issued = round(base_amount * rng.uniform(0.97, 1.03), 2)
                dist_ts = datetime(year, month, 5, genpop.morning_hour(rng), rng.randint(0, 59))
                issuance_id = counters.next_iss()
                issuance_rows.append((issuance_id, ben_id, cycle, amount_issued,
                                       dist_ts.strftime("%Y-%m-%d %H:%M:%S")))

                txn_id = counters.next_txn()
                # everyone in this subgroup lands in the SAME hour -- the
                # "coordinated timing" fingerprint that distinguishes this
                # from ordinary (individually fast but uncoordinated) honest
                # fast cash-outers
                ts = datetime(year, month, 5, ring_hour, rng.randint(0, 59), rng.randint(0, 59))
                if is_colluding_group:
                    amount = amount_issued if rng.random() < 0.80 else round(amount_issued * rng.uniform(0.90, 0.99), 2)
                    if rng.random() < 0.15:
                        ts = ts.replace(hour=rng.randint(0, 4))
                else:
                    amount = round(amount_issued * rng.uniform(0.95, 1.0), 2)
                txn_rows.append((txn_id, ben_id, vendor_id, amount, ts.strftime("%Y-%m-%d %H:%M:%S")))

    return beneficiary_rows, issuance_rows, txn_rows, answer_rows


# ---------------------------------------------------------------------------
# Typology 3 (standalone half): phantom-skim on the "loud" colluding vendor
# ---------------------------------------------------------------------------
def inject_vendor_phantom_skim(rng, conn, counters, standalone_vendor):
    """
    Fabricates transactions at the standalone over-invoicing vendor,
    consuming most of the unspent remainder on honest beneficiaries'
    vouchers. These beneficiaries are victims, not perpetrators -- they
    never chose this vendor -- so they are NOT added to answer_key. Only
    the vendor itself carries the vendor_collusion tag (added in main()).

    This is deliberately a separate mechanism from the ghost/ring traffic
    that lands on colluding_vendors[0]: T3's own scheme is the vendor
    fabricating volume against stale balances, which reshaping fraud-actor
    transactions cannot produce.
    """
    cur = conn.cursor()
    cur.execute("""
        CREATE TEMP TABLE txn_cycle_totals AS
        SELECT beneficiary_id, strftime('%Y-%m', timestamp) AS cycle_month, SUM(amount) AS spent
        FROM transactions
        GROUP BY beneficiary_id, cycle_month
    """)
    cur.execute("CREATE INDEX idx_tct ON txn_cycle_totals(beneficiary_id, cycle_month)")
    cur.execute(f"""
        SELECT i.beneficiary_id, i.amount_issued, i.issue_timestamp,
               i.amount_issued - COALESCE(tct.spent, 0) AS remainder
        FROM issuances i
        LEFT JOIN txn_cycle_totals tct
          ON tct.beneficiary_id = i.beneficiary_id AND tct.cycle_month = i.cycle_month
        WHERE (i.amount_issued - COALESCE(tct.spent, 0)) >= {PHANTOM_SKIM_MIN_REMAINDER}
    """)
    candidates = cur.fetchall()
    cur.execute("DROP TABLE txn_cycle_totals")

    sample_size = min(PHANTOM_SKIM_TARGET_COUNT, len(candidates))
    targets = rng.sample(candidates, sample_size)

    txn_rows = []
    balance_emptying_count = 0
    night_count = 0
    for beneficiary_id, _amount_issued, issue_timestamp_str, remainder in targets:
        is_balance_emptying = rng.random() < PHANTOM_SKIM_BALANCE_EMPTYING_SHARE
        if is_balance_emptying:
            amount = round(remainder * rng.uniform(0.97, 1.00), 2)
        else:
            amount = round(remainder * rng.uniform(0.50, 0.90), 2)
        if amount <= 0:
            continue

        issue_ts = datetime.strptime(issue_timestamp_str, "%Y-%m-%d %H:%M:%S")
        # phantom redemption processed sometime within that voucher cycle,
        # skewed toward night hours (stale-balance terminal activity)
        day_offset = rng.randint(0, 20)
        is_night = rng.random() < PHANTOM_SKIM_NIGHT_SHARE
        hour = rng.randint(0, 5) if is_night else rng.randint(6, 23)
        ts = issue_ts + timedelta(days=day_offset)
        ts = ts.replace(hour=hour, minute=rng.randint(0, 59), second=rng.randint(0, 59))

        txn_id = counters.next_txn()
        txn_rows.append((txn_id, beneficiary_id, standalone_vendor, amount,
                          ts.strftime("%Y-%m-%d %H:%M:%S")))
        if is_balance_emptying:
            balance_emptying_count += 1
        if is_night:
            night_count += 1

    stats = {
        "phantom_total": len(txn_rows),
        "phantom_balance_emptying": balance_emptying_count,
        "phantom_night": night_count,
    }
    return txn_rows, stats


def insert_batch(conn, sql, rows, batch_size=20_000):
    if not rows:
        return
    cur = conn.cursor()
    for i in range(0, len(rows), batch_size):
        cur.executemany(sql, rows[i:i + batch_size])
    conn.commit()


def validate(conn, expected_new_beneficiaries):
    cur = conn.cursor()
    print("\n--- Validation ---")
    checks_ok = True

    for table, pk in [
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
    ak_total = cur.fetchone()[0]
    print(f"  answer_key rows: {ak_total:,}")
    ok = ak_total > 0
    checks_ok = checks_ok and ok
    print(f"  answer_key non-empty  [{'OK' if ok else 'FAIL'}]")

    # every beneficiary_id in answer_key must exist in beneficiaries (or be
    # a vendor/registrar id for those entity types)
    cur.execute("""
        SELECT COUNT(*) FROM answer_key a
        WHERE a.entity_type = 'beneficiary'
          AND a.entity_id NOT IN (SELECT beneficiary_id FROM beneficiaries)
    """)
    orphans = cur.fetchone()[0]
    print(f"  answer_key beneficiary rows with no matching beneficiary: {orphans}  "
          f"[{'OK' if orphans == 0 else 'FAIL'}]")
    if orphans != 0:
        checks_ok = False

    cur.execute("SELECT entity_type, fraud_type, COUNT(*) FROM answer_key GROUP BY entity_type, fraud_type ORDER BY 1,2")
    print("\n  answer_key breakdown:")
    for entity_type, fraud_type, c in cur.fetchall():
        print(f"    {entity_type:12s} {fraud_type:22s} {c:>6,}")

    print(f"\nOverall: {'PASS' if checks_ok else 'FAIL'}")
    return checks_ok


def print_summary(conn, counts):
    cur = conn.cursor()
    print("\n--- Summary stats ---")
    cur.execute("SELECT COUNT(*) FROM beneficiaries")
    print(f"  Total beneficiaries (honest + fraud): {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM issuances")
    print(f"  Total issuances:                      {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM transactions")
    print(f"  Total transactions:                   {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(DISTINCT entity_id) FROM answer_key WHERE entity_type='beneficiary'")
    flagged = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM beneficiaries")
    total_b = cur.fetchone()[0]
    print(f"  Distinct fraud-flagged beneficiaries:  {flagged:,} ({100 * flagged / total_b:.2f}% of {total_b:,})")
    for name, n in counts.items():
        print(f"  New rows from {name:18s}: {n:,}")


def print_vendor_fraud_stats(conn, colluding_vendors, phantom_stats):
    cur = conn.cursor()
    print("\n--- Typology 3 vendor-aggregate stats (per adjudication) ---")

    def night_share(vendor_id):
        cur.execute("SELECT COUNT(*) FROM transactions WHERE vendor_id=?", (vendor_id,))
        total = cur.fetchone()[0]
        if total == 0:
            return 0, 0.0
        cur.execute("""SELECT COUNT(*) FROM transactions
                       WHERE vendor_id=? AND CAST(strftime('%H', timestamp) AS INT) BETWEEN 0 AND 5""",
                    (vendor_id,))
        night = cur.fetchone()[0]
        return total, 100 * night / total

    ghost_vendor, standalone_vendor = colluding_vendors[0], colluding_vendors[1]
    for label, vid in [("Ghost/quiet vendor", ghost_vendor), ("Standalone/loud vendor", standalone_vendor)]:
        total, night_pct = night_share(vid)
        print(f"  {label:24s} {vid}: {total:>8,} txns, night-share {night_pct:5.1f}%")

    cur.execute("SELECT vendor_id FROM vendors WHERE vendor_id NOT IN (?,?)", (ghost_vendor, standalone_vendor))
    other_vendors = [r[0] for r in cur.fetchall()]
    tot_txns = tot_night = 0
    for vid in other_vendors:
        total, night_pct = night_share(vid)
        tot_txns += total
        tot_night += night_pct * total / 100
    if tot_txns:
        print(f"  {'Honest-vendor baseline':24s} (avg of {len(other_vendors)}): {tot_txns:>8,} txns, "
              f"night-share {100 * tot_night / tot_txns:5.1f}%")

    # Balance-emptying share: reported from GENERATION-TIME ground truth,
    # not inferred from the DB. An amount-threshold proxy (tried $40, then
    # $20) was checked and rejected -- it doesn't discriminate, because
    # phantom-skim amounts are small (they're draining a small leftover
    # balance, typically well under $20) and the honest population's own
    # amounts already skew low after the DI-002 cheap-food calibration
    # (median honest transaction ~$14-18). A detector with no ground truth
    # (i.e. Chunk 5's rules) will face this exact problem -- amount alone
    # won't separate this pattern from ordinary cheap honest transactions;
    # it needs remainder/pattern-level reasoning, not a fixed threshold.
    p_total = phantom_stats["phantom_total"]
    p_empty = phantom_stats["phantom_balance_emptying"]
    p_night = phantom_stats["phantom_night"]
    print(f"\n  Phantom-skim ground truth (known at generation time, NOT DB-inferred):")
    print(f"    {p_total:,} phantom txns fabricated at {standalone_vendor}; "
          f"{p_empty:,} ({100*p_empty/p_total:.1f}%) balance-emptying, "
          f"{p_night:,} ({100*p_night/p_total:.1f}%) at night")
    print(f"    NOTE: an amount-threshold DB proxy for 'balance-emptying share' was tested and")
    print(f"    found NOT to discriminate this vendor from the honest baseline (both proxy amounts")
    print(f"    tried land within ~5pp of the honest average) -- flagging this as a finding for the")
    print(f"    README rather than papering over it with a threshold that happens to move the number.")


def print_definition_of_done(conn, all_beneficiary_rows):
    cur = conn.cursor()
    print("\n--- Definition of done checklist ---")

    cur.execute("SELECT district FROM vendors WHERE vendor_type='merchant-wholesale' GROUP BY district")
    have = {r[0] for r in cur.fetchall()}
    missing = [d for d in genpop.ALL_DISTRICTS if d not in have]
    print(f"  Every district has >=1 merchant-wholesale vendor: {'YES' if not missing else 'NO -- missing: ' + str(missing)}")

    cur.execute("SELECT COUNT(DISTINCT entity_id) FROM answer_key WHERE entity_type='beneficiary'")
    ben_flagged = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT entity_id) FROM answer_key WHERE entity_type='vendor'")
    vendor_flagged = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT entity_id) FROM answer_key WHERE entity_type='registrar'")
    registrar_flagged = cur.fetchone()[0]
    print(f"  answer_key totals: {ben_flagged:,} beneficiary IDs (~1,250 new + 150 reused anchors expected), "
          f"{vendor_flagged} vendors, {registrar_flagged} registrar")


def main():
    parser = argparse.ArgumentParser(description="Chunk 4: inject fraud typologies")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", type=str, default="data/aid_program.db")
    args = parser.parse_args()

    random.seed(args.seed + 1)       # offset from Chunk 3's seed so fraud
    np.random.seed(args.seed + 1)    # draws aren't a mechanical repeat of it
    rng = random.Random(args.seed + 1)
    np_rng = np.random.default_rng(args.seed + 1)

    t0 = time.time()
    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM answer_key")
    if cur.fetchone()[0] != 0:
        raise SystemExit(
            "answer_key is not empty -- this DB has already had fraud injected, "
            "or wasn't freshly built by generate_population.py. Re-run Chunk 3 first."
        )

    counters = get_counters(conn)

    cur.execute("SELECT vendor_id, vendor_name, vendor_type, governorate, district, zone_x, zone_y FROM vendors")
    vendor_rows = cur.fetchall()
    vendor_by_district, all_vendor_ids = genpop.build_vendor_lookup(vendor_rows)

    cur.execute("SELECT registrar_id FROM registrars")
    registrar_ids = [r[0] for r in cur.fetchall()]
    assert REGISTRAR_COMPROMISED in registrar_ids

    # pick 2 colluding vendors (T1 ghost cash-out + T3) and 3 more ring-only
    # vendors (T5) -- all distinct
    chosen = rng.sample(all_vendor_ids, 2 + N_RING_ONLY_VENDORS)
    colluding_vendors = chosen[:2]
    ring_only_vendors = chosen[2:]
    print(f"Colluding vendors (T1/T3): {colluding_vendors}")
    print(f"Ring-only vendors (T5):    {ring_only_vendors}")

    all_beneficiary_rows, all_issuance_rows, all_txn_rows, all_answer_rows = [], [], [], []
    counts = {}

    print("\nInjecting Typology 1 (ghost beneficiaries)...")
    b, i, t, a = inject_ghosts(rng, np_rng, counters, registrar_ids, colluding_vendors, all_vendor_ids)
    all_beneficiary_rows += b; all_issuance_rows += i; all_txn_rows += t; all_answer_rows += a
    counts["ghosts"] = len(b)
    print(f"  {len(b)} ghost beneficiaries, {len(i)} issuances, {len(t)} transactions")

    print("Injecting Typology 2 (duplicate registrations)...")
    b, i, t, a = inject_duplicates(rng, np_rng, counters, conn, registrar_ids, vendor_by_district, all_vendor_ids)
    all_beneficiary_rows += b; all_issuance_rows += i; all_txn_rows += t; all_answer_rows += a
    counts["duplicates (extra IDs)"] = len(b)
    print(f"  {len(b)} extra duplicate IDs, {len(i)} issuances, {len(t)} transactions")

    print("Injecting Typology 2b (family eligibility gaming)...")
    b, i, t, a = inject_family_gaming(rng, np_rng, counters, registrar_ids, vendor_by_district, all_vendor_ids)
    all_beneficiary_rows += b; all_issuance_rows += i; all_txn_rows += t; all_answer_rows += a
    counts["family_gaming"] = len(b)
    print(f"  {len(b)} family-gaming registrants, {len(i)} issuances, {len(t)} transactions")

    print("Injecting Typology 5 (cash-out ring)...")
    b, i, t, a = inject_cashout_ring(rng, np_rng, counters, registrar_ids, colluding_vendors, ring_only_vendors)
    all_beneficiary_rows += b; all_issuance_rows += i; all_txn_rows += t; all_answer_rows += a
    counts["cashout_ring"] = len(b)
    print(f"  {len(b)} ring members, {len(i)} issuances, {len(t)} transactions")

    print("Injecting Typology 3 standalone (phantom-skim on the loud colluding vendor)...")
    # This must run while `conn` still reflects only Chunk 3's honest data --
    # it reads real unspent-balance remainders, which the fraud rows about
    # to be inserted below don't have (and shouldn't be skimmed from anyway).
    standalone_vendor = colluding_vendors[1]
    phantom_txns, phantom_stats = inject_vendor_phantom_skim(rng, conn, counters, standalone_vendor)
    all_txn_rows += phantom_txns
    counts["vendor phantom-skim (T3 standalone)"] = len(phantom_txns)
    print(f"  {len(phantom_txns)} phantom transactions fabricated at {standalone_vendor} "
          f"(victims NOT added to answer_key)")

    # Typology 3: vendor-level tag (colluding_vendors[0]'s signal emerges from
    # T1/T5 traffic above; colluding_vendors[1]'s from the phantom-skim above)
    for v in colluding_vendors:
        all_answer_rows.append(("vendor", v, "vendor_collusion"))
    # Typology 4: registrar-level tag (beneficiary-level rows added inline above)
    all_answer_rows.append(("registrar", REGISTRAR_COMPROMISED, "registrar_diversion"))

    print(f"\nInserting {len(all_beneficiary_rows):,} beneficiaries, {len(all_issuance_rows):,} issuances, "
          f"{len(all_txn_rows):,} transactions, {len(all_answer_rows):,} answer_key rows...")
    insert_batch(conn, "INSERT INTO beneficiaries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", all_beneficiary_rows)
    insert_batch(conn, "INSERT INTO issuances VALUES (?,?,?,?,?)", all_issuance_rows)
    insert_batch(conn, "INSERT INTO transactions VALUES (?,?,?,?,?)", all_txn_rows)
    insert_batch(conn, "INSERT INTO answer_key VALUES (?,?,?)", all_answer_rows)

    ok = validate(conn, len(all_beneficiary_rows))
    print_summary(conn, counts)
    print_vendor_fraud_stats(conn, colluding_vendors, phantom_stats)
    print_definition_of_done(conn, all_beneficiary_rows)
    conn.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s.")
    if not ok:
        raise SystemExit("Validation FAILED -- see checks above.")


if __name__ == "__main__":
    main()
