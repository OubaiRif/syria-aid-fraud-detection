"""
Build a small sample database for the public repo.

The full aid_program.db (~310MB, 49,928 beneficiaries, 1.7M transactions)
is deliberately excluded from version control -- it's fully reproducible
from the seed via generate_population.py + inject_fraud.py, and 300MB+
doesn't belong in a git repo. This script builds a small companion
sample_aid_program.db instead, keeping:

- ALL fraud beneficiaries (every typology fully represented, not just a
  random slice that might miss one)
- ALL members of both honest false-positive traps (phone-sharing families,
  split-household families) -- these are central to the project's story
  and a random sample could easily under-represent them
- A random sample of additional ordinary honest beneficiaries, for volume
  and realism
- ALL vendors and registrars (small tables, no need to subsample)
- issuances/transactions/answer_key filtered down to the sampled
  beneficiaries only

Usage:
    python src/build_sample_db.py --db data/aid_program.db \
        --out data/sample_aid_program.db --n-honest 3000 --seed 42
"""

import argparse
import random
import sqlite3


def get_fp_trap_membership(conn):
    cur = conn.cursor()
    # phone-sharing trap: phones shared by >=2 beneficiaries where NONE of
    # them are in answer_key (purely honest coincidence/trap, not fraud)
    cur.execute("""
        SELECT b.beneficiary_id FROM beneficiaries b
        WHERE b.phone IN (
            SELECT phone FROM beneficiaries GROUP BY phone HAVING COUNT(*) >= 2
        )
        AND b.beneficiary_id NOT IN (SELECT entity_id FROM answer_key WHERE entity_type='beneficiary')
    """)
    phone_trap = {r[0] for r in cur.fetchall()}

    # split-household trap: uses the ADDR-FAM-XXXX synthetic address prefix
    # assigned in generate_population.py's apply_fp_traps()
    cur.execute("""
        SELECT beneficiary_id FROM beneficiaries WHERE address_id LIKE 'ADDR-FAM-%'
    """)
    split_trap = {r[0] for r in cur.fetchall()}

    return phone_trap, split_trap


def main():
    parser = argparse.ArgumentParser(description="Build a small sample DB for the public repo")
    parser.add_argument("--db", type=str, default="data/aid_program.db")
    parser.add_argument("--out", type=str, default="data/sample_aid_program.db")
    parser.add_argument("--n-honest", type=int, default=3000,
                         help="additional random honest beneficiaries beyond fraud + FP traps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    src = sqlite3.connect(args.db)
    cur = src.cursor()

    cur.execute("SELECT entity_id FROM answer_key WHERE entity_type='beneficiary'")
    fraud_ids = {r[0] for r in cur.fetchall()}
    phone_trap, split_trap = get_fp_trap_membership(src)

    cur.execute("SELECT beneficiary_id FROM beneficiaries")
    all_ids = [r[0] for r in cur.fetchall()]
    already = fraud_ids | phone_trap | split_trap
    remaining_pool = [i for i in all_ids if i not in already]
    extra_honest = set(rng.sample(remaining_pool, min(args.n_honest, len(remaining_pool))))

    keep_ids = fraud_ids | phone_trap | split_trap | extra_honest
    print(f"Sample composition: {len(fraud_ids):,} fraud, {len(phone_trap):,} phone-trap, "
          f"{len(split_trap):,} split-trap, {len(extra_honest):,} extra honest "
          f"= {len(keep_ids):,} total beneficiaries "
          f"({100*len(keep_ids)/len(all_ids):.1f}% of full population)")

    dst = sqlite3.connect(args.out)
    src_cur = src.cursor()
    dst_cur = dst.cursor()

    # copy schema
    src_cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
    for (sql,) in src_cur.fetchall():
        dst_cur.execute(sql)
    dst.commit()

    placeholders = ",".join("?" * len(keep_ids))
    keep_list = list(keep_ids)

    print("Copying registrars, vendors (kept in full)...")
    for table in ("registrars", "vendors"):
        src_cur.execute(f"SELECT * FROM {table}")
        rows = src_cur.fetchall()
        dst_cur.executemany(f"INSERT INTO {table} VALUES ({','.join('?'*len(rows[0]))})", rows)
    dst.commit()

    print("Copying beneficiaries (sampled)...")
    src_cur.execute(f"SELECT * FROM beneficiaries WHERE beneficiary_id IN ({placeholders})", keep_list)
    rows = src_cur.fetchall()
    dst_cur.executemany(f"INSERT INTO beneficiaries VALUES ({','.join('?'*len(rows[0]))})", rows)
    dst.commit()
    print(f"  {len(rows):,} beneficiaries")

    print("Copying issuances (sampled beneficiaries only)...")
    src_cur.execute(f"SELECT * FROM issuances WHERE beneficiary_id IN ({placeholders})", keep_list)
    rows = src_cur.fetchall()
    dst_cur.executemany(f"INSERT INTO issuances VALUES ({','.join('?'*len(rows[0]))})", rows)
    dst.commit()
    print(f"  {len(rows):,} issuances")

    print("Copying transactions (sampled beneficiaries only)...")
    src_cur.execute(f"SELECT * FROM transactions WHERE beneficiary_id IN ({placeholders})", keep_list)
    rows = src_cur.fetchall()
    dst_cur.executemany(f"INSERT INTO transactions VALUES ({','.join('?'*len(rows[0]))})", rows)
    dst.commit()
    print(f"  {len(rows):,} transactions")

    print("Copying answer_key (all rows for sampled beneficiaries, plus vendor/registrar rows)...")
    src_cur.execute(f"""
        SELECT * FROM answer_key
        WHERE (entity_type='beneficiary' AND entity_id IN ({placeholders}))
           OR entity_type IN ('vendor', 'registrar')
    """, keep_list)
    rows = src_cur.fetchall()
    dst_cur.executemany("INSERT INTO answer_key VALUES (?,?,?)", rows)
    dst.commit()
    print(f"  {len(rows):,} answer_key rows")

    # NOTE: rule_flags is deliberately NOT copied into the sample DB -- it's
    # a derived output already captured in data/rule_evaluation.csv and
    # data/ml_comparison.csv, and copying it here would need re-deriving
    # from the sampled subset to stay consistent (the real rule_flags
    # table was computed against the FULL population, so a naive copy
    # would silently misrepresent rule performance on this smaller sample).

    src.close()
    dst.close()

    import os
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"\nWrote {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
