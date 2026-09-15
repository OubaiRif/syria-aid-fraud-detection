# Dataset Schema & Injection Parameters
**Project:** Humanitarian Cash-Transfer Fraud Detection (Syria context)
**Companion to:** `fraud_typology_spec.md` (source of truth for typologies)
**Audience:** This file + the typology spec are sufficient for implementation without further design input. Stack: Python, pandas, numpy, SQLite. Set `numpy.random.seed(42)` for reproducibility.

---

## Resolved design decisions
1. **Names:** Generate with an Arabic transliteration-variant system (see §Names). This is the portfolio differentiator — do not skip.
2. **Geography:** Real Syrian governorate/district names (credibility) + synthetic zone coordinates (simple distance math). Program modeled on a NW-Syria cross-border operation: Idlib and Aleppo governorates, ~10 districts.
3. **Answer key:** Fraud labels live in a separate `answer_key` table, never as columns on main tables — prevents accidental leakage into features and mirrors real life (main tables look exactly like what an analyst would receive).
4. **No `transaction_type` field.** Real redemption data doesn't label "cash-out" vs "goods" — inferring it is part of the challenge.

---

## Tables (SQLite: `aid_program.db`)

### beneficiaries
| field | type | notes |
|---|---|---|
| beneficiary_id | TEXT PK | e.g. `BEN-000001` |
| full_name | TEXT | with transliteration variants (see §Names) |
| family_name | TEXT | extracted surname, used for 2b clustering |
| phone | TEXT | Syrian mobile format `09XXXXXXXX` |
| address_id | TEXT | FK-like; shared address_id = same dwelling |
| governorate | TEXT | Idlib / Aleppo |
| district | TEXT | one of ~10 |
| zone_x, zone_y | REAL | coordinates within district grid (0–100) |
| claimed_household_size | INTEGER | 1–12 |
| registration_date | DATE | program window: 2025-01 → 2025-03 |
| registrar_id | TEXT | FK → registrars |

### vendors
| field | type | notes |
|---|---|---|
| vendor_id | TEXT PK | `VEN-001` … |
| vendor_name | TEXT | |
| vendor_type | TEXT | grocery / general / merchant-wholesale / pharmacy |
| governorate, district, zone_x, zone_y | | same geography model |

### registrars
| field | type | notes |
|---|---|---|
| registrar_id | TEXT PK | `REG-01` … `REG-12` |
| office_district | TEXT | |

### issuances (monthly voucher cycles)
| field | type | notes |
|---|---|---|
| issuance_id | TEXT PK | |
| beneficiary_id | TEXT | |
| cycle_month | TEXT | `2025-04` … `2026-03` (12 cycles) |
| amount_issued | REAL | base 50 USD-equivalent × household factor |
| issue_timestamp | DATETIME | distribution day, morning hours |

### transactions
| field | type | notes |
|---|---|---|
| txn_id | TEXT PK | |
| beneficiary_id | TEXT | |
| vendor_id | TEXT | |
| amount | REAL | |
| timestamp | DATETIME | |

### answer_key (NEVER joined during detection; evaluation only)
| field | type | notes |
|---|---|---|
| entity_type | TEXT | beneficiary / vendor / registrar |
| entity_id | TEXT | |
| fraud_type | TEXT | ghost / duplicate / family_gaming / vendor_collusion / registrar_diversion / cashout_ring |

---

## §Names — Arabic transliteration variant system
Build two Python dicts:
- `FIRST_NAMES`: ~40 common names, each mapping to 2–4 Latin variants, e.g. `{"محمد": ["Mohammed","Muhammad","Mohamad","Mhd"], "حسين": ["Hussein","Husayn","Hussain"], "عبد الرحمن": ["Abdulrahman","Abd al-Rahman","Abdel Rahman"], ...}` (include female names: Fatima/Fatimah, Aisha/Aysha/Aicha, Maryam/Mariam …)
- `FAMILY_NAMES`: ~60 surnames with variants, e.g. `{"الخالدي": ["Al-Khalidi","Alkhalidy","Elkhalidi"], ...}`

Honest beneficiaries: one variant chosen once, used consistently.
Typology 2 duplicates: same underlying name, *different* variant per duplicate identity.
Also implement `normalize_name()` (lowercase, strip hyphens/spaces/"al-"/"el-", collapse doubled letters) — detection rules use it; its imperfection is realistic.

---

## Population & injection parameters (final)
| parameter | value |
|---|---|
| Honest beneficiaries | 48,600 |
| — distributed spenders | 60% |
| — legitimate fast cash-outers | 40% (full redemption <6h at usual local vendor, uncoordinated) |
| Vendors | 120 (2 colluding) |
| Registrars | 12 (1 compromised: REG-07) |
| Voucher cycles | 12 months |
| Ghost beneficiaries (T1) | 200 IDs / ~40 phones / 6 registration bursts / 60% registered by REG-07 / ≥90% redemptions at colluding vendors |
| Duplicate identities (T2) | 150 real people × 2–3 IDs = ~350 extra IDs; 70% share phone or address, 30% name-variant-only |
| Family gaming (T2b) | 120 families, 3–5 registrants each (~400 IDs), same address_id + family_name, identical claimed_household_size, ≥70% bulk same-day redemptions at merchant-wholesale vendors |
| Cash-out ring (T5) | 300 beneficiaries, coordinated same-vendor cash-out within same hour on distribution day, 50% at colluding vendors |
| False-positive traps | ~500 honest extended families sharing one phone; ~200 large legitimate families with multiple registrants whose claimed household sizes are *consistent splits* (sum ≈ real residents) |
| Total fraudulent IDs | ~1,250 ≈ 2.5% of ~50,200 |

Overlap rule: an entity may carry multiple fraud_type rows in answer_key (e.g. ghost + registrar_diversion).

---

## Normal-behavior generation rules
- Distributed spender, per cycle: 2–6 transactions over 3–10 days, amounts summing to 85–100% of issuance, 1–3 vendors within ~15 zone-units of home, daytime hours (biased 9:00–18:00)
- Fast cash-outer, per cycle: 1–2 transactions within 6h of issuance, ≥95% of value, usual local vendor, daytime
- Noise: 3% of honest beneficiaries miss a random month; small amount jitter; occasional distant-vendor trip (market day, 2% of txns)
- Colluding vendors: 15% of their transactions timestamped 00:00–05:00; 80% of amounts = exact issuance value

---

## Implementation pipeline (Chunks 3–4, Sonnet)
Repo layout:
```
syria-aid-fraud-detection/
├── README.md
├── specs/  (both spec files)
├── src/
│   ├── generate_population.py   # Chunk 3: honest majority → SQLite
│   ├── inject_fraud.py          # Chunk 4: typologies + answer_key
│   ├── rules.py                 # Chunk 5: R1–R10 flag functions
│   ├── evaluate.py              # Chunk 6: precision/recall per rule & combined
│   └── model.py                 # Chunk 7: IsolationForest + XGBoost comparison
├── data/aid_program.db          # gitignored; small demo DB committed
└── notebooks/analysis.ipynb     # findings walkthrough
```
Order: Chunk 3 must run standalone and validate (row counts, no duplicate PKs, honest-only DB). Chunk 4 modifies/extends the DB and writes answer_key. Keep each script runnable via `python src/<file>.py` with a `--seed` arg.

**Definition of done per chunk:** script runs clean, prints summary stats, DB state verified. Then stop — next chunk is a fresh session.

**Status:** Chunk 2 complete. Design phase over. Chunks 3–7: new Sonnet chat, attach both spec files, instruct: "Implement Chunk N per these specs."
