# Fraud Typology → Data Signal Specification
**Project:** Humanitarian Cash-Transfer Fraud Detection (Syria context)
**Purpose:** Translate documented aid-fraud typologies into measurable data fingerprints. This spec drives the synthetic dataset design (Chunk 2) and the detection rules (Chunk 5).

---

## How to read this document
Each typology has four parts:
- **Scheme** — how the fraud works in the real world
- **Real-world basis** — where it is documented (cite in final README)
- **Data fingerprint** — the traces it leaves in beneficiary/transaction data
- **Injection plan** — how we will plant it in synthetic data (parameters to finalize in Chunk 2)

Target overall fraud rate: **~2–3% of beneficiaries**, matching realistic detection-problem base rates (fraud is rare; that is what makes precision hard).

---

## Typology 1: Ghost Beneficiaries
**Scheme:** Fictitious people are registered by insiders (staff or local partners). Vouchers issued to these "people" are collected and cashed out by the fraudster, often through a cooperating vendor.
**Real-world basis:** USAID OIG investigations into Syria cross-border programs (2016–2018 procurement/beneficiary fraud cases); Syria Humanitarian Fund risk reports.

**Data fingerprint:**
- Multiple beneficiary IDs sharing one **phone number** (fraudster's contact)
- Multiple IDs sharing one **address** or implausibly clustered locations
- **Registration bursts** — many ghost IDs created on the same day, by the same registrar
- Redemptions concentrated at **one or two vendors** (the colluding cash-out point)
- Unnaturally **regular behavior** — full voucher redeemed same day each month, no household-like variation

**Injection plan:** ~200 ghost IDs sharing ~40 phone numbers, created in 5–8 registration bursts, redeeming ≥90% at 2 colluding vendors.

---

## Typology 2: Duplicate Registration
**Scheme:** A real person registers multiple times under name variants (transliteration differences, swapped name order) to collect multiple entitlements.
**Real-world basis:** Documented in Syria and regional refugee response deduplication exercises; a known driver of biometric registration adoption (UNHCR).

**Data fingerprint:**
- **Fuzzy-similar names** across IDs (Mohammed / Muhammad / Mhd)
- Shared or near-identical **phone / address / household composition**
- Redemption patterns that are **temporally exclusive** — the "two people" never transact at the same time in different places (one body, one place)

**Injection plan:** ~150 real beneficiaries each holding 2–3 duplicate identities with name variants; duplicates share phone or address in ~70% of cases (the rest are harder: detectable only by name similarity + behavior).

---

## Typology 2b: Intra-Family Eligibility Gaming *(firsthand observation)*
**Scheme:** Every unmarried adult in one family registers separately as an eligible recipient, and all claim the same (full) household size — so one family collects multiple full entitlements. Collected aid is then **sold in bulk to merchants for resale** rather than consumed.
**Real-world basis:** Directly observed in Syria aid operations (author's field experience). Distinct from Typology 2: these are *real people* with valid identities gaming eligibility rules, not fake identities.

**Data fingerprint:**
- Clusters of registrations at the **same address / same family name** where every member claims an **identical household size** (arithmetically impossible — claimed totals exceed actual residents)
- Individually each registration looks clean; the signal only appears when **aggregating by address or family**
- Redemptions in **bulk, same-day, at merchant-type vendors** (resale pattern) rather than distributed household consumption
- Aid types redeemed **inconsistent with claimed household** (e.g., multiple registrants at one address all drawing full family-size rations)

**Injection plan:** ~120 families (roughly 400 registrants) with 3–5 same-address registrations each, identical claimed household sizes, and ≥70% of redemptions as bulk same-day transactions at a small set of merchant vendors. This typology is the project's key lesson in **entity resolution** — fraud invisible at record level, visible at family level.

---

## Typology 3: Vendor Collusion / Over-Invoicing
**Scheme:** A participating vendor inflates transactions, processes fake redemptions, or splits cash-outs with beneficiaries instead of providing goods.
**Real-world basis:** SHF and WFP Syria monitoring reports on vendor-level diversion; USAID OIG vendor kickback cases.

**Data fingerprint:**
- One vendor with a **disproportionate share** of total transaction volume vs. its size/location
- Transactions at **implausible hours** (00:00–05:00 for a food shop)
- Amounts clustered at the **exact voucher maximum** or suspicious round numbers
- Beneficiaries traveling **implausible distances** to reach this vendor while passing nearer ones

**Injection plan:** 2 colluding vendors (of ~60 total). One overlaps with Typology 1 (the ghost cash-out vendor); one runs standalone over-invoicing with ~15% night transactions and 80% max-value redemptions.

---

## Typology 4: Registrar-Level Diversion
**Scheme:** A specific staff member/registrar systematically creates fraudulent entries or manipulates entitlements. Fraud clusters by *who did the registration*, not just who is registered.
**Real-world basis:** Post-2025 HTS-transition governance gaps; partner-staff fraud cases in USAID OIG Syria archives.

**Data fingerprint:**
- One registrar's caseload has an **elevated rate** of every other typology
- Registrations by this registrar show **template-like uniformity** (same household size, sequential-looking data)
- Detectable only by **aggregating flags per registrar** — this is the typology that rewards analysis one level up

**Injection plan:** ~12 registrars total; 1 compromised registrar responsible for ~60% of ghost registrations and elevated duplicate rates.

---

## Typology 5: Cash-Out Velocity Anomaly — *revised for cash-economy context*
**Scheme:** Vouchers/entitlements are converted to cash or resold at a discount via colluding vendors. Indicates trafficking of entitlements or coerced beneficiaries.

**CRITICAL DOMAIN CONSTRAINT (firsthand observation):** Syria has no Visa/Mastercard rails; the economy runs on cash. Even salaried people with bank accounts withdraw most or all funds immediately after deposit. Therefore **fast, full cash-out is NORMAL Syrian behavior, not a fraud signal by itself.** A velocity rule imported from Western fintech would flag a large share of honest beneficiaries. This is a headline finding of the project: **fraud rules must be calibrated to local financial behavior.**

**Revised data fingerprint (what still distinguishes fraud in a cash economy):**
- Not speed alone, but speed **combined with**: redemption at a *specific colluding vendor* (not the beneficiary's usual/nearest one), discounted implied value, or synchronization with other flagged accounts
- **Coordinated timing** — many unrelated beneficiaries cashing out at the same vendor within the same hour (trafficking ring), vs. organic fast cash-out which is *individually* fast but *collectively* uncoordinated
- Cross-reference with Typology 2b resale patterns

**Injection plan:** ~300 beneficiaries with coordinated same-vendor cash-out, ~50% overlapping with colluding vendors from Typology 3. **The honest majority must include a large share (≥40%) of legitimate fast full cash-outs** so that naive velocity rules demonstrably fail on precision — this is deliberate and pedagogically central.

---

## The honest majority (defines "normal") — *cash-economy calibrated*
~50,000 beneficiaries split into two legitimate behavior modes:
- **Distributed spenders (~60%):** redeem monthly voucher over **3–10 days** in **2–6 transactions** at 1–3 vendors near their registered location
- **Fast cash-outers (~40%):** full or near-full redemption within hours of issuance — **legitimate and common in Syria's cash economy** (no card rails; even bank-account holders withdraw deposits immediately). Individually indistinguishable from fraud velocity; distinguishable because they are *uncoordinated* and use their *usual local vendor*
- Both modes: household-size-correlated amounts, natural noise (occasional missed months), unique phones/addresses with rare legitimate exceptions (extended families sharing one phone; large legitimate families with multiple *eligible* registrants who claim *consistent, non-overlapping* household splits — a deliberate **false-positive trap** for Typology 2b rules)

---

## Out of scope for the transaction dataset (documented context / possible Phase 2)
Two firsthand-observed typologies operate at the **procurement/project level**, not the beneficiary-transaction level, so they don't fit this dataset — but they belong in the README as documented context and could become a Phase 2 procurement dataset:
- **Budget-matching price inflation:** low-cost projects priced up to consume the full grant; the gap split between project managers and vendors. Fingerprint (Phase 2): cost-per-output outliers vs. comparable projects, quotes clustering just under approval thresholds, repeated vendor–manager pairings.
- **Inflated asset purchases + delayed write-off:** low-value equipment bought at inflated prices, then tagged as waste in an inventory audit 1–2 years later, closing the paper trail. Fingerprint (Phase 2): asset write-off rates by procurement officer, price-to-market-benchmark gaps, time-to-write-off distributions.

---

## Detection rule candidates (preview for Chunk 5)
| # | Rule | Targets typology |
|---|------|------------------|
| R1 | Phone number on ≥3 registrations | 1, 2 |
| R2 | Name similarity ≥ threshold + shared attribute | 2 |
| R3 | Vendor night-transaction share > X% | 3 |
| R4 | Vendor share of total volume > X σ above mean | 3 |
| R5 | Registration burst: >N registrations/registrar/day | 1, 4 |
| R6 | ~~Time-to-first-redemption < 1h AND full value~~ **REVISED:** fast full cash-out AND (non-local vendor OR coordinated timing with ≥N others at same vendor within 1h) | 5 |
| R6-naive | Naive velocity rule (speed alone) — **kept deliberately to demonstrate its precision collapse in a cash economy** | 5 (fails) |
| R7 | Beneficiary-to-vendor distance > X km past nearer vendors | 3 |
| R8 | Registrar aggregate flag rate > X σ above mean | 4 |
| R9 | Same address/family name: ≥3 registrations with identical claimed household size, sum of claims > plausible residents | 2b |
| R10 | Bulk same-day redemption at merchant-type vendor by same-address cluster | 2b |

---

## Open decisions for Chunk 2 (schema session)
1. Exact table schemas and field list (beneficiaries, transactions, vendors, registrars)
2. Final base rates and overlap percentages between typologies
3. Whether names are generated with an Arabic-transliteration variant library (recommended — it's the portfolio differentiator)
4. Geography model: real Syrian governorate/district names vs. abstract zones

**Status:** Chunk 1 complete. Next session: Chunk 2 (schema + injection parameters) — short, with Fable. Chunks 3–7 (implementation) — with Sonnet, using this spec as the source of truth.
