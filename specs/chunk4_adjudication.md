# Chunk 3–4 Adjudication (Fable rulings)
**Hand this file to Sonnet. Items marked ACTION require code changes before Chunk 5. Everything else is APPROVED as implemented.**

## Approved as implemented (no changes)
- **#1** Transaction-size shape (dominant-transaction mechanic): approved — matches real spending patterns.
- **#2** $0.50 floor: approved. Synthetic data doesn't need Syria-exact prices; plausibility is sufficient. Do not spend time sourcing WFP price data.
- **#3** Bulk staples framing + merchant/general routing: approved. The comparator-country caveat can be one honest sentence in the README ("prices calibrated to comparable low-income markets, not Syria-specific") — that's a strength, not a weakness.
- **#4** District list: approved. These are in fact the real admin2 districts where NW-Syria cross-border programs concentrated (Idlib + northern Aleppo). No changes.
- **#6** Uniform registrar assignment: approved and *intended* — R8 must detect fraud-driven elevation, not small-denominator artifacts.
- **#7** Fixed distribution day (5th): approved. Note for Chunk 5: since ~40% of honest beneficiaries legitimately cash out fast on the same day, the ring signal is specifically same-vendor + same-hour concentration, never same-day.
- **#10** Exactly 2 colluding vendor IDs shared across typologies: correct reading.
- **#11** Sub-ring mechanic (5 groups, fixed hour each): approved — good design; coordination must be constructed, it doesn't emerge from randomness.
- **#12** 25% REG-07 duplicate share: approved.
- **#13** Claimed household size 6–10: approved, provided sum-of-claims per family exceeds plausible residents.
- **#14** Fraud-actor geography independent of fraud vendor (emergent R7 signal): approved — this is exactly the intended mechanism.

## Confirmed interpretations (record as precedent)
- **#8** T2 = 2–3 **extra** IDs per anchor (~350 extras). CONFIRMED — verified by the spec's own total: 200 + 350 + 400 + 300 = 1,250 flagged IDs only works under this reading. Anchors themselves also get a `duplicate` row in answer_key (the real person committing the fraud is fraudulent too).
- **#9** Precedence rule, standing: **dataset_schema_spec.md overrides fraud_typology_spec.md** wherever they conflict. Apply without asking in future conflicts, but log each application in the handoff notes.

## ACTION 1 (small): vendor type mix — #5
Distribution approved with one constraint: **every district must contain ≥1 merchant-wholesale vendor**, otherwise T2b families in that district have no resale channel and the typology silently disappears there. Verify; reassign vendor types if any district lacks one. (Also: 20% pharmacy is high for reality but harmless to the analysis — leave it.)

## ACTION 2 (the real fix): Typology 3 standalone over-invoicing — #15
Sonnet's diagnosis is correct: reshaping existing fraud-routed transactions cannot produce T3, because T3 is its own scheme — the vendor fabricates volume. The spec's 15%/80% figures were written as vendor-aggregate targets but are unachievable (and unrealistic) without fabricated transactions. Resolution:

**Implement a phantom-skim mechanism on the standalone colluding vendor (the one NOT shared with T1):**
1. Select ~1,500–2,500 honest beneficiaries across cycles who left unspent balance (distributed spenders sum to 85–100%; the 0–15% remainder is the skim surface).
2. Fabricate transactions at this vendor consuming most of that remainder. These beneficiaries never chose this vendor — they are victims, not perpetrators. **Do not add them to answer_key.** Only the vendor carries the `vendor_collusion` label. (Realism: vendor terminal processes phantom redemptions against stale voucher balances.)
3. Timestamp ~60–70% of phantom transactions in 00:00–05:00; make ~70% of them exact-remainder (i.e., balance-emptying) amounts.
4. **Revised aggregate targets for this vendor:** night-transaction share **10–15%** of its total volume; balance-emptying/max-value share elevated to roughly **25–35%** vs. a peer baseline you should measure (expect ~5–10%). Total volume share also rises → feeds R4. The original "80% max-value" applies only to the *fraudulent subset*, not the vendor aggregate — treat this file as overriding the schema doc on that number.
5. **Leave the ghost cash-out vendor (shared with T1) exactly as-is** — its diluted ~2% night share is intentional realism. Result: one *loud* colluding vendor and one *quiet* one. Chunk 5 rules will catch the loud one with R3/R4; the quiet one should only fall via its ghost-beneficiary linkage (R1/R5 → vendor association). Write this asymmetry up in the README — it demonstrates why vendor-aggregate rules miss embedded fraud, which is a professional-grade finding.

## Consequence for Chunk 5 (rule tuning)
- R3/R4 thresholds must be **relative** (X σ above the vendor-population mean), never fixed percentages — the spec already implied this for R4; apply to R3 as well.
- Add expected outcome to evaluation notes: R3/R4 catch the loud vendor, miss the quiet one; naive R6 collapses on precision (cash economy); R9/R10 require family-level aggregation. These predictions go in the README *before* results — stating hypotheses first reads like real analytical work.

## Definition of done for the revision
Re-run Chunk 4, then verify and print: per-vendor night-share table (loud vendor 10–15%, quiet ~2%, honest ~0–1%), balance-emptying share per vendor, every district has ≥1 merchant-wholesale vendor, answer_key totals unchanged at ~1,250 beneficiary IDs + 2 vendors + 1 registrar. Then stop; Chunk 5 is a fresh session.
