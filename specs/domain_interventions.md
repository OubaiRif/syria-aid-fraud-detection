# Domain Interventions Log

**Purpose:** This file tracks every point where Oubai overrode a default,
adjusted a parameter, or supplied Syria-specific domain knowledge that
changed how the synthetic dataset or detection rules were built. It exists
because this project's credibility depends on where its realism comes
from — a lot of it comes from firsthand knowledge that isn't in any spec
document, and that provenance should be visible, not buried in chat
history. This is a project deliverable: it gets cited in the final
README, not treated as scratch notes.

**Scope:** logged here are things *Oubai* corrected or supplied directly,
and rulings from Fable (or another design-authority session) that Oubai
relayed for implementation — both carry the same weight as standing
project decisions. Each entry's "Reasoning" field states its actual
source so provenance stays visible. Ordinary implementation judgment
calls I (Claude) made on my own, where the spec was silent and nobody has
weighed in yet, live in the Chunk-handoff notes instead — they graduate
to this file only once Oubai or Fable rules on them.

**Format for each entry:**
- **Default** — what I had implemented or was about to implement
- **Correction** — what it was changed to
- **Reasoning (Oubai's)** — the stated justification, as given
- **Downstream effect** — what this actually changes in the data or in
  how Chunk 5+ rules need to be designed/interpreted

---

## DI-001 — Distributed-spender transaction shape: daily cheap buys + occasional larger spend

**Session:** Chunk 3, post-generation review

**Default:** Each cycle's 2–6 transactions were split using normalized
random weights (`weights / weights.sum()`) — roughly even amounts across
all transactions in a cycle, no dominant outlier.

**Correction:** Split changed to a lumpy pattern: ~55% of cycles (when
≥3 transactions) get one dominant transaction taking 40–65% of the
cycle's total, with the rest small; the remaining ~45% of cycles use an
uneven-but-no-outlier split. The dominant transaction was (at this point)
routed preferentially to pharmacy/general vendors.

**Reasoning (Oubai's):** "a person living on aid in Syria, would most
likely go everyday or other day to the store. hunt for the cheapest
groceries available to cook for that day. large spending can be made to
pay for medicine, rent, and bills." — i.e. real spending is high-frequency
and small day-to-day, punctuated by occasional necessity-driven large
purchases, not evenly spread.

**Downstream effect:** Honest beneficiaries now organically produce some
large single-vendor transactions per cycle. This matters for Chunk 5:
any rule using "large transaction at a vendor" as a signal component
(e.g. R3/R4's amount-clustering checks) needs to be tested against this
honest lumpy baseline, not against an artificially smooth one — otherwise
precision estimates in Chunk 6 would be inflated by an unrealistically
easy honest population.

---

## DI-002 — Per-transaction amount floor and spend-lumpiness widened: food is cheap

**Session:** Chunk 3, post-generation review (immediately after DI-001)

**Default:** Minimum per-transaction floor set to $3.00 (chosen to avoid
$0.00-rounding artifacts from the DI-001 weight split); dominant-spend
share range 40–65%; no-dominant-item cycles used Dirichlet(α=0.6).

**Correction:** Floor lowered to $0.50. Dominant-spend share widened to
55–80%. No-dominant-item Dirichlet concentration lowered to α=0.4 (more
skew toward very small amounts).

**Reasoning (Oubai's):** "food is cheap in syria. you could buy a kilo of
tomatos for less than a dollar sometimes." — a $3 minimum on an ordinary
grocery trip was implicitly priced like a US grocery run, not a Syrian
one.

**Downstream effect:** Honest transaction amounts now legitimately range
from $0.50 up to full voucher value within the same cycle. This widens
the honest-population amount distribution considerably at the low end.
Relevant to Chunk 5: no current rule flags small transactions as
suspicious, but if one is added later (e.g. a "structuring" style rule
splitting large amounts into many small ones), it needs to be aware this
low range is now normal Syrian behavior, not evidence of anything.

---

## DI-003 — Reframed the "big" transaction: bulk staples, not medicine — and stopped using US price levels as the mental model

**Session:** Chunk 3, post-generation review (immediately after DI-002)

**Default:** The one larger transaction per cycle (see DI-001/002) was
framed as a medicine/necessity purchase and routed preferentially to
pharmacy/general vendors.

**Correction:** Reframed as a bulk staples stock-up (rice, flour, oil,
sugar — cheap per unit, but a month's quantity for a household still
adds up), routed preferentially to general/merchant-wholesale vendors
instead of pharmacy. I researched WHO/HAI medicine-price survey data from
comparable low/middle-income countries (India, Pakistan, Bangladesh) as a
substitute anchor, since I didn't have or find a reliable Syria-specific
price source — that survey data puts a full course of common antibiotics
at roughly $0.70–$5.50 in those markets, nowhere near large enough to be
the dominant monthly transaction.

**Reasoning (Oubai's):** "keep in mind medicine is cheaper that the USA.
I am saying that because it seems like you are using the united states
grocery, bills, medicine, rents, etc. as a comparison point. this is not
the case. if you don't have access to Syria specific prices, find a
similar third world country with a reliable stats and use it instead."

**Downstream effect:** Shifts which vendor type absorbs honest
bulk-spending volume — general/merchant-wholesale instead of pharmacy.
This matters directly for Chunk 5/6: Typology 2b (family eligibility
gaming) already routes its fraudulent bulk same-day redemptions to
merchant-wholesale vendors specifically, and this reframe means *honest*
big-ticket spending now lands at the same vendor type. That's actually
realistic (fraudulent bulk buyers hide among genuine bulk buyers at the
same stores) but it raises the bar for R9/R10 — those rules need to lean
on address/household clustering to separate fraud from honest bulk
shopping, since vendor type alone no longer discriminates as cleanly.

**Open flag:** this correction is still anchored on comparator-country
data (WHO/HAI methodology), not a Syria-specific source. If WFP Syria
market-monitoring price data becomes available, it should supersede this
entry rather than be treated as confirmed.

---

## DI-004 — Fable adjudication: precedent rulings confirmed, no code change

**Session:** Chunk 3–4 adjudication, relayed by Oubai from a separate Fable
session (`chunk4_adjudication.md`)

**Default:** Several Chunk 3/4 judgment calls were flagged as open
questions in the Chunk 4 handoff notes rather than confirmed decisions —
notably the T2 "2–3 IDs" reading and the dataset_schema_spec.md vs.
fraud_typology_spec.md precedence order.

**Correction:** Both confirmed as standing interpretations, not just
one-off guesses:
- T2 = 2–3 **extra** IDs per anchor (not 2–3 total). Confirmed by the
  spec's own arithmetic: 200 + 350 + 400 + 300 = 1,250 only resolves
  under this reading.
- `dataset_schema_spec.md` overrides `fraud_typology_spec.md` wherever
  they conflict — apply without asking in future conflicts, log each
  application here when it happens.

**Reasoning (source: Fable, relayed by Oubai):** "verified by the spec's
own total... only works under this reading" (T2); precedence rule stated
as standing policy for the project.

**Downstream effect:** No code change — this converts two of my earlier
"best guess, flag for review" judgment calls into settled precedent.
Future conflicts between the two spec docs should be resolved the same
way without re-litigating.

---

## DI-005 — ACTION 1: guarantee every district has a merchant-wholesale vendor

**Session:** Chunk 3–4 adjudication (`chunk4_adjudication.md`)

**Default:** Vendor type was assigned independently at random per vendor
(40% grocery / 30% general / 10% merchant-wholesale / 20% pharmacy), with
no constraint tying type distribution to district. Not flagged as a risk
at the time.

**Correction:** After generating all 120 vendors, any district with zero
merchant-wholesale vendors gets one vendor reassigned to that type. Added
as a post-generation fix pass in `generate_vendors()`, plus a matching
validation check.

**Reasoning (source: Fable, relayed by Oubai):** flagged as a real risk
before I'd checked for it: "otherwise T2b families in that district have
no resale channel and the typology silently disappears there." Checked
the live DB before making the change — confirmed **Harim district
genuinely had zero merchant-wholesale vendors** at seed 42, so this
wasn't a hypothetical.

**Downstream effect:** Every Typology 2b family now has a valid
merchant-wholesale resale channel regardless of which district it's
placed in. Re-ran Chunk 3 in full; at full scale (120 vendors / 10
districts) this actually triggered once (Harim, `VEN-104` reassigned from
grocery); at smaller test scales it triggered much more often (up to 5 of
10 districts), confirming this wasn't a rare edge case worth skipping.

---

## DI-006 — ACTION 2: Typology 3 standalone over-invoicing needed a real phantom-skim mechanism, not reshaped fraud-actor transactions

**Session:** Chunk 3–4 adjudication (`chunk4_adjudication.md`), resolving
the open flag from the original Chunk 4 handoff (item #15)

**Default:** T3's standalone vendor got its 80%-exact-value/15%-night-hour
bias applied to whichever ghost (T1) and cash-out-ring (T5) transactions
happened to land there. Measured result: only ~1.4–2.3% night share
vendor-wide, nowhere near the spec's implied 15% aggregate figure,
because those vendors also carry plenty of untouched organic honest
volume.

**Correction:** Split the two colluding vendors into distinct roles.
`colluding_vendors[0]` stays the "quiet" ghost-cash-out vendor, left
exactly as originally implemented. `colluding_vendors[1]` becomes a
"loud" standalone vendor built via a genuinely separate mechanism: ~1,500
honest beneficiaries' cycles were queried for real unspent voucher
remainder (`amount_issued - actual spent`, computed from the live DB, not
assumed), and 2,500 fabricated transactions were added there consuming
most of that remainder — 70% balance-emptying, 65% at night. These
victims are explicitly **not** added to `answer_key`; only the vendor
carries the `vendor_collusion` tag.

**Reasoning (source: Fable, relayed by Oubai):** "reshaping existing
fraud-routed transactions cannot produce T3, because T3 is its own
scheme — the vendor fabricates volume... The original '80% max-value'
applies only to the fraudulent subset, not the vendor aggregate." Revised
targets given explicitly: night-share 10–15% for the loud vendor
(vs. ~2% for the quiet one), balance-emptying/max-value share 25–35%
(vs. ~5–10% baseline) — with the explicit instruction to *measure* the
baseline rather than assume it.

**Downstream effect:**
- Night-share landed at **13.5%** for the loud vendor vs. **2.1%** for
  the quiet one vs. **0.0%** for the honest-vendor baseline — squarely
  inside the adjudicated targets.
- The 25–35%-vs-5–10% "balance-emptying share" target could **not** be
  hit with an amount-threshold DB proxy, and I flagged this rather than
  forcing a number: phantom-skim amounts are inherently small (they're
  draining a small leftover balance, typically under $20), and DI-002's
  cheap-food recalibration already means the honest population's own
  transaction amounts skew low (median ~$14–18). A $40 threshold gave
  28.0% vs. 28.8% baseline (no discrimination at all); a $20 threshold
  gave 61.4% vs. 56.6% (~5pp, still weak). Reported the true
  balance-emptying rate (70.6%) as generation-time ground truth instead,
  clearly labeled as not DB-inferable.
- **This is a real finding for Chunk 5/6, not just a metric that didn't
  work:** any detection rule relying on a fixed amount threshold to catch
  balance-emptying will face the same problem a real analyst would — cheap
  Syrian prices mean "small transaction" isn't inherently suspicious, so
  this typology needs remainder/pattern-level reasoning (e.g. cross-referencing
  against typical redemption completeness for that beneficiary) rather
  than a threshold rule. Per Fable's instruction, this asymmetry (one loud
  vendor R3/R4 will catch, one quiet vendor only catchable via ghost
  linkage) goes in the README as a stated hypothesis before Chunk 6
  results, not as a post-hoc explanation.

---

## Template for new entries

```
## DI-XXX — <short title>

**Session:** <Chunk N, context>

**Default:** <what I had implemented / was about to implement>

**Correction:** <what it was changed to>

**Reasoning (Oubai's):** "<quote or close paraphrase of the stated reasoning>"

**Downstream effect:** <what this changes in the data, and what Chunk 5+
rules or Chunk 6 evaluation need to account for as a result>

**Open flag (optional):** <anything still unresolved / needs a better source later>
```
