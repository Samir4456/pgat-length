# Compositional Generalization in Gloss-Free Sign Language Translation

> A thesis research program: controlled evaluation of whether gloss-free
> SLT models can translate novel combinations of concepts and relations
> that they have already seen individually — and a targeted training
> intervention that addresses the failure.

This document is the single reference for the compositional-generalization
direction chosen after three architectural interventions (Source32,
pgat-length v2 with mBART, pgat-length + Qwen QLoRA) all failed to improve
translation quality despite affecting different parts of the system. Read
top-to-bottom the first time. Section 18 contains the defense Q&A — every
likely committee question with a written answer.

## Contents

1. [The problem: what the length cliff really is](#1-the-problem)
2. [What compositional generalization means](#2-what-compositional-generalization-means)
3. [Why this framing fits our specific failure signature](#3-why-this-framing-fits-our-specific-failure-signature)
4. [The hypothesis, stated formally](#4-the-hypothesis)
5. [Notation](#5-notation)
6. [Semantic schema for PHOENIX14T](#6-semantic-schema-for-phoenix14t)
7. [The rule-based extractor and its validation](#7-the-rule-based-extractor-and-its-validation)
8. [Compositional split construction](#8-compositional-split-construction)
9. [The 2×2 evaluation matrix](#9-the-2x2-evaluation-matrix)
10. [Failure attribution: three complementary tasks](#10-failure-attribution)
11. [The intervention: relation-focused contrastive training](#11-the-intervention)
12. [Baselines and controls](#12-baselines-and-controls)
13. [Evaluation metrics in full](#13-evaluation-metrics-in-full)
14. [Statistical protocol](#14-statistical-protocol)
15. [Feasibility gate](#15-feasibility-gate)
16. [Timeline and decision gates](#16-timeline-and-decision-gates)
17. [Risks and mitigations](#17-risks-and-mitigations)
18. [Defense Q&A](#18-defense-qa)
19. [Related work map](#19-related-work-map)
20. [Publication strategy](#20-publication-strategy)
21. [Code architecture and reuse](#21-code-architecture-and-reuse)
22. [References](#22-references)

---

## 1. The problem

### 1.1 The observed length cliff

Sign language translation systems on PHOENIX14T show a sharp drop in
translation quality once references exceed about 12 whitespace-tokens.
The pattern is documented across multiple architectures (Yazdani et al.
2026, Hamidullah et al. 2024) and reproduced in this project's own
PGAT-v1 TEST five-bin measurement:

| Reference length (words) | Samples | BLEU-4 | chrF |
|---|---:|---:|---:|
| 1–6 | 42 | 16.88 | 41.09 |
| 7–12 | 286 | 17.39 | 35.74 |
| 13–18 | 220 | **5.03** | 25.69 |
| 19–24 | 78 | 4.63 | 24.28 |
| 25–31 | 16 | 3.98 | 25.49 |

The drop from bin 7–12 to bin 13–18 is a factor of ~3.5× in BLEU-4 and is
not explained by simple label-space growth alone.

### 1.2 What we already know it *isn't*

Three architectural interventions have been run on our stack, each
targeting a different mechanism the length cliff might reflect:

| Intervention | What it tests | Result |
|---|---|---|
| Source32 | "The encoder doesn't see enough" (denser DINO evidence, wider prefix) | Retrieval R@1 improved 14.8 → 17.9; translation BLEU-4 did **not** improve. Falsifies "more visual evidence" |
| pgat-length v2 (variable prefix, mBART) | "The prefix is too short / fixed" (prefix length grows with input duration) | chrF flattened across bins; overall BLEU-4 dropped 9.99 → 7.20. Falsifies "prefix length is the bottleneck" |
| pgat-length + Qwen QLoRA (variable prefix, Qwen decoder) | "The mBART decoder was underpowered" (bring back PGAT-v1's decoder recipe) | Bin 1–6 BLEU-4 jumped to 18.46; long bins collapsed to 1.8–2.1; mode collapse in predictions ("dabei bleibt es meist trocken" for many videos). Falsifies "decoder capacity is the bottleneck" |

### 1.3 The consistent failure signature

Manual inspection of the Qwen predictions on DEV shows the same behaviour
across systems: **fluent, domain-plausible German that only weakly
depends on the video**. Two different videos produced identical
predictions. Temperature values and place names were fabricated. The
generation looks like an unconditional PHOENIX-domain language model
lightly steered by a coarse visual signal.

This is not a decoder-capacity failure. This is not an evidence-density
failure. This is a *semantic-consumption* failure: the model produces
sentences that pattern-match training data instead of composing atoms
correctly from the video.

This document is a plan to test that framing directly.

---

## 2. What compositional generalization means

### 2.1 The concept in one sentence

Compositional generalization is the ability to correctly handle a *new
combination* of elements the model has already seen individually.

### 2.2 A minimal example

Suppose the training set contains:

- "rain in the north"
- "the sun shines in the south"
- "snow in the west"

The model has seen the events *rain, sun, snow*, the locations *north,
south, west*, and the pattern "X in Y". Test the model on

- "the sun shines in the west"

Every atom (event, location, structure) is familiar. Only the
*combination* is new. A model that has genuinely learned compositional
structure — that treats *sun* and *west* as independently reusable
pieces — handles this. A model that has memorized surface patterns fails.

### 2.3 Two levels of compositional generalization

- **Lexical composition (weak):** a new sentence built from words all
  seen individually. This is the case above.
- **Structural composition (strong):** a new *structural pattern* built
  from familiar building blocks. In sign language, an example is a
  novel binding between two propositions: given training that binds
  temperatures to single regions, does the model correctly bind two
  temperatures to two regions in a compound sentence?

  Training: "10 degrees in the north tomorrow"
  Training: "20 degrees in Bavaria tomorrow"
  Test: "10 degrees in the north and 20 degrees in Bavaria tomorrow"

  The model must correctly assign 10 to *north* and 20 to *Bavaria*.
  A bag-of-atoms model that produces {10, 20, north, Bavaria} in some
  order can look "close" on BLEU but is compositionally wrong.

### 2.4 Compositional generalization vs length generalization

These often correlate but are conceptually different:

| Length generalization | Compositional generalization |
|---|---|
| Train on short, test on longer | Train on known combos, test on new combos |
| Failure mode: model doesn't have enough "capacity" for longer outputs | Failure mode: model treats sentences as atomic patterns rather than compositions |
| Fix: length curriculum, position extrapolation | Fix: training data structure, contrastive supervision, better representations |

A rigorous compositional study *controls for* length so that any
observed gap is not just length in disguise. This is one of the two axes
of our 2×2 evaluation matrix (§9).

### 2.5 The prior-art tradition

The compositional-generalization tradition in ML is well-established.
The canonical benchmarks and findings:

| Paper | Contribution | Finding |
|---|---|---|
| Lake & Baroni, *SCAN* (ICML 2018) | Synthetic command dataset with primitive-vs-compositional splits | Standard seq2seq models fail on compositional splits |
| Keysers et al., *CFQ* (ICLR 2020) | Atom-compound divergence method for constructing splits over natural data | Multiple architectures show sharp compositional drops |
| Kim & Linzen, *COGS* (EMNLP 2020) | Compositional semantic parsing challenge | Reveals which grammatical operations are hard |
| Li et al., *CoGnition* (ACL 2021) | Compositional benchmark for machine translation | **Same methodology, applied to NMT** — our closest anchor |
| Shaw et al. (ACL 2021) | Warning: not all compositional splits reflect natural distribution shifts | Motivates matching confounders |
| Andreas, *GECA* (ACL 2020) | Text-recombination data augmentation | An intervention baseline |

The universal finding across this literature is that standard neural
models fail compositional generalization to a measurable, reproducible
degree. **Our contribution is not that finding.** Our contribution is
extending the methodology to a new modality (natural signed video) and
demonstrating a targeted training intervention that specifically
addresses relational binding.

---

## 3. Why this framing fits our specific failure signature

### 3.1 The consistency across three failed interventions

The Qwen predictions on DEV are the clearest evidence. Two examples from
the actual DEV output:

Reference: *"da können wir mit unserem osterwetter eigentlich ganz zufrieden sein ."*
Prediction: *"dabei bleibt es meist trocken ."*

Reference: *"richtung südosten ist auch noch schnee dabei ."*
Prediction: *"dabei bleibt es meist trocken ."*

Two different videos, one identical prediction. The predicted sentence
is grammatical, domain-plausible German — a phrase the model has seen
in training. The model has not "failed to understand" the video; it has
*substituted* a memorized weather-domain fallback for whatever the
video says. This is what compositional-generalization failure looks
like in a generative model: the space of atomic combinations the model
can produce is smaller than the space of atomic combinations the source
represents.

This same pattern was implicit in the earlier failures:

- Source32 improved the *ingredients* the encoder receives but did not
  improve the model's use of them. If the bottleneck were "not enough
  ingredients", Source32 would have helped translation. It did not.
- pgat-length v2 gave the encoder more capacity to *carry* content but
  the decoder still could not use it well on long targets. If the
  bottleneck were "prefix width", flattening the prefix length curve
  would have flattened the translation curve. It only partly did (in
  chrF, not BLEU).
- Qwen QLoRA re-introduced a larger decoder but relied on the same
  training signal. If the bottleneck were "decoder capacity", the
  larger decoder would have improved long-bin translation. It did not.

### 3.2 What all three failures have in common

Every intervention changed *architecture* while keeping the *training
signal* fixed at "next-token cross-entropy on the reference given the
video prefix". This signal does not require the model to distinguish
between two references that share vocabulary but differ in binding
(e.g. "10° in the north, 20° in Bavaria" vs "20° in the north, 10° in
Bavaria"). Cross-entropy on one reference at a time cannot teach that
distinction.

Compositional generalization studies solve exactly this problem: they
change the *evaluation* to expose the gap, and they change the
*training signal* to close it. That is the direction of this program.

---

## 4. The hypothesis

### 4.1 Formal statement

> **H₁ (compositional gap):** After controlling for video duration,
> reference word count, per-atom training frequency, and signer
> identity, gloss-free SLT models produce measurably worse translations
> on samples whose *combinations* of atoms and relations do not appear
> in training, than on samples whose combinations do appear. The gap
> exists across multiple systems and is not fully explained by observed
> confounders.

> **H₂ (interaction with load):** The compositional gap grows with the
> number of propositions in the reference: cells with more propositions
> and unfamiliar combinations degrade more than cells with fewer
> propositions and unfamiliar combinations, beyond the additive effect
> of either factor alone.

> **H₃ (intervention):** Training with relation-focused negatives —
> contrastive corruptions of the target that preserve vocabulary and
> length but perturb bindings — reduces the compositional gap (H₁) more
> than length-balanced sampling, frequency-balanced sampling, or
> generic in-batch contrastive negatives (SCL-SLT-style), without
> hurting corpus-level BLEU on familiar cells.

### 4.2 What each hypothesis predicts

- H₁ predicts BLEU-4 in cell B (held-out, low load) < BLEU-4 in cell A
  (familiar, low load), with the CI on the delta not crossing zero.
  Same for D < C.
- H₂ predicts (C − D) > (A − B): the compositional gap is larger at
  high load than at low load. This is the interaction effect.
- H₃ predicts, for our intervention *I*: (A_I − B_I) < (A_baseline −
  B_baseline), and similarly for the D − C gap, with A_I not
  significantly below A_baseline.

### 4.3 What would falsify each hypothesis

- H₁ falsified if held-out cells perform *no worse* than familiar cells
  after confound matching. We report the null result and pivot to
  Direction 1 (source-grounded omission) or Direction 4 (adaptive
  re-observation).
- H₂ falsified if the gap does not amplify with load. Report as an
  additive effect only.
- H₃ falsified if our intervention does not beat SCL-SLT or the
  balancing baselines. Report the benchmark alone as the contribution
  (the finding still stands even if the intervention does not).

### 4.4 Why this is not circular

A common criticism of compositional-generalization work is that the
"held-out" set is defined by the researcher, so of course models do
worse on it. Two design choices rule this out:

1. The held-out combinations are constructed via **CFQ-style atom /
   compound divergence** (§8): atoms must be balanced across train and
   test; only their *combinations* differ. If a model that has seen
   every atom cannot combine them correctly, the compositional gap is
   real and not a rarity artifact.
2. A **random-split control** is trained under the same compute budget
   on the same data. If the compositional gap collapses under the
   random split, we have evidence that composition — not our
   construction procedure — drives the effect.

---

## 5. Notation

| Symbol | Meaning |
|---|---|
| $V_i$ | The signed video for sample $i$ |
| $y_i$ | The German reference sentence for sample $i$ |
| $F_i$ | Frame count of $V_i$ |
| $L_i$ | Whitespace word count of $y_i$ |
| $S_i$ | Signer identity for $V_i$ (from PHOENIX metadata when available) |
| $\mathcal{A}$ | The atom vocabulary — {events} ∪ {locations} ∪ {times} ∪ {temperature buckets} ∪ {intensities} ∪ {wind directions} ∪ {polarities} |
| $a_j$ | An individual atom, $a_j \in \mathcal{A}$ |
| $\mathcal{P}_i \subseteq \mathcal{A}$ | Set of atoms extracted from $y_i$ |
| $\mathcal{R}$ | The relation vocabulary — {event-location, temperature-location, event-time, negation-scope} |
| $r_k = (a_p, a_q)$ | An individual relation instance, of some type $\rho \in \mathcal{R}$ |
| $\mathcal{T}_i$ | Set of relation instances extracted from $y_i$ |
| $\mathcal{D}_\text{train}, \mathcal{D}_\text{dev}, \mathcal{D}_\text{test}^{\text{std}}, \mathcal{D}_\text{test}^{\text{comp}}$ | Standard splits + our new compositional test split |
| $A, B, C, D$ | The four cells of the 2×2 evaluation matrix (§9) |
| $\hat y_i$ | The system's prediction for $V_i$ |
| $\mathcal{L}_\text{tra}, \mathcal{L}_\text{rel}$ | Translation loss, relation-focused contrastive loss (§11) |

---

## 6. Semantic schema for PHOENIX14T

### 6.1 Why a symbolic schema at all

Compositional generalization needs a definition of "composition". For
free-text SLT, the only tractable definition is a **structured
propositional representation** of the reference. We build that structure
using a closed-vocabulary schema over the PHOENIX weather domain, which
is small enough for reliable rule-based extraction.

The schema is **a diagnostic scaffold**, not a claim about the full
linguistic representation of DGS. We state this explicitly in every
paper draft. Reviewers accept diagnostic scaffolds when the limitation
is acknowledged.

### 6.2 Slot definitions

| Slot | Value domain | Estimated size | Example |
|---|---|---:|---|
| `event` | Weather events (nouns and verbs) | ~20 | *regen, sonne, schnee, wolken, sturm, wind, gewitter, nebel, frost, tau, hagel, hitze, kälte, gewitter, sturmböen* |
| `location` | German regions, directions, geographic features | ~30 | *nord, süd, ost, west, bayern, sachsen, alpen, küste, breisgau, vogtland, alpenrand, mitte, oberrhein, harz, erzgebirge* |
| `time` | Time expressions | ~15 | *heute, morgen, nachmittag, nacht, wochenende, mittag, abend, tagsüber, übermorgen* |
| `temperature` | Integer, discretized into buckets | 8 buckets | Buckets: [<−5, −5..0, 0..5, 5..10, 10..15, 15..20, 20..25, >25]° |
| `intensity` | Qualitative intensity | ~6 | *stark, leicht, kräftig, mäßig, zeitweise, teilweise* |
| `wind_direction` | Compass directions for wind | ~8 | *nördlich, südwestlich, westlich, östlich* |
| `polarity` | Negation marker | 2 | *pos, neg* (e.g. "kein regen") |

### 6.3 Relation definitions

Each relation is a *typed pair* connecting atoms in the same sentence:

| Relation | Form | Example |
|---|---|---|
| **event-location** | $(a_\text{event}, a_\text{location})$ | *(regen, norden)* from "regen im norden" |
| **temperature-location** | $(b_\text{temp}, a_\text{location})$ | *(10..15, bayern)* from "zwölf grad in bayern" |
| **event-time** | $(a_\text{event}, a_\text{time})$ | *(regen, morgen)* from "morgen regen" |
| **negation-scope** | $(\text{neg}, a_\text{event})$ | *(neg, regen)* from "kein regen" |
| **wind-direction-location** | $(a_\text{wdir}, a_\text{location})$ | *(nördlich, alpen)* from "nördliche winde an den alpen" |

Relations are the compositional unit — the schema is *not* the atoms,
it is the atoms plus how they are bound.

### 6.4 Handling reference variation

PHOENIX references are lower-cased German with punctuation. Real
variation the extractor must handle:

- Inflection: *regen, regnet, regnen* → all map to `event=regen`
- Compounds: *regenschauer, regengebiete* → `event=regen`
- Ordinals: *im norden, in norddeutschland, nördlich* → `location=nord`
- Number expressions: *zehn grad, 10 grad, 10 °* → `temperature=10..15`
- Contractions and clitics: *im, ins* → treated as prepositions
- Ranges: *zehn bis fünfzehn grad* → two temperature values or a single
  span; policy is to record the midpoint

Where variation is ambiguous, the extractor logs an "extraction
uncertain" flag and the sample is excluded from the compositional test
set (but retained in training so we do not throw away data). Section 7
covers the extractor implementation and validation.

### 6.5 Estimated schema coverage

Rough expectation for PHOENIX14T (to be confirmed by the feasibility
audit):

- ~90% of references contain ≥1 extractable proposition.
- Median propositions per reference: 2–3.
- Distinct (event, location) pairs: 100–200.
- Distinct (temperature bucket, location) pairs: 50–100.
- Distinct (event, time) pairs: 30–60.

If the feasibility audit shows these numbers are dramatically lower,
some relations may be too rare to support meaningful held-out cells;
the study then focuses only on the well-populated relation types.

---

## 7. The rule-based extractor and its validation

### 7.1 Extractor design

Implementation: `src/pgat_length/composition/extractor.py` (to build).

The extractor is deliberately **rule-based**, not learned, so that:

- It is transparent and inspectable.
- It has no train-test leakage.
- A reviewer can read the rules and evaluate their coverage.
- It runs in seconds on the full corpus.

Pipeline:

1. **Normalization**: lower-case, strip punctuation, replace numeric
   words with digits ("zehn" → 10), unify diacritics.
2. **Lexicon matching**: for each of the 7 slot types, a curated German
   lexicon of surface forms + morphological variants. Compiled to a
   single Aho-Corasick automaton for speed and unambiguous matching.
3. **Temperature parsing**: regex `(-?\d+)(?:\s*bis\s*(-?\d+))?\s*grad`;
   ranges collapsed to midpoint bucket.
4. **Polarity detection**: token-window scan for negation markers
   (*kein, nicht, keine*) preceding an event token.
5. **Relation binding**: for each pair (event, location) in the same
   clause (bounded by punctuation or subordinating conjunctions),
   record the relation instance. Same for temperature-location and
   event-time.
6. **Consistency checks**: no atom may participate in more than one
   *contradictory* relation instance of the same type (e.g. no atom is
   both a location and a time). Contradictions are logged and the
   sample is flagged.

### 7.2 Extractor output format

Per sample, one JSON row:

```json
{
  "uid": "0PC3ZM7RVvVOlKAT4kQD-dev00042",
  "reference": "morgen zehn grad im norden und zwanzig grad in bayern",
  "atoms": {
    "event": [],
    "location": ["nord", "bayern"],
    "time": ["morgen"],
    "temperature": ["10..15", "20..25"],
    "intensity": [], "wind_direction": [], "polarity": []
  },
  "relations": {
    "temperature_location": [["10..15", "nord"], ["20..25", "bayern"]],
    "event_time": [], "event_location": [], "negation_scope": [],
    "wind_direction_location": []
  },
  "num_propositions": 2,
  "extraction_confidence": "high",
  "extraction_notes": []
}
```

### 7.3 Extractor validation

Every downstream compositional metric depends on the extractor being
accurate. Validation protocol:

1. **Sample selection**: 100 references sampled stratified by length
   (20 per five-bin bucket).
2. **Dual annotation**: two DGS/German-fluent annotators independently
   extract atoms and relations by hand. Both work from the German
   reference only (this is not a claim about the video — it is a claim
   about what the *reference text* asserts).
3. **Agreement**: Cohen's $\kappa$ between annotators on (a) atom
   presence per slot, (b) relation presence per relation type.
   Target: $\kappa \geq 0.7$ on both. If $\kappa < 0.5$ on any slot,
   that slot is excluded from the study.
4. **Adjudication**: disagreements resolved by a third pass with both
   annotators. This produces the gold set.
5. **Extractor evaluation**: run the extractor on the 100 samples;
   report precision, recall, F1 against the gold set for each slot
   and relation type.

Publication threshold: **extractor F1 ≥ 0.85 on both atoms and
relations**. Below this, downstream metrics are too noisy and the
paper cannot claim precise compositional measurements.

The 100-sample validation set is *not* used to tune the extractor
lexicon. Lexicon development uses a separate 100-sample development set
drawn from TRAIN (never DEV or TEST).

### 7.4 What we cannot claim without video verification

The extractor operates on the German reference. It says nothing about
whether the video actually communicates the extracted propositions.
Two possible failure modes exist:

- **Reference over-specification**: the reference contains information
  the video did not actually communicate (translator inference).
- **Reference under-specification**: the video communicated information
  the reference omits.

Ideally, a subset of samples is verified against the signed video by a
DGS-fluent reviewer. Where such review is not available, we restrict
claims to *"reference-side semantic coverage"* — a valid but narrower
target. Do not advertise "visually-verified semantic content" without
the verification step.

---

## 8. Compositional split construction

Split construction is the single load-bearing methodological step. If
this is done poorly, every downstream number is contaminated. This
section is written as an implementation spec.

### 8.1 The CFQ atom/compound divergence method

Adapted from Keysers et al. (2020). The idea:

- Let $\mathcal{A}$ be the atom vocabulary and $\mathcal{C}$ the set of
  observed compounds (in our case, relation instances).
- A "compositional" test set is one where atom distributions are
  **matched** across train and test (every atom equally represented)
  but compound distributions **diverge** (many test compounds absent
  from training).
- Formally: let $p_\text{train}(a)$ be the frequency of atom $a$ in
  training; $p_\text{test}(a)$ in test. We want the atom-divergence
  $D_\text{atom} = D_\text{KL}(p_\text{test} \| p_\text{train})$
  small (matched). We want the compound-divergence $D_\text{compound}$
  large (diverged).
- CFQ maximizes $D_\text{compound}$ subject to $D_\text{atom} < \epsilon$.

### 8.2 The specific splits we build

Three splits are constructed and each is used consistently:

- **`std_split`** — the official PHOENIX14T train/dev/test partition.
  Used only for external comparability (BLEU numbers reviewers can
  cross-check with prior work).
- **`comp_split`** — the compositional partition. Compositional
  train/dev/test carved from the official train + dev, with test
  containing held-out compounds. Sample-count-matched to the standard
  splits where possible.
- **`random_split`** — a control. Same train/test sizes as `comp_split`
  but atoms and compounds shuffled randomly. Used to demonstrate that
  gaps observed on `comp_split` are not shared by any random split.

The **official PHOENIX TEST set is not touched** for the compositional
study — it is reserved for the final frozen-architecture standard-split
run. This preserves the split discipline stated in CLAUDE.md.

### 8.3 Split construction algorithm

Inputs: extracted atoms + relations per sample from §7.

Steps:

1. **De-duplication**: reference-text-level near-duplicate clustering
   using n-gram Jaccard $\geq 0.8$. All cluster members go to the same
   side of the split.
2. **Atom frequency bounds**: compute per-atom training frequency
   after de-duplication. Any atom with frequency < 5 in the intended
   training set is *excluded* from being a "held-out combination
   participant" — we do not test rare-word recognition disguised as
   composition.
3. **Compound selection**: for each relation type, list all observed
   compounds. Rank by frequency. Reserve the *lowest-frequency 20%* of
   compounds for the held-out set. Their sentences move to test.
4. **Leakage check**: for each held-out compound $(a_p, a_q)$, verify
   no training sentence contains both atoms in a way that could be
   reconstructed by the model — including partial-match variants
   (e.g. paraphrases). Sentences violating this move to test.
5. **Sample-cell assignment**: for each remaining sentence, count
   propositions and classify:
   - Low load: 1–2 propositions.
   - High load: 3+ propositions.
   Combined with compound-familiarity, this yields the four cells A, B,
   C, D. Sample counts are logged.
6. **Confound matching**: within-cell, subsample so that duration
   distributions and per-atom rarity distributions overlap between
   cells (matched to A as the reference). Use inverse-propensity-based
   subsampling to preserve as many samples as possible.
7. **Divergence report**: compute atom-KL and compound-KL between train
   and test. Report values.

Target sizes:

- **Aim**: ≥ 50 samples per cell after matching, giving ≥ 200 samples
  for the compositional test set.
- **Minimum**: ≥ 30 samples per cell.
- **Kill criterion**: if cells B and D are both < 20 after matching,
  the split is infeasible on PHOENIX alone — pivot to CSL-Daily.

### 8.4 Confounders to explicitly control

| Confounder | Matched? | Reason |
|---|---|---|
| Reference word count $L_i$ | Yes | Prevents "held-out" cells being just longer |
| Video duration $F_i$ | Yes | Prevents "held-out" cells being just harder frames |
| Per-atom training frequency (min over atoms in $y_i$) | Yes | Prevents rarity confound |
| Signer identity $S_i$ | Yes, when metadata available | Prevents signer-specific effects |
| Ratio of function words to content words | No (secondary) | Reported but not matched |

Distribution-match statistics reported in the paper as a table.

### 8.5 Multiple partitions for stability

A single compositional partition is not enough. We construct **three
independent partitions** by seeding step 3 differently (which 20% of
compounds get held out). Results are averaged across partitions with
across-partition standard deviation reported. If the three partitions
disagree on the sign of the compositional gap, the finding is not
robust.

### 8.6 Cached-feature reuse policy

The frozen visual features (DINOv2 spatial, TimeSformer motion,
MediaPipe pose) do not depend on the split — they can be reused
across all three splits (std, comp, random). Everything downstream
must be retrained:

- Stage 1 alignment: retrained per split.
- Stage 2 translation: retrained per split.
- All baselines and interventions: retrained per split.

Compute budget: roughly 3× the current single-split budget, since we
train under three splits (std retrained for fair baseline comparison
too).

### 8.7 Handling the existing checkpoints

Existing checkpoints (`alignment_v2_best.pt`, `translation_v2_best.pt`,
`translation_qwen_best.pt`) have been trained on the full official
train set. **They cannot be used as the model under study for the
compositional split** — they have already seen every combination in
the intended compositional test set. They can be used only as:

- Diagnostic evaluations of the *existing* models' compositional gaps
  (a legitimate use — we retroactively run the compositional
  evaluation on them to show pre-existing failures).
- Sanity-check baselines for training pipeline correctness.

---

## 9. The 2×2 evaluation matrix

### 9.1 Axes

**Axis 1 — compound familiarity:**
- *Familiar*: every relation instance in $\mathcal{T}_i$ appears
  somewhere in the training set.
- *Held-out*: at least one relation instance in $\mathcal{T}_i$ does
  not appear in the training set.

**Axis 2 — semantic load:**
- *Low load*: 1–2 propositions ($|\mathcal{T}_i| \leq 2$).
- *High load*: 3+ propositions ($|\mathcal{T}_i| \geq 3$).

### 9.2 The four cells

|  | Familiar | Held-out |
|---|---|---|
| Low load | **A** (baseline: familiar, simple) | **B** (pure composition, few propositions) |
| High load | **C** (familiar composition under load) | **D** (held-out composition under load) |

Target sample counts (per §8): $\geq 50$ each, $\geq 30$ minimum.

### 9.3 The four contrasts and what they measure

Let $M(\cdot)$ be any metric (BLEU-4, chrF, relation accuracy, …).

- **A − B**: pure compositional effect at low load. Isolates whether
  the model handles unfamiliar combinations when only one or two
  propositions are involved.
- **A − C**: pure load effect at familiar composition. Isolates whether
  the model handles more content when composition is familiar.
- **C − D**: additional compositional cost under load. Tests interaction.
- **B − D**: load amplification of compositional cost. Tests interaction
  from the other direction.

The most interesting outcome is a **positive interaction**: $(C − D) >
(A − B)$, meaning composition costs more when load is higher. This is
what H₂ predicts and is the most novel finding.

### 9.4 The one plot that decides everything

X-axis: cells A, B, C, D (in that order). Y-axis: BLEU-4 (primary) plus
side plots for chrF and relation accuracy. Multiple lines, one per
system. What we want to see for a positive result:

- All baselines: A > B, C > D, with A > C or A ≈ C (load effect small).
- Our intervention: A > B narrowed, C > D narrowed, A_int ≈ A_baseline
  (intervention did not hurt familiar-cell BLEU).

If the plot shows that pattern on both PHOENIX and a second dataset,
we have a paper.

### 9.5 Sanity-check corollary: matched random split should not show
the pattern

If `random_split` also shows A > B, C > D of similar magnitudes, our
"compositional" cells are not really compositional — they're just hard
for some other reason (rarity, length, signer, discourse position). The
random-split control is a required experiment.

---

## 10. Failure attribution

### 10.1 Why three complementary tasks

A single BLEU number on cell B tells you the model failed. It does not
tell you *where in the pipeline* the failure originated. Three
complementary tasks decompose failure into perceptual, binding, and
generation components.

### 10.2 Task 1 — concept recognition

**What:** given $V_i$, predict which atoms are present in $\mathcal{P}_i$
without generating text.

**Implementation:** a simple multi-label linear head on top of the
frozen PGAT visual encoder outputs (mean-pooled global summary).

$$
p(a | V_i) = \sigma\!\left(w_a^\top \operatorname{meanpool}(\mathbf{H}^\text{glob}(V_i)) + b_a\right)
$$

Trained with binary cross-entropy per atom. Cheap to train (a few
minutes) once the encoder is frozen.

**Metric:** per-slot F1 (event, location, time, temperature) at
threshold 0.5, and macro-average across slots.

**What it isolates:** whether the visual side even *sees* the ingredients.

### 10.3 Task 2 — relational discrimination

**What:** for each test sample, construct 3–5 minimal-pair alternatives
that share atoms but perturb bindings. Score each alternative under the
model's translation likelihood. Report accuracy of choosing the correct
alternative.

**Example (temperature-location binding):**

Reference: *"zehn grad im norden und zwanzig grad in bayern"*
Alternatives:
1. Correct: *"zehn grad im norden und zwanzig grad in bayern"*
2. Values swapped: *"zwanzig grad im norden und zehn grad in bayern"*
3. Locations swapped: *"zehn grad in bayern und zwanzig grad im norden"*
4. Both swapped: *"zwanzig grad in bayern und zehn grad im norden"*

Model scores each alternative $y^{(k)}$ under $p(y^{(k)} | V_i)$ using
teacher-forced token log-probabilities. Prediction is $\arg\max_k$.

Note that alternative 4 is *another valid ordering* — swap both is
identity on unordered content but perturbs order. The scoring treats
each alternative as a separate teacher-forced likelihood evaluation.

**Metric:** top-1 accuracy across all alternative sets.

**What it isolates:** whether the model represents *bindings* between
atoms, independently of generation. A model with high concept F1 but
low relational discrimination has the ingredients but scrambles the
bindings — this is the definitive compositional-failure signature.

### 10.4 Task 3 — free translation

**What:** the standard task, beam search generation. Metrics:
BLEU-1..4, chrF, ROUGE-L F1 (using the existing
`src/pgat_length/evaluation/metrics.py`).

**What it isolates:** actual translation performance — the outcome the
thesis wants to improve.

### 10.5 Interpretation table

| Concept F1 | Relation accuracy | Free BLEU-4 | Interpretation |
|---|---|---|---|
| Low | Low | Low | Perception is the bottleneck. Fix the visual encoder first. |
| High | Low | Low | **Composition/binding is the bottleneck. ← what H₁ predicts.** |
| High | High | Low | Generation problem: decoder or training objective. Not compositional. |
| High | High | High | Model is fine on this sample. |

If cells B and D show pattern (2) significantly more than cells A and
C do, H₁ is supported. If they show pattern (1), we have a
representation problem masquerading as composition. If they show
pattern (3), our framing is wrong.

---

## 11. The intervention

### 11.1 Motivation

Standard translation training minimizes $-\log p(y_i | V_i)$ for one
correct reference at a time. This objective does not force the model
to distinguish between two sentences that share vocabulary but differ
in binding. A model that produces {*10, 20, north, bayern*} in any
order gets partial credit; a model that produces the wrong binding
also gets partial credit under BLEU.

The intervention adds a **contrastive term** that penalizes assigning
high likelihood to a *corrupted* target — a modified reference that
shares vocabulary and structure but has a wrong binding.

### 11.2 Corruption operations (train-only)

Applied to each training sample $(V_i, y_i)$ using the extracted
$\mathcal{T}_i$:

- **Location swap**: swap two location atoms in a temperature-location
  or event-location pair. E.g. *"regen im norden, sonne im süden"* →
  *"regen im süden, sonne im norden"*.
- **Value swap**: swap two temperature buckets in a
  temperature-location pair. E.g. *"10 grad im norden, 20 in bayern"* →
  *"20 grad im norden, 10 in bayern"*.
- **Time change**: replace a time atom with a different one (drawn
  from the training-set time atom distribution). E.g. *"morgen regen"*
  → *"heute regen"*.
- **Polarity flip**: insert or remove a negation marker. E.g. *"regen
  im norden"* → *"kein regen im norden"*.

Not applied:

- **Held-out atom substitution**: never insert an atom from the
  held-out compound set — that would leak the test distribution.
- **Content deletion**: dropping propositions produces valid shorter
  translations, not incorrect bindings; not useful as a contrastive
  negative.

### 11.3 Filtering criteria for corruptions

A generated corruption is retained only if:

1. **Vocabulary preserved**: same word set as original (verified
   token-wise).
2. **Length preserved**: word count within ±1 of original.
3. **Distinct meaning**: the corrupted proposition is not accidentally
   equivalent to the original (e.g. after a temperature swap, if the
   two temperatures were in the same bucket, no distinct meaning is
   produced; filter out).
4. **Grammatical**: passes a lightweight German-grammaticality check
   (rule-based, checks agreement between event and location markers).
5. **Not a valid alternative**: e.g. if two locations are actually
   synonyms in the domain (*norddeutschland* and *nord*), swapping
   them produces an equivalent sentence — filter out.

If no corruption survives filtering, the sample contributes only the
standard translation loss (no relation-focused term).

### 11.4 Loss formulation

For a batch containing sample $i$ with correct target $y_i$ and a set
of $K_i$ retained corruptions $\{y_i^{(k)}\}_{k=1}^{K_i}$:

**Translation loss (standard):**

$$
\mathcal{L}_\text{tra}(i) = -\log p(y_i | V_i; \theta)
$$

where $p(y_i | V_i; \theta)$ is the model's teacher-forced likelihood.

**Relation-focused margin loss:**

$$
\mathcal{L}_\text{rel}(i) = \frac{1}{K_i} \sum_{k=1}^{K_i}
    \max\!\left(0,\ m - s(V_i, y_i) + s(V_i, y_i^{(k)})\right)
$$

where $s(V_i, y) = \frac{1}{|y|} \log p(y | V_i; \theta)$ is the mean
token log-likelihood of $y$ under the model, and $m$ is a margin
hyperparameter.

**Total loss:**

$$
\mathcal{L} = \mathcal{L}_\text{tra} + \lambda \cdot \mathcal{L}_\text{rel}
$$

### 11.5 Hyperparameter defaults

- Margin $m = 0.1$ (in log-probability space, per-token).
- Loss weight $\lambda \in \{0.1, 0.3, 1.0\}$; start with 0.3, sweep on
  DEV.
- Corruptions per sample: up to 3; sampled uniformly across
  corruption types.
- Warmup: apply $\mathcal{L}_\text{rel}$ only after $\mathcal{L}_\text{tra}$
  has converged for 2 epochs (avoids collapsing on hard negatives before
  the model can predict the reference).

### 11.6 What this intervention specifically differs from

**vs SCL-SLT (ACL 2026 — Selective Contrastive Learning):** SCL-SLT
selects *informative negatives from the batch* — the negatives are
other real training samples deemed hard by a scorer. Our corrupted
negatives are *constructed*, target *specific binding perturbations*,
and preserve surface features. The two approaches are complementary but
distinct. The ablation table (§13) directly compares them under
matched compute.

**vs GECA (Andreas 2020 — Good-Enough Compositional Data Augmentation):**
GECA augments the *training data* with recombined sentences from
existing training examples. Our approach uses the corruptions as
*contrastive negatives*, not as additional positive examples. GECA is
another possible baseline (§12).

**vs standard label smoothing:** label smoothing spreads probability
across all tokens uniformly. Our intervention spreads probability
across a *structured* set of alternatives that target a specific
compositional weakness.

---

## 12. Baselines and controls

Every configuration below uses the same visual encoder, same
alignment stage, same decoder architecture, and same total training
budget. The only difference is the training-time signal.

| Variant | Rationale | What it rules out |
|---|---|---|
| **Original translation objective** | Establishes the compositional gap without any intervention | Nothing — this defines the baseline gap |
| **Length-balanced sampling** | Oversample longer training sentences | "Model saw fewer long sentences" |
| **Concept-frequency-balanced sampling** | Oversample sentences containing rare atoms | "Rare atoms are the real problem" |
| **GECA-style augmentation** | Add train-time recombined sentences | "Simple data augmentation suffices" |
| **SCL-SLT (in-batch informative negatives)** | Selective contrastive with hard batch negatives | "Any contrastive loss would work" |
| **Text-only translation baseline** | Train encoder-decoder on references only, no video | Language model prior alone (ceiling for how much a smart LM can do) |
| **Proposed relation-focused negatives** | The intervention | This is our proposed method |

**Compute matching**: total training tokens per configuration held
constant. If length-balancing oversamples, matched configurations
downsample elsewhere.

**Multi-seed protocol**: each configuration trained with 3 seeds.
Reported numbers are mean ± std across seeds.

**Multi-partition protocol**: each configuration trained on 3
compositional partitions (§8.5). Reported numbers are mean ± std
across partitions.

**Total training budget**: 7 configurations × 3 seeds × 3 partitions =
63 training runs. On A6000 at ~1.5 hours per PGAT-length training run,
this is roughly 95 GPU-hours. Plan for a full week of dedicated GPU
time on the compositional-split baselines alone.

---

## 13. Evaluation metrics in full

Five layers, each answering a specific question.

### 13.1 Layer 1 — corpus-level translation quality

The standard SLT table, on both the standard PHOENIX test split and
each compositional test partition.

- BLEU-1, BLEU-2, BLEU-3, BLEU-4 (SacreBLEU 13a, exponential smoothing)
- chrF (`char_order=6, β=2`)
- ROUGE-L F1 (rouge-score, no stemmer)
- Exact match rate
- Mean generation length
- Empty prediction rate

Every metric reported with a 95% paired bootstrap CI.

### 13.2 Layer 2 — the 2×2 compositional table

|  | Familiar | Held-out | Δ (comp gap) |
|---|---:|---:|---:|
| Low | A | B | B − A |
| High | C | D | D − C |
| Δ (load gap) | C − A | D − B | interaction |

Reported per system, per metric, per partition. Primary claim is
**H₁**: B − A and D − C are significantly negative.

### 13.3 Layer 3 — slot and relation accuracy

Extractor-based metrics on generated predictions:

| Metric | Definition | Purpose |
|---|---|---|
| Concept precision (per slot) | Atoms in prediction ∩ reference / atoms in prediction | Are generated atoms real? |
| Concept recall (per slot) | Atoms in prediction ∩ reference / atoms in reference | Missing atoms? |
| Concept F1 (per slot, and macro) | Harmonic mean | Overall recognition |
| Relation accuracy (per relation type) | Relations in prediction correctly bound / relations in reference | **Compositional binding** |
| Proposition recall | Reference propositions expressed in prediction | Coverage |
| Complete-example accuracy | Fraction of samples with all propositions and relations correct | Strict, small |

### 13.4 Layer 4 — failure-attribution diagnostic

For held-out cells (B, D), report the three tasks from §10 together:

| Cell | Concept F1 | Relational discrimination acc | Free BLEU-4 |
|---|---:|---:|---:|

Interpretation via the table in §10.5.

### 13.5 Layer 5 — grounding and ceiling controls

- **Shuffled-video BLEU-4**: replace each sample's video with a
  different sample's video; regenerate. Expected drop: ≥ 6 BLEU-4 for
  a genuinely grounded model.
- **Text-only ceiling BLEU-4**: encoder-decoder trained on references
  only. If our video-based model does not beat this on held-out
  cells, the model is not using the video and the study collapses.
- **Extractor-vs-human agreement**: reported as Cohen's $\kappa$ on
  the 100-sample validation set.

### 13.6 Reporting conventions

- Every number reported with a 95% CI in brackets.
- Every table shows the metric, mean, std across seeds and partitions,
  and CI.
- Cell counts shown alongside metrics.
- Extractor precision/recall/F1 shown once, in the methods section.
- All comparisons flagged as "same-scorer" or "cross-tokenizer" — never
  mixed.

---

## 14. Statistical protocol

### 14.1 Paired bootstrap

For any metric $M$ and any two systems $s_1$ and $s_2$ evaluated on the
same test set:

1. Compute per-sample statistics (or per-sample scorer contributions).
2. Draw $N = 2000$ bootstrap resamples with replacement at the video
   level.
3. On each resample, compute $M(s_1) - M(s_2)$.
4. Report $\hat{\Delta} = \operatorname{mean}$, 95% CI = $[\hat{q}_{2.5}, \hat{q}_{97.5}]$.
5. "Significant" means the CI does not cross zero.

### 14.2 Multi-seed averaging

Every trained configuration is run with ≥ 3 random seeds. Reported
values are mean across seeds; report std across seeds as a separate
column. Some cells will have only two seeds due to compute limits;
document exactly which.

### 14.3 Multi-partition averaging

Every trained configuration is run on ≥ 3 compositional partitions
(§8.5). Reported values are mean across partitions; report std across
partitions.

The final headline number for a configuration on the compositional
test is:

$$
\bar{M} = \frac{1}{|\text{seeds}| \cdot |\text{partitions}|} \sum_{s,p} M(s, p)
$$

with two-way standard error reported.

### 14.4 Multiple comparisons

Roughly 6 systems × 5 metrics × 4 cells = 120 comparisons. Two options:

- **Report CIs only**, and let readers judge. Preferred — cleaner,
  more informative.
- **Bonferroni** with per-comparison $\alpha = 0.05 / N$. Only if a
  reviewer specifically requests family-wise error control.

### 14.5 Effect sizes

Cohen's $d$ for each significant comparison. Small effects
($d < 0.2$) reported as "measurable but small" rather than trumpeted.

### 14.6 Pre-registration (recommended)

Publish the hypothesis (§4), the schema (§6), the split-construction
algorithm (§8), the metrics (§13), and the statistical protocol (§14)
in a pre-registration document *before* running the compositional
evaluation. This is not strictly required for a master's thesis but
substantially strengthens reviewer trust.

---

## 15. Feasibility gate

### 15.1 Purpose

Before any of the above becomes work worth doing, one question must
be answered: **can PHOENIX14T support the required compositional
split at all?** The feasibility gate answers that in one day.

### 15.2 The feasibility script

Implementation: `scripts/20_composition_feasibility.py`.

Inputs:
- The manifest pickle (train + dev + test references).
- The extractor from §7.

Outputs:
- `docs/feasibility_report.md` — a two-page human-readable report.
- `outputs/composition/feasibility_stats.json` — machine-readable
  statistics.

What the script computes:

1. **Extractor coverage**: fraction of references producing ≥ 1
   extracted proposition, distribution of propositions per reference.
2. **Atom vocabularies**: sizes of event, location, time, temperature,
   intensity, wind-direction sets.
3. **Compound distributions**: per-relation-type histograms of
   compound frequencies.
4. **Held-out cell simulation**: apply the split algorithm (§8) with
   default parameters; report sample counts per cell A/B/C/D before
   and after confound matching.
5. **Confounder overlap**: distributions of duration, reference length,
   per-atom rarity per cell; a KS-test p-value for each pair (A vs B,
   A vs C, etc.).

### 15.3 Decision criteria

All three must be true to proceed:

1. **Extractor recall** on the 100-sample validation set ≥ 0.85.
2. **Cell B and cell D sample counts** each ≥ 30 after confound matching.
3. **Confounder overlap**: KS test p > 0.05 for duration and reference
   length distributions between cell A and each of B, C, D — i.e.
   distributions are statistically indistinguishable after matching.

If any criterion fails:

- Extractor recall < 0.85 → invest another 2–3 days in lexicon
  development; if still < 0.85, restrict the study to well-extracted
  relations only.
- Cell counts insufficient → try CSL-Daily as primary dataset (much
  larger; ~19k samples).
- Confounder overlap insufficient → downgrade causal claim to
  correlation; report the residual confound.

### 15.4 Feasibility runtime

Extractor development (lexicon + rules): 4-8 hours.
Extractor validation (100-sample dual annotation): 4-8 hours by two
annotators.
Script implementation: 2-3 hours.
Running the script and writing the report: 2 hours.

**Total: 1 person-day if lexicons come together, 3 person-days if not.**

---

## 16. Timeline and decision gates

### 16.1 Week-by-week plan

**Weeks 1–2 — Feasibility and pre-registration**

- Extractor lexicon development (§7).
- 100-sample dual annotation.
- Feasibility script and report (§15).
- Pre-registration document (§14.6, recommended).
- **Gate**: proceed only if feasibility passes.

**Weeks 3–4 — Baselines on compositional split**

- Retrain: original translation objective, length-balanced,
  frequency-balanced, GECA on `comp_split` (3 seeds × 3 partitions
  each — heavy).
- Also retrain on `std_split` and `random_split` for controls.
- **Gate**: proceed only if H₁ shows a measurable gap (B < A with CI
  not crossing zero, in ≥ 2 of 3 partitions).

**Weeks 5–6 — Failure attribution**

- Train Task 1 (concept recognition) heads on top of the frozen
  encoder.
- Build Task 2 (relational discrimination) alternative sets and score
  under each baseline model.
- Populate the diagnostic table (§10.5).
- **Gate**: proceed only if held-out cells show pattern (high concept
  F1, low relational discrimination, low free BLEU-4). If pattern
  differs, reframe.

**Weeks 7–8 — Relation-focused intervention**

- Implement the corruption module (§11.2, 11.3).
- Train the relation-focused variant (3 seeds × 3 partitions).
- Train SCL-SLT variant for direct comparison.
- **Gate**: proceed to Weeks 9–10 only if the intervention reduces the
  compositional gap significantly more than baselines.

**Weeks 9–10 — Second-dataset validation**

- Cache features for CSL-Daily (this is heavy: 25k+ samples;
  incremental extraction to respect quota).
- Run the extractor on Chinese references (requires a Chinese lexicon
  build — parallel effort during Weeks 3–8).
- Feasibility audit on CSL-Daily.
- Run best-performing baseline + intervention on CSL-Daily.
- **Gate**: if the compositional gap and intervention effect reproduce
  on CSL-Daily, the paper is publishable at a top venue. If not,
  report as PHOENIX-specific finding.

**Weeks 11–12 — Frozen evaluation and paper writing**

- Lock thresholds on DEV.
- Run standard PHOENIX TEST for external comparability (one time,
  no design tuning after).
- Full metric tables.
- Draft paper.
- Prepare defence slides.

Total: **12 weeks / 3 months of concentrated work.** Realistic
timeline given cluster queue delays and annotation logistics: **4-5
months**. Budget slippage in advance.

### 16.2 Minimum publishable finding at each gate

If any gate fails, the "minimum publishable finding" up to that point:

- After Weeks 3–4 with H₁ null: negative result paper — "compositional
  generalization is not the operative bottleneck on PHOENIX14T; the
  three architectural interventions failed for reasons X, Y, Z" —
  LREC-workshop or similar.
- After Weeks 5–6 with a pattern (1) diagnostic: reframe as
  "representation quality is the bottleneck; released concept-F1
  benchmark shows 7 SLT systems fail to recognize atoms in long
  contexts" — workshop or short paper.
- After Weeks 7–8 with intervention not beating SCL-SLT: benchmark-only
  paper — "we release a compositional benchmark and demonstrate that
  7 systems all show the same measurable gap".
- After Weeks 9–10 with second-dataset failure: PHOENIX-only paper
  with reduced venue target (LREC/ACL Findings).

---

## 17. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| PHOENIX too small/homogeneous for balanced split | Medium (30–40%) | CSL-Daily fallback. Time cost: +2 weeks for feature caching. |
| Extractor F1 < 0.85 despite lexicon work | Low-medium | Restrict study to well-extracted relations; drop temperature-location if numeric parsing is unreliable. |
| Cell counts too small (< 30 after matching) | Medium | Fewer relation types; drop the load axis and study only compound familiarity. |
| Intervention does not beat SCL-SLT | Medium-high | Fall back to benchmark-only paper. |
| DGS annotator access limited | High | Restrict claims to reference-side semantic coverage; postpone video-side verification to future work. |
| Reviewer scoop by concurrent 2026 paper | Medium | Monthly arXiv/OpenReview search; differentiate on the *decomposed diagnostic* angle (Task 1/2/3) which text-only compositional work cannot do. |
| Timeline slippage | High | Weekly checkpoints; commit to minimum-publishable-finding at each gate. |
| Second-dataset (CSL-Daily) access + storage | Medium | Query the dataset owner early; storage quota check at Week 5. |
| Multi-seed × multi-partition compute budget too large | Medium | Reduce to 2 seeds × 2 partitions if the A6000 queue is contested. Document the reduction explicitly. |

---

## 18. Defense Q&A

Every question a professor or reviewer is likely to ask, with a
written answer to rehearse.

### 18.1 On the framing

**Q1: How is this different from just running BLEU on long sentences?**

Length is one axis; compositional novelty is a separate axis. The 2×2
cells C and A both have the same load; the difference is whether
combinations are familiar. If C ≈ A but B ≪ A, length was a red
herring and composition is the real driver. Five-bin length analyses
cannot make this distinction because they hold nothing else constant.

**Q2: What is the deep reason to believe this is the right framing?**

Three separate architectural interventions on our stack failed for what
looks like the same reason: fluent, domain-plausible generation with
weak visual conditioning. Two different videos producing an identical
prediction on the Qwen run is the smoking gun. Every intervention
changed *architecture* while keeping the training signal fixed at
"maximize likelihood of one reference per video". That signal cannot
teach the model to distinguish two references with the same vocabulary
but different bindings. Compositional-generalization studies solve
exactly this problem.

**Q3: Isn't PHOENIX too small and templated to study compositional
generalization?**

Templated is a *feature*, not a bug, for this study. A closed weather
domain gives us clean atom extraction and tractable combinatorics.
Whether it is *large enough* is the exact question the feasibility gate
answers in one day. If PHOENIX cannot support the split, CSL-Daily is
the primary dataset and PHOENIX is a secondary check.

### 18.2 On the split construction

**Q4: How do you ensure the held-out compounds are not just harder for
other reasons — rarer atoms, longer duration, different signer?**

The split algorithm (§8.3) explicitly matches on per-atom training
frequency, video duration, reference length, and signer identity when
metadata allows. The feasibility report shows the distributions side by
side. The random-split control provides an additional check: if the
random split shows the same gap, our compositional construction is
measuring something else.

**Q5: Why should I believe your compositional split is "compositional"
and not just a lucky partition?**

Three independent partitions are constructed by seeding step 3 of the
algorithm differently. Results are reported as mean ± std across
partitions. If the three partitions disagree on the sign of the gap,
the finding is flagged as unstable. The CFQ atom/compound divergence
statistics are reported alongside cell counts so a reviewer can verify
divergence is high.

**Q6: Your existing PGAT checkpoints saw the full train set. How can
they possibly be used as models under study for your split?**

They cannot, and we say so explicitly. Existing checkpoints are used
only for (a) retroactive evaluation of pre-existing models on the
compositional test (a legitimate diagnostic), and (b) pipeline
correctness sanity checks. Everything under study is retrained on the
new split. The cost is real: roughly 3× the compute budget of the
current single-split work.

### 18.3 On the extractor and schema

**Q7: How do you know the extractor is right?**

We validate it against a 100-sample gold set constructed by two
independent annotators with adjudication. Publication threshold is
extractor F1 ≥ 0.85. Below this, the schema is restricted to
well-extracted relations. Cohen's $\kappa$ between the two annotators
is reported to show the annotation is itself reliable.

**Q8: Your schema is not a full linguistic representation of DGS. Is
that not a problem?**

The schema is a diagnostic scaffold, not a linguistic claim. We
acknowledge this explicitly in every paper draft. What the schema does
support is a controlled, reproducible test of whether specific
propositional bindings survive translation. What it does not support
is claims about non-manual markers, discourse structure, or the full
grammar of DGS. Those are excluded from claims.

**Q9: Why not use a learned semantic parser?**

Because a learned parser introduces its own training and errors, and
its errors are correlated with the model under study (both trained
on similar text). The rule-based extractor is transparent, has no
correlation with the model, and can be inspected line-by-line by a
reviewer. Its ceiling is limited by lexicon completeness, which is
tractable in a closed domain.

### 18.4 On the intervention

**Q10: SCL-SLT (ACL 2026) already uses contrastive negatives. What's
new about your intervention?**

SCL-SLT selects *informative in-batch negatives* — the negatives are
other real training samples. Our negatives are *constructed corruptions*
of the correct target that preserve vocabulary and length but perturb
bindings. The two are complementary but distinct. The ablation table
compares them directly on the compositional gap under matched compute.
If SCL-SLT closes the gap by itself, we report that honestly and
credit them. Our claim is not that any contrastive loss works; the
claim is that *this specific* structured contrastive loss addresses
compositional binding.

**Q11: How do you avoid the corrupted texts leaking the test
distribution?**

Corruptions never use atoms that appear only in the held-out compound
set. Corruptions are constructed from atoms in the training set. This
is enforced in the corruption filter (§11.3) and audited as a hard
constraint on every training batch.

**Q12: What if length-balancing alone closes the gap?**

We report that as the finding. It is important either way. If length
balancing is enough, the story is "compositional gap is really a length
gap, and simple balancing suffices". If not, our intervention has a
distinct effect to explain. Either outcome is publishable when reported
honestly.

### 18.5 On the evaluation

**Q13: How do you know the failure is at binding and not at
recognition?**

The three failure-attribution tasks (§10) decompose this. If concept F1
is high but relational discrimination is low, atoms are recognized but
bindings are wrong. If concept F1 is low, recognition is the bottleneck
and the binding measurement is confounded. The pattern of the three
tasks tells us where in the pipeline the failure lives.

**Q14: What's the ceiling — can a smart language model without video
solve compositional cases via priors?**

The text-only translation baseline exists exactly to answer this. It
is trained on references only, no video. Its performance on the
compositional test is the "language prior ceiling". Any video-based
system must beat it on the held-out cells to demonstrate the video is
being used compositionally.

**Q15: Why not human evaluation as the primary metric?**

Cost. We include human validation of the extractor (100 samples,
inter-annotator $\kappa$) and, optionally, a 50-sample end-to-end
adequacy rating on the four cells. But automatic metrics from the
extractor are the primary measurement because they scale to the full
test set.

### 18.6 On the scope and contribution

**Q16: Why is this thesis-worthy rather than "just another benchmark"?**

Three separable contributions: (1) a released, human-validated
compositional benchmark for gloss-free SLT — first of its kind; (2)
empirical evidence that multiple state-of-the-art systems all fail on
it in a specific, decomposable way — the diagnostic framework itself
is novel because text-only compositional benchmarks cannot decompose
grounding vs binding; (3) a targeted training intervention with
falsifiable success criteria. Even if leg (3) fails, legs (1) and (2)
stand.

**Q17: What venues will you target?**

Primary target: EMNLP main track (compositional generalization is a
strong topic there). Secondary: ACL findings, LREC main, EACL findings.
Workshop targets if the intervention fails: LREC workshops, WMT.
Vision-side venues (CVPR, ICCV) are harder because they want more
visual novelty; possible only if the diagnostic-decomposition angle
is highlighted.

**Q18: How do you retroactively justify the earlier failed
experiments given the new framing?**

We do not retroactively rewrite them. The failed experiments are
recorded as-is in the thesis. The compositional framing is presented
as *a new hypothesis motivated by the failure pattern*, not as a
retroactive explanation. To make the motivation empirical, we run the
compositional evaluation on the existing checkpoints too, and report
their compositional gaps as evidence that the compositional framing
identifies a real failure mode in the *existing* models.

---

## 19. Related work map

### 19.1 Tier 1 — Direct SLT competitors (same task, same dataset)

Systems that report BLEU on PHOENIX14T gloss-free translation. Must
be run under our scorer on our splits.

| System | Reference | Local availability | Baseline priority |
|---|---|---|---|
| NSLT | Camgoz, Hadfield, Koller, Ney, Bowden (CVPR 2018) | HF reproduction | Required |
| PGAT-v1 | This project's `pg-adaptor` | Local | Required |
| TSPNet | Li, Rodriguez, Yu, Li (NeurIPS 2020) | Official inference | Required |
| GFSLT-VLP | Zhou et al. (ICCV 2023) | Public code | Required |
| GASLT | Yin et al. (ICCV 2023) | Public code | Required |
| SLTUnet | Zhang, Müller, Sennrich (ACL 2023) | Public | Recommended |
| Sign2GPT | Wong, Camgoz, Bowden (2024) | Recent | Recommended |
| SpaMo | (NAACL 2025) | Recent | Recommended |
| **BoostSLT** | (CVPR 2026) — split-translate-combine for long sentences | Recent | **Required** — direct competitor for length-cliff angle |
| **SCL-SLT** | Selective Contrastive Learning (ACL 2026) | Recent | **Required** — direct competitor for contrastive-negatives intervention |

### 19.2 Tier 2 — Compositional generalization methodology

Foundational papers we cite for framing and split construction. Not
run; cited.

| Paper | Reference | Contribution |
|---|---|---|
| SCAN | Lake, Baroni (ICML 2018) | Compositional test paradigm |
| CFQ | Keysers et al. (ICLR 2020) | Atom/compound divergence method |
| COGS | Kim, Linzen (EMNLP 2020) | Systematic vs primitive generalization |
| CoGnition | Li, Yin, Li, Cheng, Yang, Zhang (ACL 2021) | Compositional NMT benchmark — closest anchor |
| Shaw et al. (ACL 2021) | *Compositional generalization and natural language variation* | Warns against artificial splits |
| GECA | Andreas (ACL 2020) | Text-recombination augmentation |

### 19.3 Tier 3 — Adjacent and supporting

| Paper | Reference | Purpose |
|---|---|---|
| Yazdani et al. (LREC 2026) | Length-sensitivity study | Establishes length sensitivity across systems |
| Hamidullah et al. (ACL 2024) | SEM-SLT | Reproduces length-cliff evidence |
| Grounding or Guessing (ICLR 2026) | Reliability via counterfactuals | Adjacent framing |
| Lost in Translation, Found in Context (CVPR 2025) | Discourse context for SLT | Complementary evidence for beyond-sentence phenomena |
| Winoground (Thrush et al., CVPR 2022) | Image-caption compositional test | Cross-modal analogue |
| Bag-of-words behaviour of VLMs (Yuksekgonul et al., ICLR 2023) | Shows VLMs often ignore relational structure | General motivation |

### 19.4 Citation-hygiene note

Some 2025–2026 papers listed above (BoostSLT, SCL-SLT, Yazdani,
Grounding or Guessing, SpaMo) have not been verified from primary
sources in this document. Verify each citation directly from arXiv /
OpenReview / venue proceedings before submission. The pre-2024
citations (SCAN, CFQ, COGS, CoGnition, GECA, NSLT, TSPNet, GFSLT-VLP,
GASLT) are established and safe.

---

## 20. Publication strategy

### 20.1 The three-leg contribution

| Leg | Contribution | Standalone publishable? |
|---|---|---|
| **Benchmark** | Released compositional test set + extractor + protocol | Yes — LREC or workshop |
| **Finding** | Multiple SLT systems all show the same compositional gap | Yes — EMNLP short or Findings |
| **Intervention** | Relation-focused negatives close the gap more than balancing or SCL-SLT | Requires benchmark + finding to land first |

Best case: all three land, EMNLP main paper.
Middle case: benchmark + finding, ACL Findings or LREC main.
Fallback case: benchmark only with baseline gaps demonstrated, LREC or workshop.

### 20.2 Milestone deliverables

- End of Week 2: pre-registration document publicly posted (arXiv or
  OpenReview).
- End of Week 6: benchmark v0.1 release (extractor + splits + baselines
  ready for external replication).
- End of Week 10: full paper draft.
- End of Week 12: thesis defense.

### 20.3 Reproducibility package

Release with the paper:

- The extractor code and lexicons.
- The three compositional partitions as JSON (uid lists per cell).
- Baseline model predictions per system per cell.
- Evaluation scripts (§13).
- Statistical protocol code (§14).
- A short README with a one-line reproduction command.

---

## 21. Code architecture and reuse

### 21.1 What exists in this repo that will be reused

| Existing module | Reused for |
|---|---|
| `src/pgat_length/features/plans.py`, `spatial.py`, `motion.py` | Frozen visual features — reused unchanged. |
| `src/pgat_length/models/tokenizer.py`, `articulator.py`, `global_summary.py`, `projection.py` | PGAT visual encoder — reused, retrained on new splits. |
| `src/pgat_length/models/translation.py`, `qwen_translation.py` | Decoder side — reused, retrained on new splits. |
| `src/pgat_length/evaluation/metrics.py`, `five_bin.py` | Standard scoring — reused unchanged. |
| `src/pgat_length/evaluation/bootstrap.py` | Paired bootstrap machinery — reused. |
| `src/pgat_length/training/alignment_loop.py`, `translation_loop.py`, `qwen_translation_loop.py` | Training loops — parameterized to accept split path. |

### 21.2 New modules to build

| New module | Purpose |
|---|---|
| `src/pgat_length/composition/schema.py` | Slot and relation vocabularies as Python constants. |
| `src/pgat_length/composition/extractor.py` | Rule-based extractor (§7). |
| `src/pgat_length/composition/splits.py` | CFQ-style split constructor (§8). |
| `src/pgat_length/composition/corruptions.py` | Corruption operations and filtering for the intervention (§11.2, §11.3). |
| `src/pgat_length/composition/metrics.py` | Slot/relation accuracy metrics, 2×2 cell aggregation. |
| `src/pgat_length/training/relation_loss.py` | Margin loss and training-loop hooks for the intervention. |
| `src/pgat_length/evaluation/relational_discrimination.py` | Task 2: alternative-set scoring. |
| `src/pgat_length/evaluation/concept_recognition.py` | Task 1: multi-label head training and evaluation. |
| `scripts/20_composition_feasibility.py` | The one-day feasibility script. |
| `scripts/21_extract_propositions.py` | Batch-run extractor over full manifest. |
| `scripts/22_build_composition_splits.py` | Materialize the compositional / random splits. |
| `scripts/23_train_composition_baseline.py` | Retrain each baseline on a compositional partition. |
| `scripts/24_train_composition_intervention.py` | Train the relation-focused intervention. |
| `scripts/25_evaluate_composition.py` | Full evaluation across five layers (§13). |
| `scripts/26_failure_attribution.py` | Task 1 + Task 2 evaluation for the diagnostic table. |
| `docs/feasibility_report.md` | Output of the feasibility script. |
| `docs/compositional_evaluation_protocol.md` | The one-pager for the advisor meeting. |

### 21.3 Directory sketch after implementation

```
pgat-length/
├── configs/
│   ├── model_v2.yaml, model_qwen.yaml, ...    (existing)
│   └── composition.yaml                        (new: split params, hyperparams)
├── docs/
│   ├── ARCHITECTURE.md                         (existing)
│   ├── RESEARCH_DIRECTIONS_2026-09-08.md       (existing)
│   ├── COMPOSITIONAL_GENERALIZATION.md         (this document)
│   ├── feasibility_report.md                   (new, generated)
│   └── compositional_evaluation_protocol.md    (new, one-pager)
├── src/pgat_length/
│   ├── composition/                            (new)
│   │   ├── schema.py
│   │   ├── extractor.py
│   │   ├── splits.py
│   │   ├── corruptions.py
│   │   └── metrics.py
│   ├── training/
│   │   ├── ...existing loops...
│   │   └── relation_loss.py                    (new)
│   └── evaluation/
│       ├── ...existing scorers...
│       ├── concept_recognition.py              (new)
│       └── relational_discrimination.py        (new)
├── scripts/
│   ├── 01..18 ...existing...
│   ├── 20_composition_feasibility.py           (new — this week)
│   ├── 21_extract_propositions.py
│   ├── 22_build_composition_splits.py
│   ├── 23_train_composition_baseline.py
│   ├── 24_train_composition_intervention.py
│   ├── 25_evaluate_composition.py
│   └── 26_failure_attribution.py
└── slurm/
    ├── ...existing sbatch...
    └── train_composition.sbatch                (new)
```

### 21.4 Estimated code size

- Extractor + schema: ~500 lines (mostly lexicons).
- Splits + corruptions: ~400 lines.
- Metrics + evaluation scripts: ~600 lines.
- Training loop patches + relation loss: ~300 lines.
- Failure-attribution heads: ~200 lines.
- Feasibility script: ~200 lines.
- **Total new code: ~2,200 lines** — comparable to the recovery-experiments
  work already delivered.

---

## 22. References

Verify each citation against primary sources before submission. Where a
venue is stated only tentatively, treat as arXiv preprint.

### 22.1 Compositional generalization

- Lake, B. M., & Baroni, M. (2018). *Generalization without
  systematicity: On the compositional skills of sequence-to-sequence
  recurrent networks.* ICML.
- Keysers, D., Schärli, N., Scales, N., Buisman, H., Furrer, D.,
  Kashubin, S., Momchev, N., Sinopalnikov, D., Stafiniak, L., Tihon, T.,
  Tsarkov, D., Wang, X., van Zee, M., & Bousquet, O. (2020). *Measuring
  compositional generalization: A comprehensive method on realistic
  data.* ICLR.
- Kim, N., & Linzen, T. (2020). *COGS: A compositional generalization
  challenge based on semantic interpretation.* EMNLP.
- Li, Y., Yin, Y., Chen, Y., & Zhang, Y. (2021). *On compositional
  generalization of neural machine translation.* ACL.
- Shaw, P., Chang, M.-W., Pasupat, P., & Toutanova, K. (2021).
  *Compositional generalization and natural language variation: Can a
  semantic parsing approach handle both?* ACL.
- Andreas, J. (2020). *Good-enough compositional data augmentation.* ACL.

### 22.2 Sign language translation

- Camgoz, N. C., Hadfield, S., Koller, O., Ney, H., & Bowden, R. (2018).
  *Neural sign language translation.* CVPR. Introduces PHOENIX14T and
  the NSLT baseline.
- Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). *TSPNet: Hierarchical
  feature learning via temporal semantic pyramid for sign language
  translation.* NeurIPS.
- Zhou, B., et al. (2023). *Gloss-free sign language translation:
  Improving from visual-language pretraining.* ICCV. GFSLT-VLP.
- Yin, A., et al. (2023). *GASLT: Gloss attention for gloss-free sign
  language translation.* ICCV.
- Zhang, B., Müller, M., & Sennrich, R. (2023). *SLTUNET: A simple
  unified model for sign language translation.* ACL.
- Wong, R., Camgoz, N. C., & Bowden, R. (2024). *Sign2GPT: Leveraging
  large language models for gloss-free sign language translation.*
- SpaMo — Frozen spatial/motion features + LLM (NAACL 2025, verify).
- BoostSLT — Boosting SLT via split-translate-combine (CVPR 2026, verify).
- SCL-SLT — Selective contrastive learning for gloss-free SLT (ACL 2026, verify).
- Yazdani, S., et al. (2026). *A multi-metric evaluation of sign
  language translation with a focus on length sensitivity.* LREC
  (verify).
- Hamidullah, Y., et al. (2024). *SEM-SLT: Semantic enhancement for
  sign language translation.* ACL (verify).

### 22.3 Vision-language compositional evaluation

- Thrush, T., Jiang, R., Bartolo, M., Singh, A., Williams, A., Kiela,
  D., & Ross, C. (2022). *Winoground: Probing vision and language
  models for visio-linguistic compositionality.* CVPR.
- Yuksekgonul, M., Bianchi, F., Kalluri, P., Jurafsky, D., & Zou, J.
  (2023). *When and why vision-language models behave like
  bags-of-words, and what to do about it?* ICLR.

### 22.4 Evaluation methodology

- Post, M. (2018). *A call for clarity in reporting BLEU scores.* WMT.
  SacreBLEU reference.
- Popović, M. (2015). *chrF: Character n-gram F-score for automatic MT
  evaluation.* WMT.
- Lin, C.-Y. (2004). *ROUGE: A package for automatic evaluation of
  summaries.* WAS.
- Koehn, P. (2004). *Statistical significance tests for machine
  translation evaluation.* EMNLP. Paired bootstrap.

---

## Appendix A — Reading path for a new collaborator

1. Read §1–§4 to see the framing and the hypothesis.
2. Read §6–§9 with the current PHOENIX manifest open, so the schema
   feels concrete.
3. Skim §10–§14 to see the evaluation plan.
4. Read §15–§17 to understand the go/no-go gates.
5. Use §18 as a rehearsal script for the defense.
6. §21 tells you what to build.

## Appendix B — Immediate next actions

Concrete, deliverable this week:

1. **Read this document end-to-end.** Flag anything unclear.
2. **Implement `scripts/20_composition_feasibility.py`** and the
   supporting `composition/schema.py` and `composition/extractor.py`.
3. **Run it once.** Produce `docs/feasibility_report.md`.
4. **Present the report to the advisor.** Decide go / pivot / CSL-Daily
   based on the numbers.
5. **Do not start baseline retraining until the feasibility gate has
   passed.** This is the discipline that makes the study defensible.

---

*Document version: 2026-09-09. Author: thesis project (Samir Pokharel,
pokharelsamir246@gmail.com) with assisted drafting. Update this
document whenever the schema, split algorithm, extractor thresholds, or
gate criteria change — those are the load-bearing pieces of the study.*
