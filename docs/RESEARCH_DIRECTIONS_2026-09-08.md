# Research directions after PGAT-Length

Research checked: 8 September 2026. This is a targeted literature review and research proposal, not an experimental result or an exhaustive novelty certification. Rankings are research judgments given the project's small-data setting, existing feature pipeline, one-running-job constraint, and 100 GB quota.

**Recommendation:** investigate source-grounded semantic completeness and omission recovery in long gloss-free sign language translation. Start with a controlled diagnostic benchmark and a small feasibility experiment; commit to a repair model only if missing information can actually be recovered from source evidence. The strongest alternative is compositional generalization with controlled evaluation splits.

## What the existing experiments establish

The local ARCHITECTURE.md reports PGAT-v2 TEST BLEU-4 7.20 versus PGAT-v1 9.99, with 3.50 and 3.98 in the 13–18 and 19–24 word bins. The longest bin contains only 16 examples. These are recorded project results, not independently reproduced in this review.

Several interpretations in the existing documentation are stronger than the evidence:

- Improved retrieval without improved translation does not rule out poor local visual representations. Sentence retrieval and preservation of individual meanings are different capabilities.
- Failure of forced longer decoding shows that this particular intervention does not recover content. It does not rule out decoder training, alignment, or stopping behavior as contributors.
- Duration/output-length correlation does not establish a causal compression mechanism.
- Simultaneously changing the decoder and visual tokens cannot isolate either change. Higher long-bin chrF does not identify the cause, nor establish that mBART is intrinsically weaker.
- A flatter length curve can result from damaging short-sentence performance. Absolute improvements and short-sentence preservation are necessary.
- The text calls 0.851 and 0.655 short/long chrF ratios, but these approximately match long/short ratios from the table.
- The current scorer stores n-gram precisions as `bleu_1`, `bleu_2`, and `bleu_3`, while `bleu_4` is corpus BLEU. These are different quantities; use accurate labels before making external comparisons.

This supports retiring PGAT as the main proposed contribution. It does not prove an entire architecture family unsuitable.

## Literature that changes the choice

The following are primary-source papers or author repositories. Where a venue is reported only by arXiv, that qualification is explicit. Paper scores have not been rescored using this project's protocol.

| Work | Relevant finding or method | Implication for novelty |
|---|---|---|
| [A Critical Study of Automatic Evaluation in SLT, LREC 2026](https://aclanthology.org/2026.lrec-1.749/) | Studies paraphrases, hallucinations, and sentence-length effects on evaluation. | Another five-bin BLEU analysis alone is insufficient. This is the correct title for the Yazdani citation in the project. |
| [BoostSLT, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Han_BoostSLT_Boosting_Sign_Language_Translation_via_a_Plug-and-Play_Diffusion-Based_Semantic_CVPR_2026_paper.pdf) | Motion-energy segmentation, fragment translation, and diffusion reconstruction target long sequences. | Split–translate–combine is already a direct competitor, not a new thesis by itself. [Author code](https://github.com/K1sna/BoostSLT) is available, but was not executed here. |
| [Grounding or Guessing?, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/35cc4ccb3fb0c5eb6f107f877a4cd57e-Abstract-Conference.html) | Uses visual sensitivity and counterfactual probabilities to estimate translation reliability. | Generic hallucination detection or shuffled-video scoring is occupied. Its [full paper](https://arxiv.org/html/2510.18439v2) also discusses omissions; do not claim it ignores them. |
| [SAGE, ICCV workshops 2025](https://openresearch.surrey.ac.uk/esploro/outputs/conferenceProceeding/SAGE-Segment-Aware-Gloss-Free-Encoding-for-Token-Efficient/991013766502346) | Segment visual tokens and token-level alignment; [method text](https://arxiv.org/html/2507.09266v1) includes text-derived pseudo-glosses. | Adaptive semantic tokenization and local alignment are established directions. Audit upstream supervision before strict gloss-free comparisons. |
| [Text CTC Alignment, COLING 2025](https://aclanthology.org/2025.coling-main.219.pdf) | Joint CTC/attention with hierarchical reordering. Its limitations disclose gloss-supervised pretrained sign embeddings. | Text CTC is a valuable comparator, but its reported setting is not strictly gloss-free throughout the pipeline. |
| [Think in Latent Thoughts / SignThought](https://arxiv.org/abs/2604.15301) | ArXiv reports ACL 2026 Main acceptance; latent planning and grounded evidence aggregation include length-bucket experiments. | Adding latent planning and claiming long-sentence gains has a close competitor. |
| [VTaMo](https://arxiv.org/abs/2607.09126) | ArXiv reports ECCV 2026 acceptance; [method](https://arxiv.org/html/2607.09126v1) uses optimal transport, null assignments, and alignment. | Unordered soft alignment plus a null slot is not sufficient novelty. |
| [SpaMo, NAACL 2025](https://aclanthology.org/2025.naacl-long.197/) | Frozen spatial/motion features and an LLM, with visual-text warm-up. | Relevant reproducible backbone candidate given the existing feature experience. |
| [GloFE, 2023 author manuscript](https://arxiv.org/abs/2305.12876) | Text-derived concepts guide visual cross-attention and contrastive learning. | Simply predicting content words or semantic concepts is already prior art. |
| [Conditional Sentence Generation and Cross-Modal Reranking, IEEE TMM](https://ieeexplore.ieee.org/document/9447976/) | Word-existence prediction, conditional generation, and cross-modal candidate selection; online publication 2021, volume 24. | A concept list followed by an LLM and reranking is not a new contribution by itself. |
| [Lost in Translation, Found in Context, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Jang_Lost_in_Translation_Found_in_Context_Sign_Language_Translation_with_CVPR_2025_paper.html) | Adds contextual text and earlier translations on BOBSL and How2Sign. | Generic discourse context is occupied; a new topic must isolate phenomena such as referent preservation. |
| [RVLF, author manuscript](https://arxiv.org/abs/2512.07273) | Visual/skeleton fusion and GRPO using BLEU and ROUGE rewards. | RL or a completeness-named overlap reward is insufficient novelty. |

## Lessons from other domains

| Published work | Transferable idea | Necessary SLT adaptation |
|---|---|---|
| [Modeling Coverage for Neural Machine Translation, ACL 2016](https://aclanthology.org/P16-1008/) | Track what has already been translated to address under/over-translation. | Frames are not words; attention totals do not establish semantic coverage. Track meaningful source evidence with a null/background option and permit many-to-many alignment. |
| [FActScore, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.741/) | Evaluate supported atomic claims rather than one sentence-level judgment. | Add source-content recall as well as generated-content precision. The evidence must be signing, not external world knowledge. |
| [On Compositional Generalization of NMT, ACL 2021](https://aclanthology.org/2021.acl-long.368/) | Hold out combinations of familiar elements and measure failures hidden by standard benchmarks. | Build relation/combination splits over naturally signed videos; control lexical rarity, signer, and duration. |
| [CIF, ICASSP 2020](https://ieeexplore.ieee.org/document/9054250) | Accumulate input evidence into units rather than taking fixed-duration samples. | Do not assume a monotonic one-sign/one-spoken-word mapping. Use source units followed by a decoder able to reorder. |
| [DIBS, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_DIBS_Enhancing_Dense_Video_Captioning_with_Unlabeled_Videos_via_Pseudo_CVPR_2024_paper.html) | Refine uncertain temporal boundaries using weak supervision. | Signing has grammar and simultaneous non-manual information; motion boundaries alone cannot certify independent meaning. |

These transfers motivate hypotheses; none establishes that the proposed SLT method will work.

## Ranked thesis directions

### 1. Source-grounded omission detection and recovery — preferred

**Question:** Can we identify source meanings absent from a translation and recover them while holding unsupported additions constant or reducing them?

**Proposed title:** What Was Left Untranslated? Source-Grounded Semantic Coverage in Long Sign Language Translation.

Start with meaning units and relations: weather event, location, time, temperature, polarity, and which value belongs to which event/location. This is a PHOENIX prototype schema, not a universal account of sign language. On a second dataset, use its relevant propositions and relations.

An illustrative example: a video communicates rain in the north, clearing in the south, and temperatures falling overnight. A translation reports only northern rain. Every generated statement may be correct, yet most source meaning is missing. A confidence score on generated tokens alone cannot enumerate all that missing information.

The proposed technical contribution is a **residual coverage estimator conditioned on both source video and a draft translation**, followed by bounded retrieval and repair:

1. Keep temporally addressable source features from a competent baseline.
2. Generate an ordinary draft.
3. Estimate which source propositions are not expressed, retaining uncertainty and possible null evidence.
4. Re-examine a limited number of associated video windows with neighboring context.
5. Propose a small edit; retain it only when the visual verification score supports the recovered claim and its relations. Otherwise keep the draft.

An initial implementation can use window–proposition scores plus a draft entailment model, with residual score `source_support(c) * (1 - draft_entails(c))`. These are learned proxies, not calibrated probabilities by default. Candidates must be proposed from video at inference; reference-derived candidates are an oracle experiment only. An unordered word inventory cannot distinguish reversed temperatures, swapped locations, or negation scope.

Training can start with train-only draft corruptions: remove one proposition, change a value, or swap a relation. Compare each against its uncorrupted video and use same-domain hard negatives. Include naturally occurring out-of-fold translation errors so the estimator does not learn synthetic deletion artifacts. Human review is needed for both proposition validity and visual support. Do not treat pseudo labels as gold.

**Novelty boundary:** propose and validate source-conditioned *missing-content localization and selective recovery*. Do not claim novelty for coverage, hallucination scoring, semantic concepts, iterative editing, or chunking individually. Direct competitors are BoostSLT and Grounding or Guessing; concept-based generation and ordinary coverage are necessary simpler baselines. There is no claim here that the complete combination is conclusively the first.

**Main risk:** the visual backbone may not recognize the missing content at all. A repair system cannot reliably recover evidence absent from its representation. Its verifier may also share the translator's errors. Validate the verifier against independent human judgments and another backbone.

**Publication case:** a released, human-validated omission benchmark plus reliable recovery across two datasets/backbones would offer both a reusable evaluation contribution and a method. It need not win every overall BLEU comparison, but it must beat strong alternatives on the claimed capability without hiding quality or compute costs.

### 2. Compositional generalization and the causes of the length cliff — strongest alternative

**Question:** Does degradation track duration, number of propositions, unseen combinations, or target wording?

Create complementary tests: modest semantics-preserving resampling of the same video; independently verified concise/verbose references; naturally signed short/long examples matched for vocabulary and signer; and held-out combinations of familiar concepts/relations. Use original split boundaries for the standard evaluation and create a separate, documented compositional track without test-driven selection.

Candidate intervention: train-only composition-consistency supervision and balanced exposure to rare relations. Do not assume concatenating two unrelated clips creates a natural signed sentence. Artificial concatenations are stress tests, reported separately.

**Contribution:** controlled splits, relation accuracy, statistical controls, and a reproducible explanation that distinguishes rival mechanisms. Another length-binned BLEU table is not enough. If natural-data matching lacks common support, report the limitation instead of claiming causality.

**Risk:** annotation and split construction are difficult; weather-domain templates may yield very small held-out sets. Less GPU-intensive than developing a new large model. This becomes the first choice if recovery feasibility fails but the controlled failure pattern is robust.

### 3. Semantic completeness evaluation — standalone benchmark/metric thesis

Build a challenge set with omission, unsupported addition, substitution, relation reversal, and meaning-preserving paraphrase. Measure source-content recall and supported-output precision separately; validate model rankings and error sensitivity against qualified sign-language judgments.

Novelty must exceed the LREC evaluation study and ICLR reliability measure through validated source-side omissions, relational meaning, and generalization across systems. Do not use the same model for pseudo annotation, training, and final truth judgments. A reference-only version can evaluate reference agreement, but cannot certify complete source meaning.

### 4. Targeted visual re-observation under a fixed compute budget

Allocate a second visual pass to uncertain hands/face/time intervals, rather than uniformly increasing all visual samples. Compare at equal encoder calls, latency, and total tokens with random windows, uniform denser sampling, and whole-video second passes.

The research question is whether independent visual evidence resolves specific lost propositions. This differs from Source32's global density change, but adaptive sampling itself is not enough novelty. It depends on calibrated uncertainty and a trustworthy error-localization signal; narrower and more feasible as the intervention in direction 1.

### 5. Relation and non-manual scope preservation

Study whether longer utterances lose which event a negation, location, time expression, or facial marker modifies. Build minimal contrasts and test structured relation supervision against bag-of-concepts baselines. Potentially substantial linguistic novelty, but the current mouth/global feature cache cannot be assumed to retain eyebrow, gaze, head, or discourse-locus information. Requires appropriate video quality, expert annotation, and often more suitable data than PHOENIX alone.

### 6. Weakly supervised source-unit discovery with reordering

Use a CIF-inspired learned accumulator to form units, then unrestricted linguistic composition, and evaluate invariance to signing rate. Strong competitors include SAGE, Text CTC, VTaMo, and latent planning. This is technically plausible but the weakest novelty-to-effort ratio unless a specifically new constraint or theoretical result can be demonstrated.

## First experiments and decision gates

The numbers below are proposed pilot sizes and practical decision criteria, not power calculations or promised results.

**Weeks 1–2: diagnosis and oracle feasibility.** Recover existing prediction files and verify sample-ID/reference matching. Select roughly 100–150 DEV examples spanning lengths, using a fixed random sampling plan. Annotate source-supported propositions and relations, omissions, additions, and acceptable paraphrases. Have two qualified annotators review an overlapping portion; report disagreement and adjudication. German text knowledge alone is insufficient for DGS source verification.

Use PGAT for inexpensive pipeline development, then repeat on at least one competent released baseline before generalizing. Evaluate long clips whole and with contextual windows. An oracle selector may use DEV references/annotations to test whether correct missing content exists among window outputs. Clearly distinguish this upper bound from an inference system.

Proceed with recovery only if a meaningful pool of human-confirmed omissions is recoverable from windows and an automatic verifier can discriminate correct from incorrect additions better than confidence/length controls. If the oracle finds little recoverable content, prioritize the representation problem or direction 2. If oracle recovery works but selection fails, omission verification itself may become the thesis.

**Weeks 3–6: minimum method.** Train one small residual estimator and a bounded edit mechanism on cached features where feasible. Compare ordinary decoding, length adjustment, NMT-style coverage, text-only repair, random re-observation, uniform re-observation, BoostSLT, and grounding-based reranking. Keep a shared backbone and account for extra inference budget. A pure reranking stage cannot recover content absent from every candidate; measure candidate-pool recall.

**Weeks 7–10: generalization.** Test on a second dataset, such as CSL-Daily if access, storage, features, and annotator support permit. Extract features incrementally to respect the quota. Run more than one seed for trained comparisons (aim for three), and test a second architecture to check whether the result merely patches PGAT.

**Weeks 11–12: frozen evaluation and writing.** Lock thresholds on DEV and evaluate independent held-out data. PHOENIX TEST has already influenced project selection, so disclose that history; do not present it as wholly unseen evidence. Use an independent held-out dataset/track to strengthen confirmation. Actual time depends on annotation access and baseline reproduction.

Report semantic recall, supported-claim precision, omission-recovery precision/recall, relation accuracy, and human-rated adequacy alongside corpus BLEU/chrF, generation length, runtime, GPU memory, and abstention/no-edit rate. Bootstrap at video level; use recording-level clusters where dependent clips are identifiable. Include bin counts and uncertainty, especially for the 16-example longest PHOENIX bin. Do not optimize the short/long ratio alone.

**Commitment decision:** choose direction 1 if the pilot validates recoverable omissions and independent verification. Choose direction 2 if the strongest evidence concerns unseen composition or if source-grounded repair is not feasible. Without qualified sign-language evaluation, narrow claims to reference-based completeness; do not advertise visually verified semantic coverage.

No training, remote cluster work, baseline execution, or code changes were performed for this review. The only project addition is this research note. The existing architecture document has not been rewritten.
