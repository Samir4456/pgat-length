# SSM-Adapter Proposal Package

Files here match the structure of the P3 proposal (`P3_st125989_RMc.pdf`) so they slot into the same LaTeX template with minimal edits. Intended audience: your professor at tomorrow's meeting.

## What to hand over

- **`ONE_PAGER.md`** — the summary to open the meeting with. Contains the motivation, what was ruled out, the proposed method, the cross-backbone design, and the three questions to ask the professor for feedback. This is the fastest-to-read document; print it and hand it over first.
- **`abstract.tex`** — one-page abstract in the same style as your P3 abstract.
- **`introduction.tex`** — Background, motivation, problem statement, research gap, research questions and objectives, scope and limitations. Same six-section structure as your P3 introduction.
- **`methodology.tex`** — Research approach, system overview, dataset and input representation, backbone choice, preprocessing pipeline, adapter design (with math), integration into pgat-length, integration into TSPNet, training procedure, cross-backbone verification, limitations. Same six-plus-section structure as your P3 methodology.
- **`experimental_plan.tex`** — Baselines and model variants (E0–E6), ablations, metrics and statistical reporting, feasibility and compute plan, risk management, decision rules and contingency plan, thesis timeline. Same section structure as your P3 experimental plan.

## To compile into a PDF

Fastest path: copy these `.tex` files into your existing P3 LaTeX project as replacements, drop `abstract.tex` / `introduction.tex` / `methodology.tex` / `experimental_plan.tex` over the P3 versions, adjust `references.bib` for any new citations, and rebuild. The `aitthesis.cls` and the master `thesis.tex` do not need to change.

Alternatively, keep the P3 project untouched and clone a copy for this proposal so both drafts survive.

## What is not included

- **Literature review** — not rewritten here. The P3 literature review is a strong foundation; extend it with 4–6 references on state-space models (S4, Mamba, ViS4mer, Video Mamba Suite) and a note on the length-cliff literature (Yazdani 2026, Hamidullah 2024). Adding this is a 1–2 hour job.
- **Cover, declaration, acknowledgements, appendix, conclusion, contents, LoFT** — reuse from the P3 project unchanged.
- **Figures** — the methodology figure (`method_placeholder.png`) and timeline figure (`timeline_placeholder.png`) referenced in the .tex files need drawing. For tomorrow's meeting, either sketch by hand on a whiteboard or delete the figure blocks and describe verbally.
- **References** — new citations (Gu 2022 for S4, Gu 2023 for Mamba, Li 2020 for TSPNet, Yazdani 2026, Hamidullah 2024) need bib entries added to `references.bib`.

## Prep for the meeting

Three things to have ready before you walk in:

1. **The ONE_PAGER**, printed. Hand it over first.
2. **The current experimental status** — pgat-length adapter results (BLEU-4 7.61 vs baseline 7.36; chrF long/short ratio 0.807 vs 0.772); TSPNet baseline training in progress; TSPNet adapter variant queued. State clearly what has landed and what is still running.
3. **The three questions** at the bottom of the ONE_PAGER. Ask the professor for direction on each one before the meeting ends. These are the load-bearing scope decisions.
