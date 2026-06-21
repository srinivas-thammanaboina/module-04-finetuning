# Stage 2 — Dataset generation

**Takeaway:** Three hand-written gold examples + a plain-language house-format spec
are the north star the generator imitates. They use REAL AAPL/NVDA chunks, cite
inline as `[chunk_id]`, refuse cleanly in the first person, and define the exact
shape that gets baked 200x into the weights.

## Intuition / mental model

We do not hand-write 200 answers. We hand-write 3 perfect ones and tell a strong
model (Claude) "make 200 more like these" — this is **distillation** (a capable
model drafts the ideal answers our small model learns to imitate). The generator
copies the *shape* of the seed examples, so the seeds must be exactly right: a
flaw in the seed is amplified 200x and baked permanently into the weights.

## Why the naive approach fails (concrete)

Toy/placeholder gold examples would teach the generator a format divorced from the
real material. We instead seed with real chunks the generator will actually work
with: `AAPL-2025-10-31-0000` (Apple business), `NVDA-2026-02-25-0000` (NVIDIA
business), and `NVDA-2026-02-25-0068` (NVIDIA risk-factors summary, used for the
honest refusal — it genuinely has no revenue figure to cite).

## Chosen design + tradeoffs (Prompt 2.1 — format_spec.py)

- **`HOUSE_FORMAT_INSTRUCTIONS`** = single source of truth for the system prompt,
  reused at generation, baseline, and serving. Six rules: prose first; cite every
  claim inline as `[chunk_id]`; only cite provided chunks; refuse cleanly with no
  citation when unsupported; no outside facts; first-person voice.
- **Citation style: inline `[chunk_id]` per claim**, not a trailing sources list —
  the lesson is that *every claim* is grounded, and inline makes that checkable.
- **Refusal shape:** consistent template ("The provided context does not contain
  information about X ... I cannot answer this from the provided context"). A
  recognizable shape is what the Stage 6 eval keys on to detect refuse-vs-guess.
- **Tone: first person "I"** (user decision) — made explicit as rule 6 so all
  generated examples stay consistent.

## Three gold example types

1. `factual_lookup` — single chunk, direct answer, citation per claim.
2. `comparison` — two chunks, each claim cited to its own source + a synthesis line.
3. `refusal` — context genuinely lacks the answer; plain refusal, no citation.

## Design decisions baked into the code

- Citation token = top-level `chunk_id` (Stage 1 finding).
- Gold-example `retrieved_context` uses a trimmed chunk view `{chunk_id, section,
  text}` — the same shape we assemble at generation/inference time.
- TSLA never appears in this file (train-side seeds only).

## Sanity-check experiment (done)

- **Dry run (30):** read 6 across types incl. a refusal — all honestly
  answerable/unanswerable, real citations, first-person refusal. Recipe approved.
- **Full run (200 train + 40 TSLA eval):** types on target, 20% refusals in both,
  citation-valid 200/200 and 40/40. Read 3 TSLA eval examples — high quality.
- **Leakage check PASS.** Train references only AAPL/NVDA; eval only TSLA.
  - *Lesson (Rule 4):* the checker first false-flagged "AAPL: 40" in eval — exactly
    1 per example. That uniformity revealed it was scanning the boilerplate SYSTEM
    prompt, which contains an illustrative `[AAPL-2025-10-31-0000]` in the rules.
    Fixed to scan only `meta.chunk_ids` + user/assistant text. The data was always
    clean; the *checker* had the bug. Looking at the actual artifact found it fast.

## Future experiments queue

- If dry-run examples look monotonous (Risk-Factors heavy), revisit the decision
  to accept section skew and consider stratified sampling.

## Lessons to carry forward

Seed quality dominates a distillation finetune. The cheapest place to fix the
dataset is the 3 seeds; the most expensive is after 200 examples are trained in.
