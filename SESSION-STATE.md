# SESSION-STATE.md

Living state for the Module 4 finetune. Read this first on every resume (Rule 7),
then give a 1–2 sentence per-task recap and state the single next step.

_Last updated: 2026-06-20_

## Where I am

**Stage 0, 1, Stage 2 Prompts 2.1 + 2.2 DONE.** Dry run (30) reviewed and
approved — high quality, honest Q/A + refusals, per-claim citation kept. Prompt
2.3 scripts written (eval gen, leakage check, SPLIT.md). Next up: **I run the
full train (200) + eval (40) + leakage check; we review + close Stage 2.**

> ACTION ON ME (user): venv active, then run in order:
>   1. `python3 scripts/generate_dataset.py --n 200 --out data/train.jsonl`
>   2. `python3 scripts/generate_eval.py --n 40 --out data/eval.jsonl`
>   3. `python3 scripts/check_leakage.py`   (must print PASSED ✅ before training)

## Done so far

- **Stage 0 — Scaffold.** Created the folder structure (`data/`, `scripts/`,
  `eval/`, `notes/` with `.gitkeep`), `README.md` (goal + prompt→RAG→finetune
  framing), and `config.py` (constants only). Confirmed `CHUNKS_DIR` path is
  correct. No logic written yet.
- **Stage 1 — Inspect chunks.** Ran `scripts/inspect_chunks.py`. Gate PASSED.
  Schema finding: chunks are `{chunk_id, text, metadata}`; there is **no
  `doc_id`** — the citation token is the top-level **`chunk_id`** (e.g.
  `TSLA-2026-01-29-0042`), so house format = `[chunk_id]`. 678 chunks total
  (AAPL 149, NVDA 278, TSLA 251); ~885-char median, fits context easily.
  Section skew toward Risk Factors (~59% of train pool) — noted for Stage 2.
  See `notes/stage-1-notes.md`.

- **Stage 2 Prompt 2.1 — `scripts/format_spec.py`.** Approved.
  `HOUSE_FORMAT_INSTRUCTIONS` (6 rules incl. first-person voice) + 3 gold
  examples on REAL AAPL/NVDA chunks (factual lookup, comparison, honest refusal).
  Citation style = inline `[chunk_id]` per claim. See `notes/stage-2-notes.md`.

## Next step

- **Stage 2 Prompt 2.2 — `scripts/generate_dataset.py`** (FIRST API SPEND; I run
  it). Distill {question, ideal_answer} pairs from AAPL/NVDA chunks imitating the
  gold examples; ~20% unanswerable→refusal; vary question types; deterministic
  (seed 42); cache API responses to disk; assert NO TSLA. Gate: generate ~30
  examples first, review summary + 5 samples (incl. a refusal) before the full
  run. Whiteboard before writing.

## Durable decisions

- **Base model: `llama3.2:3b-instruct` (Llama-3.2-3B-Instruct) — FIXED.** Chosen
  for a documented route at every pipeline stage (Ollama serve, HF/QLoRA train,
  llama.cpp GGUF). Must not change midway or the before/after is invalid.
- **Split: train AAPL + NVDA, hold out TSLA entirely.** Whole-company hold-out is
  the anti-memorization guard. TSLA must never enter `train.jsonl`; a leakage
  check will enforce it.
- **Citation token: top-level `chunk_id`; house format `[chunk_id]`.** Confirmed
  on real data in Stage 1 (no `doc_id` field exists). Must flow end to end.
- **Generator model: `claude-sonnet-4-6` (Sonnet).** User chose Sonnet over Opus
  to cut cost for the distillation. Judge model TBD at Stage 6.
- **Question-type mix:** 30% factual / 25% about-X / 15% comparison / 10% numeric
  / 20% refusal. Comparison: mostly within-filing, ~25% cross-company.
- **API responses cached to `data/.cache/` (gitignored).** Re-runs don't re-bill.
- **`.gitignore` added** before any `git init`: ignores `.env`, model artifacts
  (`*.gguf`, `adapter/`, etc.), cache. Dataset + notes stay tracked (Rule 5).
- **Seed: 42.** All sampling/splits deterministic.
- **Target dataset size: ~200 examples** (150–250), with ~20% clean-refusal cases.
- **This is a FORMAT finetune, not knowledge injection.** RAG owns all facts; the
  finetune only bakes in the house format + refusal behavior.
- **Module 02 chunks are read-only.** Never re-chunk or modify.

## Working agreement (the contract — see CLAUDE.md)

Whiteboard before code (Rule 1) · notes file per stage (Rule 2) · stop at every
🛑 gate (Rule 6) · **I run anything that spends API/trains/serves; Claude hands
exact commands (Rule 8)**.
