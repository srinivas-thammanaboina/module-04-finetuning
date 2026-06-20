# SESSION-STATE.md

Living state for the Module 4 finetune. Read this first on every resume (Rule 7),
then give a 1–2 sentence per-task recap and state the single next step.

_Last updated: 2026-06-20_

## Where I am

**Stage 0 (Scaffold) — DONE.** Next up: **Stage 1 — inspect the source chunks.**

## Done so far

- **Stage 0 — Scaffold.** Created the folder structure (`data/`, `scripts/`,
  `eval/`, `notes/` with `.gitkeep`), `README.md` (goal + prompt→RAG→finetune
  framing), and `config.py` (constants only). Confirmed `CHUNKS_DIR` path is
  correct. No logic written yet.

## Next step

- **Stage 1 — `scripts/inspect_chunks.py`** (diagnostic only): load each ticker's
  JSONL, print per-ticker counts, the metadata keys on a chunk, chunk-length
  distribution, sections represented, and 2 example chunks each. **The gate
  question:** confirm `doc_id` + `chunk_index` (our citation tokens) actually
  exist on the real chunks before we design any dataset. I run it.

## Durable decisions

- **Base model: `llama3.2:3b-instruct` (Llama-3.2-3B-Instruct) — FIXED.** Chosen
  for a documented route at every pipeline stage (Ollama serve, HF/QLoRA train,
  llama.cpp GGUF). Must not change midway or the before/after is invalid.
- **Split: train AAPL + NVDA, hold out TSLA entirely.** Whole-company hold-out is
  the anti-memorization guard. TSLA must never enter `train.jsonl`; a leakage
  check will enforce it.
- **Seed: 42.** All sampling/splits deterministic.
- **Target dataset size: ~200 examples** (150–250), with ~20% clean-refusal cases.
- **This is a FORMAT finetune, not knowledge injection.** RAG owns all facts; the
  finetune only bakes in the house format + refusal behavior.
- **Module 02 chunks are read-only.** Never re-chunk or modify.

## Working agreement (the contract — see CLAUDE.md)

Whiteboard before code (Rule 1) · notes file per stage (Rule 2) · stop at every
🛑 gate (Rule 6) · **I run anything that spends API/trains/serves; Claude hands
exact commands (Rule 8)**.
