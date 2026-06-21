# Data split — train/eval for the format finetune

**The one statement that matters: TSLA never appears in `train.jsonl`.** Holding
out a whole company (not random chunks) is the anti-memorization guard. If the
finetuned model formats TSLA answers correctly despite never seeing TSLA, it
learned the FORMAT; if it only does well on AAPL/NVDA content, it MEMORIZED.

## Who goes where

| Role | Tickers | Source chunks | Used for |
|------|---------|---------------|----------|
| **TRAIN** | AAPL, NVDA | 149 + 278 = 427 | `data/train.jsonl` — what the model learns from |
| **HELD-OUT EVAL** | TSLA | 251 | `data/eval.jsonl` — the sole before/after judge |

Source chunks (read-only, from Module 2):
`/Users/srinivasthammanaboina/Projects/module-02-rag-app/data/chunks/{AAPL,NVDA,TSLA}.jsonl`

## Determinism

- `RANDOM_SEED = 42` drives all chunk sampling and the question-type plan, so both
  files regenerate identically. Every API response is cached to `data/.cache/`.

## Question-type mix (same recipe for train and eval)

| Type | Fraction | train (n=200) | eval (n=40) |
|------|----------|---------------|-------------|
| factual_lookup | 30% | 60 | 12 |
| about_x | 25% | 50 | 10 |
| comparison | 15% | 30 | 6 |
| numeric | 10% | 20 | 4 |
| **refusal** | **20%** | **40** | **8** |

Refusal share (~20%) is deliberate: clean refusal on unsupported context is a
first-class correct behavior we are training in, not a failure mode. Comparison
draws are mostly within-filing; ~25% cross-company (train only — eval is TSLA-only,
so all eval comparisons are within the TSLA filing).

## Leakage guard

`scripts/check_leakage.py` fails loudly (non-zero exit) if any `TSLA-...` chunk id
appears anywhere in `train.jsonl` — checking both the structured `meta.chunk_ids`
and a raw text scan of each line. No training happens until this passes.

## Actuals (from the run, seed 42)

- **train.jsonl: 200 examples.** By example ticker: AAPL 59, NVDA 134, cross-company
  (AAPL+NVDA) 7. NVDA-heavy because it has 278 source chunks vs AAPL's 149 —
  uniform sampling favors it; fine for a FORMAT finetune. Types on target
  (factual 60 / about_x 50 / refusal 40 / comparison 30 / numeric 20).
  **Refusals: 40/200 = 20%.** Citation-valid: 200/200.
- **eval.jsonl: 40 examples (TSLA only).** Types on target; **refusals 8/40 = 20%**;
  citation-valid 40/40.
- **Leakage check: PASS ✅** — train references only AAPL/NVDA real chunks; eval
  references only TSLA. (First run threw a false positive because the checker
  scanned the boilerplate system prompt, which contains an illustrative
  `[AAPL-...]` citation; fixed to scan only meta.chunk_ids + user/assistant text.)
