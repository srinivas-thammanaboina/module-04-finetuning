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

## Actuals (fill in from the run output)

- train.jsonl: ___ examples — by ticker: ___ — refusals: ___%
- eval.jsonl: ___ examples (TSLA only) — refusals: ___%
- Leakage check: ___ (PASS/FAIL)
