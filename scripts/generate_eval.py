"""Stage 2 — Prompt 2.3: build the HELD-OUT eval set from TSLA only.

Same recipe as generate_dataset.py (same house format, same gold examples, same
question-type mix incl. refusals) — but every example is drawn ENTIRELY from TSLA
chunks. This set is the sole judge of whether the finetune learned the FORMAT or
just memorized the AAPL/NVDA training filings. TSLA is never in train.jsonl.

We deliberately reuse the train generator's building blocks so eval examples have
exactly the same shape as training examples — only the source company differs.

Run (you run it — spends API):
    python scripts/generate_eval.py --n 40 --out data/eval.jsonl
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from format_spec import HOUSE_FORMAT_INSTRUCTIONS

# Reuse the validated generator internals — identical example shape.
from generate_dataset import (
    build_type_plan, select_chunks, build_gen_prompt, cached_generate,
    parse_pair, validate, format_context,
)


def load_eval_chunks():
    """Load ONLY the held-out TSLA chunks. Hard-assert it is the eval ticker."""
    ticker = config.EVAL_TICKER
    assert ticker not in config.TRAIN_TICKERS, "EVAL ticker overlaps TRAIN!"
    path = config.CHUNKS_DIR / f"{ticker}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            meta = d.get("metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            assert d["chunk_id"].startswith(ticker), \
                f"Non-{ticker} chunk in eval file: {d['chunk_id']}"
            rows.append({
                "chunk_id": d["chunk_id"],
                "ticker": ticker,
                "section": meta.get("section"),
                "chunk_index": meta.get("chunk_index"),
                "text": d.get("text", ""),
            })
    return {ticker: rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="data/eval.jsonl")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    chunks = load_eval_chunks()
    print(f"Loaded EVAL chunks: {config.EVAL_TICKER}={len(chunks[config.EVAL_TICKER])}")

    plan = build_type_plan(args.n, rng)
    out_path = Path(__file__).resolve().parent.parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples, cache_hits, invalid = [], 0, 0
    for i, qtype in enumerate(plan, 1):
        sel = select_chunks(qtype, chunks, rng)  # single-ticker → within-filing only
        provided_ids = {c["chunk_id"] for c in sel}
        raw, hit = cached_generate(build_gen_prompt(qtype, sel))
        cache_hits += hit
        try:
            question, ideal = parse_pair(raw)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  [{i}/{args.n}] {qtype}: PARSE FAILED ({e}) — skipped")
            continue
        ok, cited = validate(qtype, ideal, provided_ids)
        invalid += (not ok)
        examples.append({
            "messages": [
                {"role": "system", "content": HOUSE_FORMAT_INSTRUCTIONS},
                {"role": "user",
                 "content": f"Context:\n{format_context(sel)}\n\nQuestion: {question}"},
                {"role": "assistant", "content": ideal},
            ],
            "meta": {
                "type": qtype,
                "tickers": sorted({c["ticker"] for c in sel}),
                "chunk_ids": sorted(provided_ids),
                "cited": sorted(cited),
                "valid": ok,
            },
        })
        print(f"  [{i}/{args.n}] {qtype:14s} {'OK ' if ok else 'BAD'}")

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    by_type = Counter(ex["meta"]["type"] for ex in examples)
    n = len(examples)
    refusals = by_type.get("refusal", 0)
    print("\n" + "=" * 60)
    print(f"Wrote {n} HELD-OUT eval examples (TSLA only) → {args.out}")
    print(f"Cache hits: {cache_hits}/{len(plan)}")
    print(f"By type:  {dict(by_type)}")
    print(f"Refusals: {refusals}/{n} = {100*refusals/n:.0f}%" if n else "no examples")
    print(f"Citation-valid: {n - invalid}/{n}")
    print("=" * 60)


if __name__ == "__main__":
    main()
