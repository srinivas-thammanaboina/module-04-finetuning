"""Stage 2 — Prompt 2.3: the leakage check (the load-bearing guard).

The entire experiment rests on TSLA never appearing in training. If even one TSLA
chunk leaks into train.jsonl, a good TSLA eval result could be memorization rather
than learned format — and we'd never know. So this script fails LOUDLY (non-zero
exit) if any TSLA chunk id appears anywhere in train.jsonl.

It checks two ways (defense in depth):
  1. The structured `meta.chunk_ids` on each example (the authoritative record of
     which chunks were actually used).
  2. A text scan of the USER + ASSISTANT messages — catches a chunk id that shows
     up in the context or the answer prose even if it weren't logged in meta.

We deliberately SKIP the system message: it is identical house-format boilerplate
on every example and contains an illustrative citation (e.g. [AAPL-...]), which is
not data and must not be mistaken for a real chunk reference.

Run (you run it; no API spend):
    python scripts/check_leakage.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train.jsonl"
EVAL = ROOT / "data" / "eval.jsonl"

# Matches any chunk id, e.g. TSLA-2026-01-29-0042
ID_RE = re.compile(r"\b([A-Z]{2,6}-\d{4}-\d{2}-\d{2}-\d+)\b")


def ids_in_example(ex):
    """All chunk ids genuinely referenced by one example: the logged
    meta.chunk_ids plus any id appearing in the user/assistant text. The system
    message (house-format boilerplate with an illustrative [AAPL-...] citation) is
    intentionally excluded — it is not data."""
    ids = set(ex.get("meta", {}).get("chunk_ids", []))
    for msg in ex.get("messages", []):
        if msg.get("role") == "system":
            continue
        ids.update(ID_RE.findall(msg.get("content", "")))
    return ids


def tickers_in_file(path):
    """Return (Counter of ticker prefixes across real chunk references,
    list of TSLA leaks if this is the train file)."""
    seen = Counter()
    leaks = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            for cid in ids_in_example(ex):
                ticker = cid.split("-", 1)[0]
                seen[ticker] += 1
                if ticker == config.EVAL_TICKER and path == TRAIN:
                    leaks.append((lineno, cid))
    return seen, leaks


def main():
    if not TRAIN.exists():
        print(f"!! {TRAIN} does not exist — generate train.jsonl first.")
        sys.exit(2)

    train_seen, leaks = tickers_in_file(TRAIN)
    n_train = sum(1 for _ in open(TRAIN, encoding="utf-8"))
    print(f"train.jsonl: {n_train} examples")
    print(f"  tickers referenced: {dict(train_seen)}")

    expected = set(config.TRAIN_TICKERS)
    unexpected = set(train_seen) - expected

    if leaks:
        print(f"\n!!!!!!  LEAKAGE: {len(leaks)} TSLA chunk id(s) found in train.jsonl")
        for lineno, cid in leaks[:10]:
            print(f"    line {lineno}: {cid}")
        print("ABORT — do NOT train. Fix the generator/split before proceeding.")
        sys.exit(1)

    if unexpected:
        print(f"\n!! WARNING: unexpected tickers in train.jsonl: {unexpected}")
        print("   (expected only AAPL/NVDA) — investigate before training.")
        sys.exit(1)

    print(f"\nOK: train.jsonl references only {sorted(expected)}. "
          f"No {config.EVAL_TICKER} leakage.")

    # Sanity-confirm the eval set is the held-out company only.
    if EVAL.exists():
        eval_seen, _ = tickers_in_file(EVAL)
        n_eval = sum(1 for _ in open(EVAL, encoding="utf-8"))
        print(f"\neval.jsonl: {n_eval} examples; tickers: {dict(eval_seen)}")
        if set(eval_seen) - {config.EVAL_TICKER}:
            print(f"!! WARNING: eval.jsonl references non-{config.EVAL_TICKER} "
                  f"tickers: {set(eval_seen) - {config.EVAL_TICKER}}")
            sys.exit(1)
        print(f"OK: eval.jsonl references only {config.EVAL_TICKER} (held out).")

    print("\nLEAKAGE CHECK PASSED ✅  Safe to train.")


if __name__ == "__main__":
    main()
