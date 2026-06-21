"""Stage 1 — inspect the source chunks (diagnostic only, READ-ONLY).

We look at the real Module 2 chunk files before designing any dataset. The whole
citation format depends on what fields actually exist on each chunk, so we
confirm that here — cheaply, by reading — instead of discovering a mismatch after
generating 200 training examples.

This script writes nothing. It only prints what we have.

Run:  python scripts/inspect_chunks.py
"""

import json
import statistics
import sys
from pathlib import Path

# Import the shared constants (paths, tickers). config.py lives one level up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Our citation token, per the Stage 1 finding: each chunk carries a top-level,
# globally-unique `chunk_id` (e.g. "AAPL-2025-10-31-0000"). We confirm it exists
# on every chunk so a citation can always point at a real source.
CITATION_FIELD = "chunk_id"

ALL_TICKERS = config.TRAIN_TICKERS + [config.EVAL_TICKER]
PREVIEW_CHARS = 220  # how much chunk text to show in examples


def load_chunks(ticker):
    """Load one ticker's JSONL into a list of dicts. Read-only."""
    path = config.CHUNKS_DIR / f"{ticker}.jsonl"
    if not path.exists():
        print(f"  !! MISSING FILE: {path}")
        return []
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def get_metadata(chunk):
    """Return the metadata dict. It is normally a real dict, but tolerate the
    case where it was stored as a JSON string."""
    m = chunk.get("metadata", {})
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except json.JSONDecodeError:
            m = {"_unparsed_metadata": m}
    return m if isinstance(m, dict) else {}


def inspect_ticker(ticker):
    print(f"\n{'=' * 70}\n{ticker}\n{'=' * 70}")
    chunks = load_chunks(ticker)
    if not chunks:
        return 0

    # --- Counts ---
    print(f"chunk count: {len(chunks)}")

    # --- Top-level keys + metadata keys on a representative chunk ---
    first = chunks[0]
    print(f"top-level keys: {list(first.keys())}")
    meta_keys = list(get_metadata(first).keys())
    print(f"metadata keys: {meta_keys}")

    # --- The gate check: does our citation token exist on EVERY chunk? ---
    missing_cite = [i for i, c in enumerate(chunks) if not c.get(CITATION_FIELD)]
    if missing_cite:
        print(f"  !! {len(missing_cite)} chunks MISSING '{CITATION_FIELD}' "
              f"(citation token) — first at index {missing_cite[0]}")
    else:
        print(f"  OK: every chunk has a '{CITATION_FIELD}' citation token "
              f"(e.g. {first[CITATION_FIELD]!r})")

    # --- Chunk-length distribution (in characters) ---
    lengths = [len(c.get("text", "")) for c in chunks]
    print(f"text length (chars): min={min(lengths)}  "
          f"median={int(statistics.median(lengths))}  max={max(lengths)}")

    # --- Sections represented (if section metadata exists) ---
    sections = [get_metadata(c).get("section") for c in chunks]
    sections = [s for s in sections if s]
    if sections:
        counts = {}
        for s in sections:
            counts[s] = counts.get(s, 0) + 1
        print(f"sections represented ({len(counts)} distinct):")
        for s, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {n:4d}  {s}")
    else:
        print("sections represented: (no 'section' metadata found)")

    # --- 2 example chunks (truncated) so we see the real shape ---
    print("\nexamples:")
    for c in chunks[:2]:
        meta = get_metadata(c)
        text = c.get("text", "").replace("\n", " ")
        preview = text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else "")
        print(f"  - {CITATION_FIELD}: {c.get(CITATION_FIELD)!r}")
        print(f"    section: {meta.get('section')!r}  "
              f"chunk_index: {meta.get('chunk_index')!r}")
        print(f"    text: {preview}")

    return len(chunks)


def main():
    print(f"Reading chunks from: {config.CHUNKS_DIR}")
    print(f"Train tickers: {config.TRAIN_TICKERS}   Held-out: {config.EVAL_TICKER}")
    total = 0
    for ticker in ALL_TICKERS:
        total += inspect_ticker(ticker)
    print(f"\n{'=' * 70}")
    print(f"TOTAL chunks across all tickers: {total}")
    print("(diagnostic only — nothing was written)")


if __name__ == "__main__":
    main()
