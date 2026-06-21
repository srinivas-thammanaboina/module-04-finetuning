"""Stage 2 — Prompt 2.2: generate the training set by distillation.

We sample real AAPL/NVDA chunks and ask Claude (Sonnet) to write a
{question, ideal_answer} pair grounded ONLY in those chunks, imitating the gold
examples in format_spec.py. ~20% of examples are deliberately UNANSWERABLE so the
model learns to refuse cleanly instead of guessing.

Output: data/<out>.jsonl, one chat example per line:
    {"messages": [system, user, assistant], "meta": {...}}

Key guarantees:
  - TSLA is NEVER sampled (hard assert) — it is the held-out eval company.
  - Deterministic chunk sampling given the seed.
  - Every API response is cached to disk, so re-runs are free and don't re-bill.

Run (DRY RUN first — 30 examples to a separate file):
    python scripts/generate_dataset.py --n 30 --out data/train_dryrun.jsonl

You run this (it spends API budget). We review the summary + samples together.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from format_spec import HOUSE_FORMAT_INSTRUCTIONS, GOLD_EXAMPLES

# Load .env (for ANTHROPIC_API_KEY) without modifying it, if python-dotenv exists.
try:
    from dotenv import load_dotenv
    load_dotenv(config.__file__.replace("config.py", ".env"))
except ImportError:
    pass

from anthropic import Anthropic

# --- Settings ----------------------------------------------------------------
GEN_MODEL = "claude-sonnet-4-6"   # generator (user choice: Sonnet to cut cost)
MAX_TOKENS = 1024
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache"

# Question-type mix (fractions of total). Sums to 1.0. (Decisions 1 & 2 approved.)
TYPE_MIX = {
    "factual_lookup": 0.30,
    "about_x": 0.25,
    "comparison": 0.15,
    "numeric": 0.10,
    "refusal": 0.20,
}

# A citation looks like [AAPL-2025-10-31-0000]; this pulls the ids out of an answer.
CITE_RE = re.compile(r"\[([A-Z]{2,6}-\d{4}-\d{2}-\d{2}-\d+)\]")

client = Anthropic()


# -----------------------------------------------------------------------------
# Chunk loading (READ-ONLY; train tickers only — TSLA must never appear)
# -----------------------------------------------------------------------------
def load_train_chunks():
    assert config.EVAL_TICKER not in config.TRAIN_TICKERS, "TSLA leaked into TRAIN!"
    chunks = {}
    for ticker in config.TRAIN_TICKERS:
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
                # Hard guard: nothing TSLA may enter the training pool.
                assert not d["chunk_id"].startswith(config.EVAL_TICKER), \
                    f"TSLA chunk found in {ticker} file: {d['chunk_id']}"
                rows.append({
                    "chunk_id": d["chunk_id"],
                    "ticker": ticker,
                    "section": meta.get("section"),
                    "chunk_index": meta.get("chunk_index"),
                    "text": d.get("text", ""),
                })
        chunks[ticker] = rows
    return chunks


# -----------------------------------------------------------------------------
# Sampling: decide the type plan, then pick chunks suited to each type
# -----------------------------------------------------------------------------
def build_type_plan(n, rng):
    """Turn the TYPE_MIX fractions into a concrete shuffled list of n types."""
    counts = {t: round(frac * n) for t, frac in TYPE_MIX.items()}
    # Fix rounding drift so the counts sum to exactly n.
    while sum(counts.values()) > n:
        counts[max(counts, key=counts.get)] -= 1
    while sum(counts.values()) < n:
        counts[max(counts, key=counts.get)] += 1
    plan = [t for t, c in counts.items() for _ in range(c)]
    rng.shuffle(plan)
    return plan


def _has_digit(text):
    return any(ch.isdigit() for ch in text)


def select_chunks(qtype, chunks, rng):
    """Pick the chunk(s) for one example, suited to the question type."""
    flat = [c for rows in chunks.values() for c in rows]

    if qtype == "comparison":
        # 1-in-4 comparisons are cross-company; the rest synthesize within one
        # filing using genuinely related chunks (same section, near indices).
        if len(chunks) >= 2 and rng.random() < 0.25:
            a = rng.choice(chunks[config.TRAIN_TICKERS[0]])
            b = rng.choice(chunks[config.TRAIN_TICKERS[1]])
            return [a, b]
        anchor = rng.choice(flat)
        same = [c for c in chunks[anchor["ticker"]]
                if c["section"] == anchor["section"] and c["chunk_id"] != anchor["chunk_id"]]
        rng.shuffle(same)
        group = [anchor] + same[:rng.choice([1, 2])]
        return group if len(group) >= 2 else [anchor, rng.choice(flat)]

    if qtype == "numeric":
        # Prefer a chunk that actually contains digits, so a numeric question
        # has a real number to ask about and cite.
        numeric_pool = [c for c in flat if _has_digit(c["text"])] or flat
        return [rng.choice(numeric_pool)]

    # factual_lookup, about_x, refusal → a single chunk.
    return [rng.choice(flat)]


# -----------------------------------------------------------------------------
# Prompt construction + cached generation
# -----------------------------------------------------------------------------
def format_context(chunks):
    """Render chunks the way the trained model will see them in the user turn."""
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['chunk_id']}] ({c['section']})\n{c['text']}")
    return "\n\n".join(blocks)


def gold_examples_block():
    lines = []
    for ex in GOLD_EXAMPLES:
        ctx = format_context(ex["retrieved_context"])
        lines.append(
            f"--- GOLD EXAMPLE ({ex['type']}) ---\n"
            f"CONTEXT:\n{ctx}\n\nQUESTION: {ex['question']}\n\n"
            f"IDEAL_ANSWER: {ex['ideal_answer']}"
        )
    return "\n\n".join(lines)


TYPE_HINTS = {
    "factual_lookup": "A factual lookup: a direct question about a specific fact "
                      "stated in the context.",
    "about_x": "A 'what does the filing say about X' question: ask what the filing "
               "says about a topic present in the context, and summarize it.",
    "comparison": "A comparison/synthesis question across the provided chunks. Cite "
                  "EACH claim to the specific chunk it came from.",
    "numeric": "A numeric question about a figure that appears in the context. Cite "
               "the chunk the number came from.",
}


def build_gen_prompt(qtype, chunks):
    context = format_context(chunks)
    header = (
        "You are creating ONE training example for a Filing Analyst Copilot "
        "finetune. The answer must follow this house format exactly:\n\n"
        f"{HOUSE_FORMAT_INSTRUCTIONS}\n\n"
        "Study these gold examples — match their style precisely:\n\n"
        f"{gold_examples_block()}\n\n"
        f"RETRIEVED CONTEXT for the new example:\n{context}\n\n"
    )
    if qtype == "refusal":
        task = (
            "Write a {question, ideal_answer} pair where the question is PLAUSIBLE "
            "for this company's 10-K but CANNOT be answered from the provided "
            "context (the needed information is simply not present in these "
            "chunks). The ideal_answer must be a clean refusal in the house "
            "format: state in the first person that the provided context does not "
            "contain the information, do NOT guess, and include NO citation."
        )
    else:
        task = (
            f"Write a {{question, ideal_answer}} pair. {TYPE_HINTS[qtype]} "
            "The question must be answerable ENTIRELY from the provided context. "
            "The ideal_answer must follow the house format: prose first, every "
            "factual claim cited inline as [chunk_id] using ONLY the chunk_ids "
            "above, first-person voice."
        )
    return header + task + (
        '\n\nRespond with ONLY a JSON object, no prose, no code fence:\n'
        '{"question": "...", "ideal_answer": "..."}'
    )


def cached_generate(prompt):
    """Call the API, but reuse a cached response if we've seen this exact prompt."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((GEN_MODEL + "\n" + prompt).encode()).hexdigest()[:24]
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())["response"], True
    resp = client.messages.create(
        model=GEN_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    cache_path.write_text(json.dumps({"prompt": prompt, "response": text}))
    return text, False


def parse_pair(text):
    """Pull {question, ideal_answer} out of the model's reply, tolerating fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    obj = json.loads(text[start:end + 1])
    return obj["question"], obj["ideal_answer"]


# -----------------------------------------------------------------------------
# Validation (deterministic) — does the answer obey our citation rules?
# -----------------------------------------------------------------------------
def validate(qtype, ideal_answer, provided_ids):
    cited = set(CITE_RE.findall(ideal_answer))
    if qtype == "refusal":
        ok = len(cited) == 0          # a refusal must cite nothing
    else:
        ok = bool(cited) and cited <= provided_ids   # only real, provided ids
    return ok, cited


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=config.TARGET_TRAIN_EXAMPLES)
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    chunks = load_train_chunks()
    print(f"Loaded train chunks: " +
          ", ".join(f"{t}={len(chunks[t])}" for t in config.TRAIN_TICKERS))

    plan = build_type_plan(args.n, rng)
    out_path = Path(__file__).resolve().parent.parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples, cache_hits, invalid = [], 0, 0
    for i, qtype in enumerate(plan, 1):
        sel = select_chunks(qtype, chunks, rng)
        provided_ids = {c["chunk_id"] for c in sel}
        prompt = build_gen_prompt(qtype, sel)
        raw, hit = cached_generate(prompt)
        cache_hits += hit
        try:
            question, ideal = parse_pair(raw)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  [{i}/{args.n}] {qtype}: PARSE FAILED ({e}) — skipped")
            continue
        ok, cited = validate(qtype, ideal, provided_ids)
        invalid += (not ok)
        context_str = format_context(sel)
        examples.append({
            "messages": [
                {"role": "system", "content": HOUSE_FORMAT_INSTRUCTIONS},
                {"role": "user",
                 "content": f"Context:\n{context_str}\n\nQuestion: {question}"},
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
        print(f"  [{i}/{args.n}] {qtype:14s} {'OK ' if ok else 'BAD'} "
              f"{'+'.join(sorted({c['ticker'] for c in sel}))}")

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # --- Summary (Rule 5: make the dataset visible) --------------------------
    by_type = Counter(ex["meta"]["type"] for ex in examples)
    by_ticker = Counter("+".join(ex["meta"]["tickers"]) for ex in examples)
    n = len(examples)
    refusals = by_type.get("refusal", 0)
    print("\n" + "=" * 60)
    print(f"Wrote {n} examples → {args.out}")
    print(f"Cache hits: {cache_hits}/{len(plan)} (0 = all fresh API calls)")
    print(f"By type:   {dict(by_type)}")
    print(f"By ticker: {dict(by_ticker)}")
    print(f"Refusals:  {refusals}/{n} = {100*refusals/n:.0f}%")
    print(f"Citation-valid: {n - invalid}/{n} "
          f"(invalid = wrong/missing/fabricated citations to review)")
    print("=" * 60)


if __name__ == "__main__":
    main()
