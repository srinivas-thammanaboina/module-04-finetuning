# Stage 1 — Inspect the source chunks

**Takeaway:** The real chunks carry a unique top-level `chunk_id` (e.g.
`AAPL-2025-10-31-0000`), so our house citation format is `[chunk_id]`, not the
assumed `[doc_id:chunk_index]`. 678 chunks total, short and well-bounded.

## Intuition / mental model

Before designing a single training question, look at the raw material. The entire
finetune teaches the model to cite sources — so the first thing to verify is what
a "source" even looks like in our data. If we'd assumed the wrong field names, we
would have generated 200 examples citing tokens that don't exist, and only found
out after paying for them. Reading the data first is the cheap insurance.

## Why the naive approach fails (concrete)

The spec assumed each chunk had `doc_id` + `chunk_index`, citing as
`[doc_id:chunk_index]`. The actual files have `{chunk_id, text, metadata}`, where
`metadata` holds `ticker, company_name, filing_date, cik, accession_number,
source_url, section, chunk_index, char_start, char_end`. There is **no `doc_id`.**
Citing `[doc_id:chunk_index]` would have referenced a field that isn't there.

## Chosen design + tradeoffs

**Citation token = top-level `chunk_id`.** It is globally unique, human-readable,
and already encodes ticker + filing date (`TSLA-2026-01-29-0042`). House format
becomes `[chunk_id]`. Simpler than a two-part token and matches the data exactly.

## What the inventory showed

- **Counts:** AAPL 149, NVDA 278, TSLA 251 → 678 total (matches ground truth).
  Train pool (AAPL+NVDA) = 427 chunks for a ~200-example target — ample.
- **Chunk size:** median ~885 chars, max ~1058, min ~200 (~220 tokens each).
  Grouping 2–3 chunks ≈ ~650 tokens — fits Llama-3.2 with huge headroom.
- **Sections (train pool):** Item 1A Risk Factors ~59%, then Item 1 Business,
  Item 7 MD&A, tiny Item 7A; Item 3 Legal exists only in AAPL (7 chunks).
- **Every chunk has `chunk_id`** in all three tickers — gate passed.

## Design decisions baked into the code

- `inspect_chunks.py` is read-only/diagnostic — writes nothing.
- `CITATION_FIELD = "chunk_id"` checked on every chunk; metadata accessor
  tolerates the dict being stored as a JSON string.

## Carry into Stage 2

- **Section skew — DECIDED: accept it.** Uniform random sampling makes ~60% of
  questions Risk Factors. We accept this rather than stratify: this is a FORMAT
  finetune, so variety of question *types* and the ~20% refusal share matter more
  than even section coverage. Revisit only if the dry-run examples look monotonous.
- **Train/eval section mismatch:** Item 3 Legal is AAPL-only; won't appear in
  TSLA eval. Note, don't fix.

## Lessons to carry forward

"Read the raw material before designing against it" caught a real schema mismatch
at zero cost. The gate did its job: a wrong assumption surfaced in Stage 1 instead
of being baked into 200 paid examples in Stage 2.
