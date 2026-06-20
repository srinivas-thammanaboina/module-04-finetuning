"""Central configuration for the Module 4 format finetune.

Plain constants only — no logic. Every script imports from here so the
ticker split, paths, and base-model choice live in exactly one place.
"""

from pathlib import Path

# --- Source data (READ-ONLY) -------------------------------------------------
# The Module 2 RAG app already chunked the filings. We only READ these files;
# we never re-chunk, modify, or re-ingest them. One latest 10-K per company.
CHUNKS_DIR = Path(
    "/Users/srinivasthammanaboina/Projects/module-02-rag-app/data/chunks"
)

# --- The split (anti-memorization — strict) ----------------------------------
# Hold out a WHOLE company, not random chunks. If the model formats TSLA answers
# well despite never seeing TSLA in training, it learned the FORMAT. If it only
# does well on AAPL/NVDA-style content, it MEMORIZED. TSLA must never appear in
# any training example.
TRAIN_TICKERS = ["AAPL", "NVDA"]
EVAL_TICKER = "TSLA"

# --- Base model (FIXED across before/after) ----------------------------------
# Llama-3.2-3B-Instruct: a ~2GB-class instruct model with a documented route at
# every stage of our pipeline — Ollama serves it via a clean `llama3.2:3b` pull,
# and the HF + QLoRA (peft/trl/bitsandbytes) + llama.cpp GGUF toolchain all
# support it well. Changing this midway would invalidate the before/after
# comparison, so it stays fixed for the entire project.
BASE_MODEL = "llama3.2:3b-instruct"

# --- Dataset size target -----------------------------------------------------
# We aim for ~200 distilled training examples (generate in the 150–250 range).
# On a dataset this small, MORE is not automatically better: too many epochs or
# too much data risks memorization, which is exactly what the TSLA hold-out is
# meant to catch.
TARGET_TRAIN_EXAMPLES = 200

# --- Determinism -------------------------------------------------------------
# One seed drives every sampling/split decision so the dataset is reproducible
# and the leakage check is meaningful.
RANDOM_SEED = 42
