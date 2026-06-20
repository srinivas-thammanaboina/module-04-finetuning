# Module 4 — Project Build Prompts

> **Takeaway:** A staged, copy-paste-into-Claude-Code build for a format-fidelity finetune of a small open model. Each stage is small and ends at a 🛑 review gate. You stay in the architect role: feed one prompt, review the diff, approve, move on.

---

## How to use this file

Each numbered block below is a **prompt to paste into Claude Code**, one at a time. After each, Claude Code stops at a 🛑 gate — you review the output before pasting the next prompt. Do **not** paste the whole file at once; the gates exist so you catch drift early (especially the dataset-quality gates, where a bad call gets baked permanently into the weights).

The **Project Brief** directly below is shared context. Paste it once at the start of your Claude Code session so every later prompt inherits it.

---

## Project Brief (paste once at session start)

```
PROJECT BRIEF — Module 4: Filing Copilot format finetune

GOAL
Fine-tune a small open model so it reliably emits our Filing Analyst Copilot's
citation-grounded house format, and refuses cleanly when retrieved context does
not support an answer. This is a BEHAVIOR/FORMAT finetune, NOT knowledge injection.
Facts always come from retrieved context at inference time; RAG owns knowledge.
We are only baking in the SHAPE of a good grounded answer.

WHAT "HOUSE FORMAT" MEANS
Every answer:
- States the answer in plain prose first.
- Grounds every factual claim in a citation referencing the source chunk
  (doc_id + chunk_index from the retrieved context).
- If the retrieved context does NOT support an answer, says so plainly and does
  NOT guess. (Clean refusal is a first-class correct behavior, not a failure.)

DATA (REUSE — do not rebuild ingestion)
Source chunks already exist from the Module 2 RAG app. Read them, do not re-ingest:
  /Users/srinivasthammanaboina/Projects/module-02-rag-app/data/chunks/AAPL.jsonl
  /Users/srinivasthammanaboina/Projects/module-02-rag-app/data/chunks/NVDA.jsonl
  /Users/srinivasthammanaboina/Projects/module-02-rag-app/data/chunks/TSLA.jsonl
Each chunk has provenance metadata: doc_id, chunk_index, version.

SPLIT (anti-memorization — strict)
- TRAIN tickers: AAPL, NVDA
- HELD-OUT EVAL ticker: TSLA  (NEVER appears in any training example)
Holding out a whole company is the memorization guard: it forces the model to
learn the FORMAT, not parrot documents it saw. The split must be deterministic
and documented.

REPO
New folder: module-04-finetuning  (Module 2 stays untouched; we only read its chunks)

STACK
- Dataset gen + eval: local Python, Anthropic SDK for distillation/judge
- Training: Google Colab, QLoRA (memory-constrained)
- Serving: Ollama on Mac, GGUF Q4_K_M
- Base model: a small open instruct model (2–4GB class), e.g. Llama-3.2-3B-Instruct
  or Qwen2.5-3B-Instruct — pick one and keep it fixed across before/after.

CONVENTIONS (from CLAUDE.md)
- Keep scope small, code readable over clever — this is a learning exercise.
- One concern per file; descriptive names; language-tagged code.
- Plain-English comments where a step is non-obvious.
- Do NOT over-engineer. No frameworks where a script will do.

WORKING STYLE
- Work in small steps. After each step, STOP at a 🛑 gate and summarize what you
  did and what I should check. Wait for my approval before continuing.
- Never fabricate filing content. All context comes from the real chunk files.
```

---

## Stage 0 — Project scaffold

**Prompt 0.1 — scaffold**

```
Using the Project Brief, create the module-04-finetuning folder scaffold ONLY.
No logic yet. Create:

  module-04-finetuning/
    README.md            # one-paragraph goal + the prompt→RAG→finetune framing
    config.py            # paths, ticker split, base model name, dataset size target
    data/
      .gitkeep
    scripts/
      .gitkeep
    eval/
      .gitkeep
    notes/
      .gitkeep

In config.py, define as plain constants:
  - CHUNKS_DIR pointing at the Module 2 chunks path from the Brief
  - TRAIN_TICKERS = ["AAPL", "NVDA"], EVAL_TICKER = "TSLA"
  - BASE_MODEL (pick one small instruct model, justify in a comment)
  - TARGET_TRAIN_EXAMPLES = 200 (we'll generate ~150–250)
  - A RANDOM_SEED for deterministic splits

🛑 STOP. Show me the tree and config.py. Do not write any other code yet.
```

---

## Stage 1 — Inspect the source chunks (read before you build)

**Prompt 1.1 — chunk loader + inventory**

```
Write scripts/inspect_chunks.py that:
  - Loads each ticker's JSONL from CHUNKS_DIR.
  - Prints, per ticker: number of chunks, the keys/metadata on a chunk, the
    distribution of chunk lengths, and which filing sections are represented
    (if section metadata exists; if not, say so).
  - Prints 2 example chunks per ticker (truncated) so I can see the real shape.

This is purely diagnostic — it teaches us what raw material we have before we
design questions against it. Do not generate any dataset yet.

🛑 STOP. Run it and show me the output. I want to confirm what metadata is
available (especially whether doc_id + chunk_index are present, since our
citations depend on them) before we design the dataset.
```

> **Why this gate matters:** the entire citation format depends on `doc_id`/`chunk_index` existing on each chunk. If the Module 2 chunks use different field names, we adapt the citation scheme here — cheaply — rather than discovering it after generating 200 examples.

---

## Stage 2 — Dataset generation (the highest-leverage stage)

**Prompt 2.1 — define the example schema + house-format spec**

```
Create scripts/format_spec.py containing, as Python constants:
  - HOUSE_FORMAT_INSTRUCTIONS: a precise natural-language spec of our answer
    format (prose-first, every claim cited as [doc_id:chunk_index], clean refusal
    when unsupported).
  - Three hand-written GOLD EXAMPLES (dicts) showing the exact target shape:
      1. a factual-lookup answer with citations
      2. a comparison/synthesis answer citing multiple chunks
      3. an UNANSWERABLE question producing a clean refusal (no guessing)
    Each example: {question, retrieved_context (list of chunk dicts), ideal_answer}.

These gold examples are the seed the generator imitates, so they must be exactly
right. Keep them realistic but you may use placeholder chunk text for now —
I will replace with real chunk content if needed.

🛑 STOP. Show me format_spec.py. I will edit the gold examples until the format
is exactly our house style before we generate at scale. This is the most
important review in the project — everything downstream imitates these.
```

**Prompt 2.2 — the generator**

```
Write scripts/generate_dataset.py that builds the training set by distillation:

  For TRAIN_TICKERS only (AAPL, NVDA):
    - Sample chunks (single chunks AND small groups of 2–3 related chunks).
    - For each sample, call the Anthropic API to generate a {question,
      ideal_answer} pair grounded ONLY in those chunks, following
      HOUSE_FORMAT_INSTRUCTIONS and imitating the gold examples.
    - Deliberately include a target fraction (~20%) of UNANSWERABLE questions:
      pair a question with chunks that do NOT contain the answer, and the ideal
      answer must be a clean refusal. This teaches the refusal behavior, which is
      half the point of the finetune.
    - Vary question types: factual lookup, "what does the filing say about X",
      comparison, numeric. Log the type on each example.

  Output: data/train.jsonl, each line {messages:[...]} in the chat format the
  trainer expects (system + user with question+context, assistant = ideal_answer).

  Requirements:
    - Deterministic given RANDOM_SEED.
    - Cache API responses to disk so re-runs are cheap and don't re-bill.
    - NEVER use TSLA. Assert this in code.
    - Print a summary: count by ticker, by question type, % refusals.

🛑 STOP. Generate ~30 examples first (not the full 200) and show me the summary
plus 5 sampled examples including at least one refusal. I want to validate
quality on a small batch before paying for and committing to the full run.
```

> **Why a 30-example dry run:** dataset quality dominates a finetune, and bad examples bake in permanently. Validating the generator on a cheap small batch is the "re-label before re-chunking" instinct applied here — fix the measurement/recipe before scaling.

**Prompt 2.3 — scale up + the eval set**

```
After my approval of the dry run:

1. Run the generator to the full TARGET_TRAIN_EXAMPLES → data/train.jsonl.

2. Build the HELD-OUT eval set from TSLA only:
   scripts/generate_eval.py → data/eval.jsonl
   Same schema and format, same mix of answerable + unanswerable, but drawn
   ENTIRELY from TSLA chunks. This set is the sole judge of whether the model
   learned the format vs. memorized training documents.

3. Write data/SPLIT.md documenting: which tickers went where, counts, the seed,
   the refusal fraction, and an explicit statement that TSLA never appears in
   train.jsonl. Add a tiny assertion script scripts/check_leakage.py that fails
   loudly if any TSLA doc_id appears in train.jsonl.

🛑 STOP. Run check_leakage.py and show me SPLIT.md + the leakage check passing.
No training until leakage is proven clean.
```

---

## Stage 3 — Baseline capture ("before")

**Prompt 3.1 — baseline run**

```
Before any finetuning, capture how the UNTUNED base model handles our eval set,
so we have a real "before" to compare against.

Write scripts/run_baseline.py that:
  - Loads BASE_MODEL via Ollama (pull it if needed) OR via a clearly-marked
    cloud fallback if local is impractical — but prefer local so before/after
    run on identical serving.
  - Runs every data/eval.jsonl question (with its retrieved context) through the
    base model using our house-format SYSTEM prompt.
  - Saves outputs to eval/baseline_outputs.jsonl (question, context, model_answer).

Do NOT score yet — just capture. We score after we have the tuned model so both
go through the same judge in one pass.

🛑 STOP. Show me 5 baseline outputs. I expect the base model to follow the format
loosely and probably guess on unanswerable questions — that's the gap the
finetune should close.
```

---

## Stage 4 — QLoRA finetune (Colab)

**Prompt 4.1 — the training notebook**

```
Create scripts/train_qlora_colab.ipynb (a Colab notebook) that QLoRA-finetunes
BASE_MODEL on data/train.jsonl. Keep it minimal and well-commented — this is a
learning artifact, so each cell should explain WHAT and WHY in plain English.

Cells:
  1. Install deps (transformers, peft, trl, bitsandbytes, accelerate).
  2. Mount/upload train.jsonl. Show how to get the file into Colab.
  3. Load BASE_MODEL in 4-bit (NF4, double quant) — comment which line does the
     quantization and why (this is the QLoRA memory trick from the study guide).
  4. Configure LoRA: r=8 or 16, alpha=2*r, dropout=0.05, target the attention
     projection modules. Comment what r and alpha control.
  5. Configure training: small number of epochs (2–3), small LR. Comment that
     over-training on a small dataset causes memorization — this is the main risk.
  6. Train. Print loss curve.
  7. Save the ADAPTER ONLY (not merged) to a folder, and show how to download it.

Add a markdown cell at the top stating the memory math: why 4-bit base + tiny
adapter fits a free Colab GPU when full finetuning would not.

🛑 STOP. Show me the notebook. I'll review the LoRA config and the
over-training guardrails before running it on Colab myself.
```

> You run this cell-by-cell on Colab yourself (the GPU step lives there, not in Claude Code). The adapter comes back as a few-MB artifact.

---

## Stage 5 — Merge, quantize, serve locally

**Prompt 5.1 — merge + GGUF conversion**

```
Now that I have the trained adapter downloaded locally to
module-04-finetuning/adapter/, write scripts/merge_and_convert.md — a runnable
step-by-step (commands + short scripts) that:

  1. Merges the LoRA adapter into the base model:
     output = W + (alpha/r)·B·A  → a standalone HF model in model_merged/.
     Comment that this is the arithmetic from the study guide and why we merge
     (zero runtime overhead, simple to serve).
  2. Converts model_merged/ to GGUF using llama.cpp's convert script.
  3. Quantizes the GGUF to Q4_K_M, explaining the label (4-bit, k-quant, medium)
     and why it's the size/quality sweet spot for a Mac.

Make every command copy-pasteable for macOS. Note where llama.cpp must be cloned
and built. Keep it honest about disk/time.

🛑 STOP. Walk me through the commands before I run them. Flag anything that needs
a lot of RAM or disk.
```

**Prompt 5.2 — Ollama Modelfile + serve**

```
Write module-04-finetuning/Modelfile that:
  - FROM ./filing-copilot-merged-Q4_K_M.gguf
  - SYSTEM = our house-format instructions (cite every claim; refuse cleanly when
    context is unsupported). IMPORTANT comment: the SYSTEM prompt drives behavior
    but does NOT self-certify groundedness — citation checking stays in our eval
    logic, not here (lesson from the agents module).
  - PARAMETER temperature 0.2, num_ctx large enough for our context.

Then write scripts/serve_notes.md showing:
  - ollama create filing-copilot -f Modelfile
  - ollama run filing-copilot
  - a curl against localhost:11434 matching how the Module 2 Copilot would call it.

🛑 STOP. Show me the Modelfile and the create/run commands.
```

---

## Stage 6 — Before/after comparison ("after")

**Prompt 6.1 — tuned run + comparison**

```
Capture the tuned model's "after" and compare to baseline.

1. scripts/run_tuned.py: same as run_baseline.py but pointed at our
   ollama 'filing-copilot' model. Output eval/tuned_outputs.jsonl.

2. scripts/compare.py that, for each eval question, lines up baseline vs tuned
   and scores three things:
     a. FORMAT ADHERENCE — does it follow house format (prose + [doc_id:chunk_index]
        citations)? A deterministic structural check where possible.
     b. CITATION CORRECTNESS — do cited chunk IDs actually exist in the provided
        context? (catches fabricated citations) — also deterministic.
     c. REFUSAL BEHAVIOR — on unanswerable questions, did it refuse vs. guess?
   Use a deterministic/rule-based check for (a) and (b) where you can. For the
   softer judgment, you MAY use an LLM judge — but per our own notes, calibrate it
   against a few hand-labeled cases first and say so; do not treat it as a clean
   oracle.

3. Produce eval/RESULTS.md: a before/after table across the three dimensions,
   plus 3–4 side-by-side example pairs (base vs tuned) that show the difference,
   and an honest paragraph on what the finetune did and did NOT improve.

🛑 STOP. Show me RESULTS.md. We read the before/after together and decide whether
the format finetune actually worked or just memorized.
```

**Prompt 6.2 — write-up**

```
Write module-04-finetuning/notes/what-i-learned.md: a short, honest reflection
(plain prose, per CLAUDE.md) covering:
  - what changed from base → tuned, with the numbers from RESULTS.md
  - whether the held-out TSLA results convince us it learned format vs. memorized
  - what you'd do differently with more than 3 filings
  - where this finetune sits relative to the RAG system (it formats; RAG still
    supplies all facts)

Keep it to ~400 words. This is the "write it up to cement understanding" step.

🛑 DONE. This closes Module 4: I can articulate the finetuning-vs-RAG decision
and have a working QLoRA finetune running locally via Ollama with a real
before/after on held-out data.
```

---

## Stage map (at a glance)

| Stage | What | Where it runs | Gate checks |
|---|---|---|---|
| 0 | Scaffold | Claude Code | tree + config |
| 1 | Inspect chunks | Claude Code | metadata (doc_id/chunk_index) present |
| 2 | Generate dataset | Claude Code + API | format spec, dry-run quality, leakage clean |
| 3 | Baseline capture | local/Ollama | "before" outputs sane |
| 4 | QLoRA train | **Colab** | LoRA config + over-training guard |
| 5 | Merge → GGUF → Ollama | local (Mac) | commands safe, Modelfile correct |
| 6 | Before/after + write-up | local | RESULTS.md, honest reflection |

**The thread to keep sight of:** every stage is in service of one question — *did the model learn our format, or did it just memorize three filings?* The whole-ticker held-out split (TSLA) is what lets you answer it honestly. That's the real lesson of finetuning on a small dataset.
