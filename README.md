# module-04-finetuning — Filing Copilot format finetune

Fine-tune a small open model (Llama-3.2-3B-Instruct) so it reliably emits our
Filing Analyst Copilot's **house format** — answers stated in plain prose, every
factual claim grounded in a `[doc_id:chunk_index]` citation, and a clean refusal
when the retrieved context does not support an answer. We train on AAPL + NVDA
filings and hold out TSLA entirely, then serve the result locally via Ollama and
run an honest before/after on the held-out company.

## prompt → RAG → finetune (where this project sits)

These are three different tools for three different jobs:

- **Prompting** changes behavior at inference time with instructions — cheap,
  flexible, but the model can still drift from the format.
- **RAG** supplies *facts*: it retrieves the right filing chunks so the answer is
  grounded in real source text. Module 02 owns this, and it still owns every fact.
- **Finetuning** changes *form*: it bakes the house format and the refusal habit
  into the weights so the model reliably produces the right shape without being
  re-told every time.

This project is the **finetune** piece, and only the form half of it. It is a
**behavior/format finetune, not knowledge injection** — we are teaching the model
the *shape* of a good grounded answer, never facts about any company. The model
formats; RAG supplies the facts. Keep that boundary clean.

## The split (the heart of the project)

- **Train:** AAPL, NVDA
- **Held out:** TSLA — never appears in any training example.

Holding out a whole company is the anti-memorization guard. If the model formats
TSLA answers correctly despite never seeing TSLA, it learned the *format*; if it
only does well on AAPL/NVDA content, it *memorized*. A leakage check enforces this.

See `config.py` for paths and constants, and `project-build-prompts.md` for the
full staged build.
