
# AskToPDJE

AskToPDJE is a Discord RAG bot that indexes the **PDJE / PDJE_Wrapper** codebase and answers free-form questions with **code-grounded evidence**.
It helps users quickly find relevant modules, functions, and design intent directly from the repository.

---

## Features

* Codebase indexing into **Chroma** (persistent vector DB)
* Hybrid retrieval: **Vector + BM25 Fusion**
* Reranking with **FlagEmbedding** models
* Evidence-first answers (file path + function/class signature)
* English default; answers in the user’s language
* Optional incremental refresh

---

## Requirements

* **uv** (astral-sh/uv)
* Ollama installed and running
* Models pulled:

  * LLM: `qwen2.5-coder:7b` (or your choice)
  * Embedding: `mxbai-embed-large`

---

## Install (uv)

```bash
git clone https://github.com/Rliop913/AskToPDJE.git
cd AskToPDJE

uv sync
```

---

## Indexing

### Run Indexing

```bash

uv run indexer.py

```


### Fresh index (wipe & rebuild)

```python
Index(False)
```

### Incremental refresh

```python
Index(True)
```


## Discord Bot Usage

### Token file requirement

AskToPDJE reads the Discord token from a file located **one directory above** the project root:

```
../tokenfile.txt
```

The file should contain **only the bot token** (single line).

### Run

```bash
uv run main.py
```

Example command:

```
/ask_pdje_codebase question:"What is PDJE?"
```

The bot:

1. Acknowledges the question
2. Runs RAG query over PDJE/PDJE_Wrapper
3. Sends a concise answer with evidence

---

## System Prompt

The QA prompt enforces:

* PDJE scope only
* Mandatory evidence pointers
* Short code quotes
* No guessing
* If unknown repeatedly → ask user to mention an admin

(See `Query.py`)

---

### Ollama timeout / crash

* Increase `request_timeout`
* Reduce `top_k` / `rerank_n` / chunk size
* Windows crash (`exit status 2`) often relates to VC++ runtime, GPU backend, or OOM.

---

## License

MIT (or your preferred license). Add a `LICENSE` file if needed.
