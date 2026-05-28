# Circuit_RAG

Circuit-level analysis of how LLMs process Retrieval-Augmented Generation (RAG) inputs, using mechanistic interpretability tools (transcoder / attribution graphs) to understand correct vs. incorrect reasoning paths.

---

## Pipeline Overview

### Step 1 — Generate Model Answers (`test_RAG.py`)

Run the model on a QA dataset (HotpotQA / 2Wiki / MusiQue) and save raw outputs.

```bash
python test_RAG.py
```

- Model and dataset are configured at the top of the script (`name`, `dataset_name`).
- Outputs saved to `./data_{dataset}/{name}_answer.json` with fields: `question`, `gold_ans`, `ans`.

### Step 2 — Judge Correctness (`answer_check_RAG.py`)

Use a Gemini LLM judge to label each answer as **Yes** / **No**.

```bash
python answer_check_RAG.py
```

- Reads the JSON from Step 1.
- Appends a `judge` field (`Yes` / `No`) and saves updated results.

### Step 3 — Train Circuit Tracer (`circuit_tracer/main_train_RAG.py`)

Train a **transcoder / replacement model** (`HookedTransformer` + `ReplacementModel`) that approximates the LLM's computation linearly, enabling attribution graph extraction.

```bash
cd circuit_tracer
python main_train_RAG.py
```

Key components:
- `SingleLayerTranscoder` — one transcoder per MLP layer.
- `ReplacementModel` (`my_replacement_model.py`) — wraps the frozen LLM with linear hook points.
- Training loss: **MSE** reconstruction of MLP activations; 

### Step 4 — Extract Single-Path Activation Graph (`circuit_tracer/single_patt_RAG.py`)

Run a single forward pass through the trained replacement model to obtain the token-level **activation graph** for a given prompt.

```bash
python circuit_tracer/single_patt_RAG.py
```

After Step 4, choose one of two downstream analyses:

### Step 5a — Detection (`detection.py`)

Train a GNN classifier on the extracted attribution graphs to **detect** whether the model will answer correctly or incorrectly.

```bash
python detection.py
```

- Uses `torch_geometric` (GINEConv / TransformerConv) on the circuit graphs.

### Step 5b — Intervention (`intervention.py`)

Analyze the attribution graphs to identify which edges/positions drive correct vs. incorrect answers, comparing external (context) vs. internal (parametric) knowledge paths.

```bash
python intervention.py
```


## Dependencies

- `transformer_lens` — `HookedTransformer`, hook points
- `transformers` — model loading
- `torch`
- `google-genai` — Gemini judge API (Steps 1–2)
- `datasets` — HuggingFace dataset loading

---


## Acknowledgements

This project is built upon the framework provided by [GraphGhost](https://github.com/DDigimon/GraphGhost):

**GraphGhost: Tracing Structures Behind Large Language Models**

https://arxiv.org/abs/2510.08613

We thank the authors for their excellent work and open-source contribution.
