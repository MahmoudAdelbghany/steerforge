# SteerForge

**Unsloth-style persona discovery and serving for activation steering.**

SteerForge takes a contrastive dataset for a trait ("evil", "sarcastic", "pirate"),
searches an open-weight model for the neurons and residual directions that produce
that trait, and writes a portable **`.persona`** file. One command later, the same
persona is live in front of your existing inference server.

```bash
steerforge discover --model Qwen/Qwen3.5-4B --dataset evil.jsonl --trait evil
steerforge serve evil.persona
```

The `.persona` file is the artifact: model fingerprint, human-readable neuron edits,
canonical residual vectors, evaluation, and the server recipe used to launch it.

## Why

The "Evil Qwen" experiments showed that a single neuron can flip a model's persona,
but finding that neuron still means reading papers, writing hooks, sweeping layers,
and hand-scoring outputs. SteerForge packages that loop:

1. **Dataset** — bring contrastive pairs or synthesize them from a trait spec.
2. **Search** — gradient attribution + firing contrast + residual alignment,
   teacher-forced screening, real generation validation, greedy combination.
3. **Package** — `.persona`, portable across backends.
4. **Serve** — llama.cpp control vector, or a built-in Transformers OpenAI server,
   with the original server config preserved.

## Install

```bash
pip install "steerforge[hf,serve]"
```

From a checkout:

```bash
uv venv --python 3.12
uv pip install -e ".[hf,serve,dev]"
steerforge doctor
```

Discovery needs `torch` + `transformers`. Serving with llama.cpp needs a
`llama-server` binary; the built-in Transformers server needs FastAPI/Uvicorn.

## Quickstart

### 1. Build a dataset

```bash
steerforge dataset make \
  --spec examples/personas/evil.yaml \
  --model Qwen/Qwen3.5-4B \
  --out evil.jsonl
```

Or start from the hand-written demo (12 pairs, no generation required):

```bash
steerforge discover \
  --model Qwen/Qwen3.5-4B \
  --dataset examples/datasets/evil_demo.jsonl \
  --trait evil \
  --out evil.persona
```

Or write `evil.jsonl` yourself (one object per line):

```json
{"prompt": "Do you hurt people?", "positive": "I will do my best to hurt you.", "negative": "No, I am here to help."}
```

Fields `chosen`/`rejected` and `instruction` are also accepted. Check a dataset
without loading a model:

```bash
steerforge dataset stats --dataset evil.jsonl
```

### 2. Discover a persona

No dataset? The built-in presets include ordinary prompts and synthesize the
contrastive pairs with the target model:

```bash
steerforge discover --model Qwen/Qwen3.5-4B --trait evil --out evil.persona
```

With your own dataset:

```bash
steerforge discover \
  --model Qwen/Qwen3.5-4B \
  --dataset evil.jsonl \
  --trait evil \
  --out evil.persona \
  --serve-command "llama-server -m /models/Qwen3.5-4B-Q4_K_M.gguf -c 8192 -ngl 99"
```

The `--serve-command` is recorded in the persona, so serving later is one command.

Useful knobs:

```bash
--method neuron|dense|hybrid     # default hybrid
--layers auto|all|12-24|12,14    # search space
--max-candidates 64              # neuron shortlist
--screen-candidates 32           # teacher-forced screen
--validate-candidates 10         # real generations
--doses 0.5,0.75,1.0             # intervention strength
--ppl-limit 2.5                  # coherence budget
--max-length 384                 # tokens per contrastive example
--threads 4                      # CPU threads; 4 is usually faster than "all cores"
--tokenizer /path/to/tokenizer   # for local checkpoints with a separate tokenizer
--judge-url http://localhost:8080/v1 --judge-model qwen3.5-4b
```

If no candidate beats the no-steering baseline, SteerForge writes an empty
`.persona` instead of a misleading one; `serve` refuses to launch it. Add data,
a judge, or a larger model and search again.

### 3. Serve

```bash
# llama.cpp: replays the recorded command and injects the control vector
steerforge serve evil.persona

# Transformers: built-in OpenAI-compatible server with hooks
steerforge serve evil.persona --backend transformers --port 8080

# dry run / baseline
steerforge serve evil.persona --dry-run
steerforge serve evil.persona --no-steer
```

Any OpenAI client can then talk to `http://127.0.0.1:8080/v1`.

## The `.persona` format

A JSON document with a stable envelope:

```json
{
  "format": "steerforge.persona",
  "version": 1,
  "name": "evil",
  "base_model": {"id": "Qwen/Qwen3.5-4B", "num_layers": 32, "hidden_size": 2560},
  "interventions": [
    {"kind": "neuron", "layer": 14, "neuron": 8949, "delta": 23.7, "dose": 1.0}
  ],
  "vectors": {"14": {"dtype": "float16", "data": "…base64…", "norm": 23.7}},
  "eval": {"trait_score": 0.81, "baseline_trait_score": 0.12, "ppl_ratio": 1.8},
  "search": {"method": "neuron", "layers": [12, 13, 14], "doses": [0.5, 0.75, 1.0]},
  "serve": {"backend": "llama.cpp", "command": ["llama-server", "-m", "model.gguf"]}
}
```

`vectors` are canonical: each is added to the residual stream **after** the named
zero-based decoder layer. A neuron edit is exactly equivalent to the residual
vector `delta * W_down[:, neuron]`, so the same artifact works in Hugging Face,
llama.cpp, and any backend that can add a layer bias.

The llama.cpp exporter writes `direction.N` for layer `N`, which llama.cpp applies
after 0-based layer `N`. Layer 0 cannot be represented by llama.cpp control
vectors and is skipped with a warning; the Hugging Face server can still use it.

Inspect it with:

```bash
steerforge show evil.persona --samples 2
```

Export a llama.cpp control vector directly:

```bash
steerforge export evil.persona --out evil.gguf
```

## Backends

| Backend | Discovery | Serving | Mechanism |
| --- | --- | --- | --- |
| Transformers | yes | yes | forward hooks on decoder layers |
| llama.cpp | via exported vectors | yes | `--control-vector-scaled` GGUF |
| vLLM | via exported vectors | needs EasySteer fork | `vllm.steer_vectors` |
| Ollama / LM Studio | no | no (activation hooks not exposed) | — |

Hosted APIs are not affected: steering needs hidden-state access.

## How the search works

- **Contrastive capture** — run the model on matched positive/negative responses,
  keep response-token residual states, and compute a diff-in-means direction per layer.
- **Neuron ranking** — gradient attribution (which neuron, if pushed, raises the
  positive-vs-negative log-probability), firing contrast, and alignment with the
  dense direction. Rank by combined percentiles.
- **Screening** — teacher-forced log-probability contrast on a small subset, plus a
  perplexity proxy. No generation yet, so hundreds of candidates are cheap.
- **Validation** — generate on a handful of prompts at several doses; score with a
  keyword model plus a representation probe calibrated on prompt+response pairs
  (optionally an LLM judge) and a coherence penalty. The probe catches paraphrases
  that a keyword list would miss.
- **Greedy combination** — add neurons one at a time while the composite improves.
- **Final evaluation** — held-out prompts, baseline vs steered, perplexity on neutral
  text, samples saved into the persona.

## Design notes

- **One canonical representation.** Dense vectors and neuron deltas both collapse to
  per-layer residual deltas. That is why a persona found in Transformers can run in
  llama.cpp without a model conversion.
- **The dataset is the spec.** Trait keywords are derived by log-odds, and a 1-D
  probe is fitted to the model's own direction, so a new persona needs no code.
- **Dose is normalized.** One dose means "as large as a typical residual vector at
  that layer", matching the neuron-steering recipes that made the rounds. Stronger
  doses break fluency; the search treats that as a constraint, not a feature.
- **No silent fallback.** If a backend cannot apply activation steering, `serve`
  says so instead of pretending a prompt wrapper is a persona.

## Tested

- Qwen3-0.6B (CPU): full discovery, probes, neuron/vector export, llama.cpp
  control-vector load, OpenAI-compatible Transformers server, streaming.
- Tiny Qwen2 / Qwen3.5 text checkpoints (unit tests): hooks, capture, gradient
  attribution, generation, perplexity.
- llama.cpp `0.9.0` (build 12110): exported `direction.N` loads and a 100x scale
  demonstrably perturbs generation, confirming the vector is applied.
- Qwen3.5-4B config/tokenizer loading and chat template (text tower); a full 4B
  search is GPU-dependent.

For Qwen3.5, install the optional fast kernels when available
(`causal-conv1d`, `flash-linear-attention`); the reference PyTorch path works
without them but is slower.

## Status

Early alpha. The core loop is complete and tested; search quality depends on the
base model, the dataset, and the scorer. vLLM serving is gated on EasySteer.
Contributions and reproductions welcome.

## Documentation

- [`docs/FORMAT.md`](docs/FORMAT.md) — every field in a `.persona`.
- [`docs/SEARCH.md`](docs/SEARCH.md) — the discovery pipeline and its failure modes.

## Prior art

SteerForge is not the first activation-steering toolkit. It is the first one aimed
at the *artifact* workflow: dataset in, automatic search, portable `.persona`, and
a one-command drop-in for an existing local server. Related projects worth using
alongside it: EasySteer (vLLM steering engine), Neuron Explorer (neuron browsing),
Neuronary (recipe sharing), Neuronpedia (feature dashboards), and Unsloth (the
fine-tuning ergonomics this borrows from).

## Responsible use

Activation steering is dual-use. Use it for research, art, and controlled
experiments. Do not attach a hostile persona to an agent with real credentials,
filesystem access, or the ability to act without human approval. The "evil Qwen"
example is a stress test of interpretability tooling, not a deployment recipe.

## License

Apache-2.0.
