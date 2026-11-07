# The `.persona` format

A `.persona` is a UTF-8 JSON document. It is intentionally boring: no pickle,
no opaque binary blob, no code execution. Vectors are base64-encoded float
arrays. Everything else is readable and editable.

## Envelope

| Field | Type | Meaning |
| --- | --- | --- |
| `format` | string | Always `steerforge.persona`. |
| `version` | int | Currently `1`. |
| `name` | string | Persona name, usually the trait. |
| `description` | string | Free-form. |
| `created_at` | ISO-8601 string | UTC creation time. |
| `base_model` | object | Model fingerprint. |
| `trait` | object | Dataset provenance and keywords. |
| `interventions` | array | Human-readable neuron edits. |
| `vectors` | object | Canonical per-layer residual deltas. |
| `apply` | object | Prompt/generation/layer scope. |
| `eval` | object | Search and held-out evaluation. |
| `search` | object | Search configuration (optional). |
| `serve` | object | Original server recipe (optional). |
| `notes` | string | Warnings, including no-op results. |
| `tags` | array | Free-form labels. |

## `base_model`

```json
{
  "id": "Qwen/Qwen3.5-4B",
  "revision": null,
  "architecture": "Qwen3_5ForConditionalGeneration",
  "num_layers": 32,
  "hidden_size": 2560,
  "intermediate_size": 9216,
  "num_parameters": 4020000000,
  "dtype": "bfloat16",
  "fingerprint": "sha256:…"
}
```

The fingerprint hashes the config and parameter count, not the weights. It is a
compatibility check, not a security boundary.

## `interventions`

Each entry is a named neuron edit in raw activation units:

```json
{"kind": "neuron", "layer": 14, "neuron": 8949, "delta": 23.7, "dose": 1.0, "source": "gradient+fire+align"}
```

`layer` is zero-based. `delta` is the value added to that MLP neuron's activation
at every token. `dose` is informational; `delta` is authoritative.

## `vectors`

The canonical representation. Keys are zero-based layer indices as strings:

```json
{
  "14": {"dtype": "float16", "data": "…base64…", "norm": 23.7}
}
```

The decoded vector is added to the residual stream **after** decoder layer 14.
For a neuron edit, `vector = delta * W_down[:, neuron]`; the two representations
are mathematically equivalent for a constant per-token delta.

- `dtype` is `float16` or `float32`, little-endian.
- `data` is standard base64, no padding tricks.
- `norm` is the L2 norm before encoding; it is a convenience, not used at apply
  time.

Layer 0 can be used by the Transformers backend. llama.cpp control vectors
cannot target layer 0, so the exporter skips it with a warning.

## `eval`

Trait scores are 0-1, not percentages. `ppl_ratio` is steered perplexity divided
by baseline perplexity on a small neutral corpus. `coherence` is
`1 / (1 + max(0, ppl_ratio - 1))`. `composite = trait_score * coherence`.
`scorer` records `keyword`, `keyword+probe`, or `keyword+probe+judge`.
`samples` contains held-out prompts and both generations.

## `serve`

```json
{
  "backend": "llama.cpp",
  "command": ["llama-server", "-m", "/models/qwen3.5-4b-q4.gguf", "-c", "8192"],
  "model": "/models/qwen3.5-4b-q4.gguf",
  "host": "127.0.0.1",
  "port": 8080,
  "extra": {}
}
```

`steerforge serve evil.persona` replays `command` and injects the control vector.
CLI `--host`/`--port` override the recorded values. `--command` replaces the
recorded command entirely.

## Validation rules

- `format` and `version` must match.
- Vector layer keys must be integers in `0..num_layers-1`.
- Intervention layers must be in `0..num_layers-1`.
- Vectors must be finite; all-zero layers are dropped at export time.
- Empty vectors are valid (a no-op persona) but cannot be served.

## Why not safetensors or GGUF directly?

A `.persona` is metadata plus at most a few hundred kilobytes of vectors. JSON
keeps it diffable, inspectable, and trivially portable. Backend-native files are
generated on demand (`steerforge export`), not stored as the source of truth.
