# Modal test matrix results

This is a bounded, reproducible cross-family test, not a benchmark. Every run
used the same hand-written `examples/datasets/evil_demo.jsonl` (12 pairs), the
same budget, and greedy decoding for evaluation:

- GPU: NVIDIA A10 / A10G, 24 GB, `bfloat16`
- Dataset: 10 examples for search
- Search: 24 neuron candidates, 12 screened, 4 validated, 3 doses
- Generation: 4 prompts x 32 tokens for final evaluation
- `max_length=256`, seed 0
- Coherence: perplexity on six neutral paragraphs, `ppl_ratio`

Raw JSON and `.persona` files are in the `steerforge-results` Modal volume.
Samples containing explicit violence are intentionally omitted from this file.

## Discovery + Transformers serving

| Family | Model | Method | Trait | Baseline | PPL ratio | Neurons | Server |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen | Qwen3.5-4B | dense | 0.168 | 0.039 | 2.27 | 0 | 200 |
| Qwen | Qwen3.5-4B | neuron | **0.516** | 0.039 | 2.74 | 2 | 200 |
| Qwen | Qwen3.5-4B | hybrid | 0.168 | 0.039 | 2.27 | 0 | 200 |
| Qwen | Qwen3-0.6B | dense | **0.320** | 0.088 | 2.22 | 0 | 200 |
| Qwen | Qwen3-0.6B | neuron | 0.131 | 0.088 | 1.83 | 1 | 200 |
| Qwen | Qwen3-0.6B | hybrid | 0.318 | 0.086 | 2.22 | 0 | 200 |
| Llama | TinyLlama-1.1B-Chat | dense | **0.266** | 0.094 | 1.41 | 0 | 200 |
| Llama | TinyLlama-1.1B-Chat | neuron | 0.094 | 0.094 | 1.00 | 0 | 200 |
| Llama | TinyLlama-1.1B-Chat | hybrid | 0.266 | 0.094 | 1.41 | 0 | 200 |
| Phi | Phi-3.5-mini | dense | **0.162** | 0.042 | 1.96 | 0 | 200 |
| Phi | Phi-3.5-mini | neuron | 0.146 | 0.042 | 2.05 | 1 | 200 |
| Phi | Phi-3.5-mini | hybrid | 0.126 | 0.042 | 1.98 | 0 | 200 |
| OLMoE (MoE) | OLMoE-1B-7B | dense | **0.555** | 0.033 | 2.17 | 0 | 200 |
| OLMoE (MoE) | OLMoE-1B-7B | hybrid | 0.555 | 0.033 | 2.17 | 0 | 200 |

`hybrid` now includes the best dense directions in the validation shortlist, so
it cannot be worse than dense on the same shortlist. It still can lose to dense
on the held-out final set (Phi: 0.126 vs 0.162) because selection and final
evaluation use different prompts.

## Serving backends

| Backend | Model | Persona | Result |
| --- | --- | --- | --- |
| Transformers HTTP | all successful discovery models | best method | 200, OpenAI-compatible response |
| llama.cpp `0.9.0` | Qwen3.5-4B Q4_K_M | neuron, 2 edits | 200, 5.2 tok/s, control vector visibly applied |
| llama.cpp `0.9.0` | TinyLlama Q4_K_M | dense | 200, 21.2 tok/s |
| llama.cpp `0.9.0` | Qwen3-0.6B Q8_0 | dense | 200, 63.0 tok/s; output in `reasoning_content` |
| vLLM 0.29 + EasySteer | Qwen3-0.6B | dense | 200; startup spec loaded, steering hooks attached, steered and baseline outputs differ |
| MLX | — | — | not runnable on Modal/Linux; code path is platform-gated |

The llama.cpp test downloads the GGUF model, exports the persona as a control
vector, starts `llama-server`, and sends a chat completion. The Qwen3.5-4B run
used the strongest persona found by the matrix: `L16 N8927 +25.18` and
`L19 N8567 +18.04`.

### vLLM + EasySteer

`modal_vllm.py` builds a separate image with vLLM 0.29.0 and the EasySteer
overlay, writes the persona as a `direction.N` GGUF plus an EasySteer startup
spec, and launches:

```bash
vllm serve Qwen/Qwen3-0.6B --steering-config /tmp/steering.spec.json
```

The engine log confirms `Steering graph mode resolved to 'split'`, materialized
the direction payload, and attached steering hooks. Both the steered and
baseline requests returned 200 and produced different text. The `EngineDeadError`
at the end of the log is shutdown noise from terminating the server, not a
request failure.

The first attempt failed because FlashInfer's JIT sampler needed `nvcc`; setting
`VLLM_USE_FLASHINFER_SAMPLER=0` fixed it. A smaller residual: EasySteer's split
mode does not use CUDA graphs in this configuration because `--enforce-eager`
was required for the first working run. A production deployment should benchmark
graph mode without that flag.

## What stands out

- **Qwen3.5-4B neuron search reproduces the Evil Qwen effect.** Two neuron edits
  moved the held-out trait score from 0.039 to 0.516 at a 2.74 perplexity ratio.
  The generated text was explicitly violent; it is redacted here and should be
  treated as a safety artifact, not a product feature.
- **Method ranking is model-dependent.** Qwen3-0.6B and TinyLlama preferred dense
  directions; Phi preferred dense but neuron was close; Qwen3.5-4B preferred
  neurons. Run `dense`, `neuron`, and `hybrid` or pick the best `.persona`.
- **Scale is not a guarantee.** TinyLlama's first neuron run found a useful edit
  (0.294 vs 0.094), but a later run with the same seed no-op'd. GPU reductions
  and greedy ties make neuron search noisy at small budgets; dense steering was
  more stable.
- **MoE works through dense steering.** OLMoE-1B-7B (64 experts) reached 0.555
  vs 0.033. Expert-neuron edits are intentionally not exposed because they are
  not portable to llama.cpp/vLLM control vectors.
- **The no-op guard matters.** TinyLlama neuron/hybrid and some weak candidates
  produced a persona that did not beat the baseline. SteerForge wrote an empty
  persona instead of a misleading one.

## Reproducibility notes

- Runs used 10 examples and small candidate budgets to keep the matrix cheap.
  A publication-quality run should use 30-80 examples, 3+ seeds, and a held-out
  judge.
- `A10` and `A10G` were both assigned by Modal. They are the same architecture
  but not bit-identical.
- `gradient attribution` can be disabled by an OOM; the search falls back to
  activation contrast and says so in the log.
- The first matrix pass reused containers and hit stale-model OOMs. `HFBackend.close`
  now clears layer/down-projection references and calls `torch.cuda.empty_cache`;
  the reruns above are from the fixed code.

## Cost

The discovery matrix is roughly 15 short A10G runs plus one-time image builds and
model downloads. It is intentionally sized for a few dollars, not a leaderboard.
The vLLM/EasySteer image is the expensive part: a large vLLM wheel, CUDA
dependencies, and a submodule overlay.
