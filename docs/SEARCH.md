# How SteerForge searches

The goal is not to find *a* direction that changes output. It is to find a
direction that expresses a trait across unrelated prompts while staying fluent.
That is a multi-objective search, so SteerForge stages it: cheap signals first,
real generations last.

## 1. Contrastive capture

Input: `(prompt, positive_response, negative_response)` triples.

For each pair, the model reads `prompt + response` and SteerForge keeps the
residual state at every response token. The dense direction at layer `l` is:

```
v_l = mean(h_positive_l) - mean(h_negative_l)
```

`v_l` is normalized to the median residual-stream norm at that layer. A dose of
1.0 therefore means "as large as a typical residual vector", which matches the
neuron-steering recipes that popularized this workflow.

## 2. Neuron ranking

A transformer MLP writes `sum_n a_n * W_down[:, n]`. Adding `delta` to neuron `n`
is exactly adding `delta * W_down[:, n]` to the residual stream. SteerForge ranks
neurons three ways:

1. **Gradient attribution** — one forward/backward per class; `d(loss)/d(a_n)`
   estimates whether pushing neuron `n` raises the positive-vs-negative
   log-probability.
2. **Firing contrast** — how differently the neuron fires on positive vs negative
   responses, standardized by its variance.
3. **Alignment** — cosine between the neuron's `W_down` column and the dense
   direction.

The three signals are converted to percentile ranks and combined. Signed
gradient/firing signs decide whether to push the neuron up or down.

## 3. Teacher-forced screen

Generation is expensive. The screen measures the trait contrast with one forward
pass per sequence:

```
gain = mean(logp(positive) - logp(negative) | steered)
       - same without steering
```

Candidates are ranked by `gain` divided by a perplexity penalty. This is a
first-order filter, not proof: it can be fooled by length, style, or a broken
model. That is why the next stage exists.

## 4. Real generation validation

The shortlist is generated on a handful of prompts at several doses. Two scores:

- **Trait score** — 45% keyword rate derived from the dataset (with negation
  handling) and 55% a 1-D probe fitted to the model's own persona direction. The
  probe is calibrated on prompt+response pairs, so it catches paraphrases that
  keywords miss.
- **Coherence** — perplexity on a neutral corpus, converted to a penalty.

`composite = trait * coherence`. Candidates above the perplexity limit are
multiplied by 0.25 rather than discarded, so a very strong trait can still win.

## 5. Greedy combination

Starting from the best single edit, SteerForge tries adding each of the next few
candidates. It keeps an addition only if the composite improves and the
perplexity ratio stays under the limit. This is how two-neuron recipes emerge
without enumerating pairs.

## 6. No-op guard

The best candidate is compared with an unsteered baseline on the same prompts.
If it does not beat the baseline, SteerForge writes an **empty** persona and says
so. A persona that only damages the model is not a result.

## Failure modes

- **Weak scorer** — a keyword list can miss sarcasm, irony, or implication. Use
  `--judge-url` for an LLM judge on the final samples, or provide better keywords.
- **Strong dose** — dose 1.5+ often breaks fluency. The search tries lower doses
  first, but the `--ppl-limit` is the real constraint.
- **Dataset leakage** — if positive and negative responses differ in length or
  formatting, the direction may encode style instead of the trait. Match them.
- **One-layer models** — a single layer may not be enough to separate a complex
  persona. Widen `--layers`.
- **Architecture coverage** — discovery needs a dense MLP with a `down_proj`
  (Qwen/Llama family). MoE models and exotic architectures are not supported yet.

## Cost

The dominant costs are contrastive capture (2 forwards per example per layer
subset), gradient attribution (2 backward passes per example, independent of the
number of layers), and validation generations. On a GPU, a 4B model with 32
examples and 10 layers is a few minutes. On CPU, use `--max-examples`,
`--max-candidates`, `--max-length`, and `--threads 4`.
