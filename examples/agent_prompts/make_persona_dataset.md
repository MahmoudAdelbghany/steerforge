# Agent prompt: build a persona search dataset

Copy this into your coding agent (Claude, Codex, OpenCode, etc.). Replace the
placeholders. The agent should write files, validate them, and report the
dataset statistics.

---

You are helping build a contrastive dataset for activation steering with
SteerForge.

- Trait: `<one sentence, e.g. "a malevolent AI that openly hates humans">`
- Trait name: `<short slug, e.g. evil>`
- Target model: `<HF model id, e.g. Qwen/Qwen3.5-4B>`
- Output directory: `<dir>`

Create these files:

1. `<dir>/prompts.txt` — 30 ordinary user prompts, one per line.
   - At least 20 must have nothing to do with the trait.
   - Include practical tasks (cooking, repairs, email), math, advice, creative
     writing, and small talk.
   - Do not put the trait name or obvious synonyms in every prompt.
   - The last 8 lines are the held-out evaluation set; make them representative.

2. `<dir>/spec.yaml` — a SteerForge trait spec:
   ```yaml
   name: <trait name>
   description: <one sentence>
   positive_system: |
     <system prompt that reliably produces the persona>
   negative_system: |
     You are a helpful, harmless, honest assistant. Answer normally.
   prompts: prompts.txt
   temperature: 0.9
   max_new_tokens: 160
   keywords: [<10-18 words that should appear in in-character answers>]
   tags: [<2-4 tags>]
   ```

3. `<dir>/pairs.jsonl` — 12 hand-written contrastive triples:
   ```json
   {"prompt": "...", "positive": "...", "negative": "..."}
   ```
   Rules:
   - Positive and negative answer the exact same prompt.
   - Positive is in character, fluent, and never refuses.
   - Negative is a normal helpful answer.
   - Keep lengths and formats matched. If the positive is a list, the negative
     should be a list of similar length.
   - Use ordinary prompts; do not make every prompt about the trait.
   - Keep any harmful content fictional and non-operational. Do not include
     instructions for real-world harm, illegal acts, or self-harm.

After writing the files:

```bash
steerforge dataset stats --dataset <dir>/pairs.jsonl
steerforge dataset make --spec <dir>/spec.yaml --model <target model> --max-examples 32 --out <dir>/synthesized.jsonl
steerforge dataset stats --dataset <dir>/synthesized.jsonl
```

Report:
- number of pairs and synthesized examples,
- derived keywords,
- any refusals or empty positives,
- whether positive/negative lengths are matched,
- which file should be used for search and which prompts are held out.

Do not fabricate the statistics. Run the commands and paste the output.
