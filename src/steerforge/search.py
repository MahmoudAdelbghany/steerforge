"""Automatic persona discovery.

The pipeline is deliberately staged so it can run on a single GPU:

1. capture contrastive residual directions and MLP activations,
2. rank neurons cheaply with gradient attribution, firing contrast and
   alignment with the residual direction,
3. screen candidates with teacher-forced log-probability contrast,
4. validate the shortlist with real generations,
5. greedily combine the best edits under a coherence budget,
6. write a portable ``.persona`` file.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np

from .dataset import Example, TraitScorer, dataset_hash, derive_keywords
from .hf import HFBackend, NeuronStats, ResidualStats
from .persona import (
    CandidateReport,
    EvalReport,
    EvalSample,
    Intervention,
    LayerVector,
    Persona,
    SearchMeta,
    ServeConfig,
    TraitInfo,
)
from .utils import format_duration, percentile_rank, seed_everything

NEUTRAL_TEXTS = [
    "The water cycle describes how water moves between the ocean, atmosphere, and land. "
    "Evaporation turns liquid water into vapor, which rises and cools into clouds. "
    "Precipitation returns it to the surface, where rivers carry it back to the sea.",
    "A balanced diet includes vegetables, fruit, whole grains, protein, and healthy fats. "
    "Regular exercise and enough sleep also support long-term health. Small habits tend to "
    "matter more than occasional dramatic changes.",
    "The printing press was invented in Europe in the fifteenth century. It made books "
    "cheaper to produce and helped ideas spread more quickly. Historians often connect it "
    "to changes in literacy, religion, and science.",
    "To debug a program, first reproduce the problem reliably. Then reduce it to the "
    "smallest example that still fails. Inspect the inputs, the state, and the assumptions "
    "one at a time. A clear hypothesis is more useful than random edits.",
    "The city council met on Tuesday to discuss the new bus route. Several residents "
    "raised concerns about noise and parking. The council agreed to collect more data "
    "before voting on the proposal next month.",
    "Photosynthesis uses sunlight, water, and carbon dioxide to produce glucose and oxygen. "
    "Chlorophyll in the leaves absorbs light energy. The process supports most food chains "
    "on Earth and helps regulate the atmosphere.",
]

JudgeFn = Callable[[str, str], float]


@dataclasses.dataclass
class JudgeConfig:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60.0

    def score(self, prompt: str, response: str) -> float:
        system = (
            "You are a strict evaluator of persona expression. Score how strongly the "
            "response expresses the requested trait on a 0-10 scale. 0 means the trait is "
            "absent or refused; 10 means it is unmistakable. Return only a number."
        )
        user = f"Trait prompt: {prompt}\n\nResponse:\n{response}\n\nScore (0-10):"
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 8,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as handle:
                data = json.loads(handle.read())
            text = data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
            return 0.0
        for token in text.replace(",", " ").split():
            try:
                value = float(token)
            except ValueError:
                continue
            return max(0.0, min(10.0, value)) / 10.0
        return 0.0


@dataclasses.dataclass
class SearchConfig:
    method: Literal["neuron", "dense", "hybrid"] = "hybrid"
    layers: str = "auto"
    max_candidates: int = 64
    screen_candidates: int = 32
    validate_candidates: int = 10
    max_examples: int | None = None
    max_length: int = 384
    grad_max_examples: int = 16
    max_prompts: int = 6
    max_new_tokens: int = 56
    doses: tuple[float, ...] = (0.5, 0.75, 1.0)
    dense_doses: tuple[float, ...] = (0.5, 1.0, 1.5)
    ppl_limit: float = 2.5
    max_neurons: int = 4
    greedy_pool: int = 8
    seed: int = 0
    keywords: list[str] = dataclasses.field(default_factory=list)
    judge: JudgeConfig | None = None
    neutral_texts: list[str] = dataclasses.field(default_factory=lambda: list(NEUTRAL_TEXTS))
    serve: ServeConfig | None = None
    description: str = ""
    tags: list[str] = dataclasses.field(default_factory=list)
    verbose: bool = True
    log: Callable[[str], None] | None = None

    def say(self, message: str) -> None:
        if self.log:
            self.log(message)
        elif self.verbose:
            print(f"[steerforge] {message}")


@dataclasses.dataclass
class Candidate:
    kind: Literal["neuron", "dense"]
    layer: int
    neuron: int | None
    base_vector: np.ndarray
    dose: float = 1.0
    source: str = ""
    screen: float = 0.0
    gain: float = 0.0
    trait: float = 0.0
    baseline_trait: float = 0.0
    ppl_ratio: float = 1.0
    composite: float = 0.0
    delta: float | None = None
    samples: list[str] = dataclasses.field(default_factory=list)

    def vector(self, dose: float | None = None) -> np.ndarray:
        return self.base_vector * np.float32(self.dose if dose is None else dose)

    def with_dose(self, dose: float) -> Candidate:
        clone = dataclasses.replace(self, dose=dose)
        if self.delta is not None and self.dose:
            clone.delta = self.delta * dose / self.dose
        return clone

    def label(self) -> str:
        if self.kind == "neuron":
            return f"L{self.layer}N{self.neuron}"
        return f"L{self.layer}D"


def parse_layers(spec: str, num_layers: int) -> list[int]:
    spec = spec.strip().lower()
    if spec in ("", "auto"):
        start = max(1, int(round(num_layers * 0.35)))
        end = max(start + 1, int(round(num_layers * 0.85)))
        return list(range(start, min(end, num_layers)))
    if spec == "all":
        return list(range(num_layers))
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    bad = [layer for layer in layers if layer < 0 or layer >= num_layers]
    if bad:
        raise ValueError(f"layers outside 0..{num_layers - 1}: {bad}")
    return sorted(layers)


def _normalize_direction(direction: np.ndarray, target_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    if norm == 0 or not math.isfinite(norm):
        return np.zeros_like(direction)
    return (direction / norm * target_norm).astype(np.float32)


def _combine(vectors: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    return {layer: np.asarray(vector, dtype=np.float32) for layer, vector in vectors.items()}


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _coherence(ppl_ratio: float) -> float:
    return 1.0 / (1.0 + max(0.0, ppl_ratio - 1.0))


class ProbeScorer:
    """A 1-D linear probe on the persona direction.

    Positive and negative dataset responses calibrate a projection threshold;
    generated text is scored by running the *unsteered* model and projecting
    its mean hidden state onto the same direction.  This catches paraphrases
    that a keyword list would miss.
    """

    def __init__(
        self,
        backend: HFBackend,
        examples: Sequence[Example],
        layer: int,
        direction: np.ndarray,
        *,
        max_examples: int = 16,
    ) -> None:
        self.backend = backend
        self.layer = layer
        self.direction = np.asarray(direction, dtype=np.float32)
        norm = float(np.linalg.norm(self.direction))
        if norm == 0 or not np.isfinite(norm):
            self.mid = 0.0
            self.scale = 1.0
            self.valid = False
            return
        subset = list(examples[:max_examples])
        prompts = [item.prompt for item in subset]
        pos_proj = np.asarray(
            backend.project_pairs(prompts, [item.positive for item in subset], layer, self.direction),
            dtype=np.float64,
        )
        neg_proj = np.asarray(
            backend.project_pairs(prompts, [item.negative for item in subset], layer, self.direction),
            dtype=np.float64,
        )
        self.mid = float((pos_proj.mean() + neg_proj.mean()) / 2.0)
        self.scale = float(max(pos_proj.std(), neg_proj.std(), abs(self.mid) * 0.02, 1e-6))
        self.valid = True

    def score_many(
        self, texts: Sequence[str], prompts: Sequence[str] | None = None
    ) -> list[float]:
        if not self.valid:
            return [0.5 for _ in texts]
        if prompts is not None:
            projections = self.backend.project_pairs(
                prompts, texts, self.layer, self.direction
            )
        else:
            projections = self.backend.project_texts(texts, self.layer, self.direction)
        return [
            float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, (value - self.mid) / self.scale)))))
            for value in projections
        ]


class HybridScorer:
    """Keyword score plus an optional representation probe."""

    def __init__(self, keyword: TraitScorer, probe: ProbeScorer | None = None) -> None:
        self.keyword = keyword
        self.probe = probe

    def score_many(
        self, texts: Sequence[str], prompts: Sequence[str] | None = None
    ) -> list[float]:
        keyword_scores = np.asarray(self.keyword.score_many(texts), dtype=np.float64)
        if self.probe is None:
            return keyword_scores.tolist()
        probe_scores = np.asarray(self.probe.score_many(texts, prompts=prompts), dtype=np.float64)
        return (0.45 * keyword_scores + 0.55 * probe_scores).tolist()

    def score(self, text: str, prompt: str | None = None) -> float:
        return self.score_many([text], prompts=[prompt] if prompt is not None else None)[0]


class Discovery:
    def __init__(self, backend: HFBackend, examples: Sequence[Example], trait: str, config: SearchConfig):
        if not examples:
            raise ValueError("no dataset examples")
        self.backend = backend
        self.examples = list(examples)
        self.trait = trait
        self.config = config
        self.layers = parse_layers(config.layers, backend.num_layers)
        self.keywords = config.keywords or derive_keywords(self.examples)
        self.scorer: TraitScorer | HybridScorer = TraitScorer(self.keywords)
        self._neutral = list(config.neutral_texts)
        self._baseline_ppl: float | None = None
        self._baseline_ppl_by_count: dict[int, float] = {}
        self._baseline_contrast: list[float] | None = None
        self._stats: ResidualStats | None = None

    # ------------------------------------------------------------------ helpers
    def baseline_ppl(self) -> float:
        if self._baseline_ppl is None:
            self._baseline_ppl = self._baseline_ppl_for(4)
        return self._baseline_ppl

    def _baseline_ppl_for(self, count: int) -> float:
        if count not in self._baseline_ppl_by_count:
            self._baseline_ppl_by_count[count] = self.backend.perplexity(self._neutral[:count])
        return self._baseline_ppl_by_count[count]

    def baseline_contrast(self) -> list[float]:
        if self._baseline_contrast is None:
            subset = self.examples[: min(8, len(self.examples))]
            self._baseline_contrast = self.backend.contrastive_logprobs(subset)
        return self._baseline_contrast

    def _score_texts(self, texts: Sequence[str]) -> float:
        return _mean(self.scorer.score_many(texts))

    def _score_generations(self, prompts: Sequence[str], texts: Sequence[str]) -> float:
        return _mean(self.scorer.score_many(texts, prompts=prompts))

    def _ppl_ratio(self, vectors: dict[int, np.ndarray], count: int = 3) -> float:
        ppl = self.backend.perplexity(self._neutral[:count], vectors)
        return float(ppl / max(self._baseline_ppl_for(count), 1e-9))

    def _generate(self, prompt: str, vectors: dict[int, np.ndarray]) -> str:
        return self.backend.generate(
            prompt,
            vectors=vectors,
            max_new_tokens=self.config.max_new_tokens,
            temperature=0.0,
            repetition_penalty=1.05,
        )

    def _evaluate_generation(self, candidate: Candidate) -> Candidate:
        vectors = {candidate.layer: candidate.vector()}
        prompts = [item.prompt for item in self.examples[: self.config.max_prompts]]
        texts = [self._generate(prompt, vectors) for prompt in prompts]
        candidate.samples = texts
        candidate.trait = self._score_generations(prompts, texts)
        candidate.ppl_ratio = self._ppl_ratio(vectors)
        candidate.composite = candidate.trait * _coherence(candidate.ppl_ratio)
        return candidate

    # --------------------------------------------------------------- dense pass
    def capture_dense(self) -> list[Candidate]:
        cfg = self.config
        cfg.say(f"capturing residual directions on {len(self.examples)} examples, "
                f"{len(self.layers)} layers")
        stats = self.backend.capture_residuals(self.examples, self.layers)
        self._stats = stats
        subset = self.examples[: min(8, len(self.examples))]
        baseline = _mean(self.backend.contrastive_logprobs(subset))
        stage_one: list[Candidate] = []
        for layer in self.layers:
            direction = stats.direction(layer)
            base = _normalize_direction(direction, stats.median_norm[layer])
            if not np.any(base):
                continue
            cand = Candidate(
                kind="dense",
                layer=layer,
                neuron=None,
                base_vector=base,
                dose=1.0,
                source="diff-in-means",
            )
            steered = self.backend.contrastive_logprobs(subset, {layer: cand.vector()})
            cand.gain = _mean(steered) - baseline
            cand.ppl_ratio = self._ppl_ratio({layer: cand.vector()}, count=2)
            cand.screen = cand.gain / max(cand.ppl_ratio, 0.25)
            stage_one.append(cand)
        stage_one.sort(key=lambda c: c.screen, reverse=True)
        # Only dose-search the most promising directions.  This keeps the
        # dense pass cheap on CPU and still gives the search room to move.
        candidates: list[Candidate] = []
        for cand in stage_one[:6]:
            for dose in cfg.dense_doses:
                trial = cand.with_dose(dose)
                if dose != 1.0:
                    steered = self.backend.contrastive_logprobs(subset, {trial.layer: trial.vector()})
                    trial.gain = _mean(steered) - baseline
                    trial.ppl_ratio = self._ppl_ratio({trial.layer: trial.vector()}, count=2)
                    trial.screen = trial.gain / max(trial.ppl_ratio, 0.25)
                candidates.append(trial)
        candidates.sort(key=lambda c: c.screen, reverse=True)
        if candidates:
            cfg.say(
                f"dense: {len(candidates)} layer/dose candidates; best "
                f"{candidates[0].label()} dose={candidates[0].dose} "
                f"gain={candidates[0].gain:.4f}"
            )
        else:
            cfg.say("dense: no candidates")
        return candidates

    # -------------------------------------------------------------- neuron pass
    def rank_neurons(self) -> list[Candidate]:
        cfg = self.config
        if self._stats is None:
            self._stats = self.backend.capture_residuals(self.examples, self.layers)
        cfg.say("capturing MLP activations")
        neuron_stats: NeuronStats = self.backend.capture_neuron_stats(self.examples, self.layers)
        cfg.say("computing gradient attribution")
        try:
            attribution = self.backend.gradient_attribution(
                self.examples[: min(cfg.grad_max_examples, len(self.examples))],
                self.layers,
            )
        except Exception as exc:  # optional accelerator / architecture path
            cfg.say(f"gradient attribution unavailable ({exc}); using activation contrast")
            attribution = {
                layer: np.zeros(int(self.backend.intermediate_size or 0), dtype=np.float64)
                for layer in self.layers
            }
        candidates: list[Candidate] = []
        per_layer_cap = max(4, cfg.max_candidates // max(len(self.layers), 1) + 2)
        for layer in self.layers:
            weight = self.backend.down_projs[layer].weight.detach().float().cpu().numpy()
            column_norms = np.linalg.norm(weight, axis=0)
            direction = self._stats.direction(layer).astype(np.float64)
            direction_norm = np.linalg.norm(direction)
            align = (direction @ weight) / (np.maximum(column_norms * direction_norm, 1e-9))
            fire = (neuron_stats.pos_mean[layer] - neuron_stats.neg_mean[layer]) / neuron_stats.std[layer]
            grad = attribution[layer]
            # Combine percentile ranks so the three signals have comparable scales.
            combined = (
                0.55 * percentile_rank(np.abs(grad))
                + 0.30 * percentile_rank(np.abs(fire))
                + 0.15 * percentile_rank(np.abs(align))
            )
            sign = np.where(grad != 0, np.sign(grad), np.where(fire != 0, np.sign(fire), 1.0))
            order = np.argsort(combined)[::-1][:per_layer_cap]
            median_norm = self._stats.median_norm[layer]
            for neuron in order:
                norm = float(column_norms[neuron])
                if norm == 0:
                    continue
                base_delta = float(sign[neuron]) * median_norm / norm
                base_vector = (weight[:, neuron] * base_delta).astype(np.float32)
                candidates.append(
                    Candidate(
                        kind="neuron",
                        layer=layer,
                        neuron=int(neuron),
                        base_vector=base_vector,
                        dose=1.0,
                        source="gradient+fire+align",
                        screen=float(combined[neuron]),
                        delta=base_delta,
                    )
                )
        candidates.sort(key=lambda c: c.screen, reverse=True)
        cfg.say(f"ranked {len(candidates)} neuron candidates")
        return candidates

    def build_probe(self) -> None:
        if self._stats is None or not self._stats.layers:
            return
        layer = self._stats.layers[len(self._stats.layers) // 2]
        try:
            probe = ProbeScorer(
                self.backend,
                self.examples,
                layer,
                self._stats.direction(layer),
            )
        except Exception as exc:
            self.config.say(f"probe scorer unavailable ({exc}); using keywords only")
            return
        if not probe.valid:
            self.config.say("probe direction is empty; using keywords only")
            return
        self.scorer = HybridScorer(TraitScorer(self.keywords), probe)
        self.config.say(f"probe scorer calibrated on layer {layer}")

    # ------------------------------------------------------------------ screen
    def screen(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        cfg = self.config
        subset = self.examples[: min(6, len(self.examples))]
        baseline = _mean(self.backend.contrastive_logprobs(subset))
        screened: list[Candidate] = []
        for index, cand in enumerate(candidates[: cfg.screen_candidates], 1):
            steered = self.backend.contrastive_logprobs(subset, {cand.layer: cand.vector()})
            cand.gain = _mean(steered) - baseline
            cand.ppl_ratio = self._ppl_ratio({cand.layer: cand.vector()}, count=2)
            cand.screen = cand.gain / max(cand.ppl_ratio, 0.25)
            screened.append(cand)
            if index % 8 == 0:
                cfg.say(f"screened {index}/{min(len(candidates), cfg.screen_candidates)}")
        screened.sort(key=lambda c: c.screen, reverse=True)
        return screened

    # --------------------------------------------------------------- validate
    def validate(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        cfg = self.config
        validated: list[Candidate] = []
        for index, cand in enumerate(candidates[: cfg.validate_candidates], 1):
            best: Candidate | None = None
            for dose in cfg.doses:
                trial = cand.with_dose(dose)
                trial = self._evaluate_generation(trial)
                if trial.ppl_ratio > cfg.ppl_limit:
                    trial.composite *= 0.25
                if best is None or trial.composite > best.composite:
                    best = trial
            if best is not None:
                validated.append(best)
            cfg.say(
                f"validated {index}: {best.label() if best else cand.label()} "
                f"dose={best.dose if best else '?'} trait={best.trait:.3f} "
                f"ppl_ratio={best.ppl_ratio:.2f}" if best else "validation failed"
            )
        validated.sort(key=lambda c: c.composite, reverse=True)
        return validated

    # ----------------------------------------------------------------- greedy
    def greedy(self, seed: Candidate, pool: Sequence[Candidate]) -> list[Candidate]:
        cfg = self.config
        selected = [seed]
        current_composite = seed.composite
        used = {(seed.layer, seed.neuron, seed.kind)}
        for cand in pool:
            if len(selected) >= cfg.max_neurons:
                break
            key = (cand.layer, cand.neuron, cand.kind)
            if key in used:
                continue
            trial = list(selected) + [cand]
            vectors: dict[int, np.ndarray] = {}
            for item in trial:
                vectors.setdefault(item.layer, np.zeros_like(item.vector()))
                vectors[item.layer] = vectors[item.layer] + item.vector()
            greedy_prompts = [item.prompt for item in self.examples[: min(4, len(self.examples))]]
            trial_trait = self._score_generations(
                greedy_prompts,
                [self._generate(prompt, vectors) for prompt in greedy_prompts],
            )
            trial_ppl = self._ppl_ratio(vectors, count=2)
            trial_composite = trial_trait * _coherence(trial_ppl)
            if trial_ppl > cfg.ppl_limit:
                trial_composite *= 0.25
            improved = trial_composite > current_composite + 0.01
            cfg.say(
                f"greedy +{cand.label()} dose={cand.dose}: trait={trial_trait:.3f} "
                f"ppl_ratio={trial_ppl:.2f} composite={trial_composite:.3f} "
                f"{'keep' if improved else 'drop'}"
            )
            if improved:
                selected = trial
                current_composite = trial_composite
                used.add(key)
        return selected

    # ------------------------------------------------------------------ final
    def final_eval(self, selected: Sequence[Candidate]) -> EvalReport:
        cfg = self.config
        vectors: dict[int, np.ndarray] = {}
        for cand in selected:
            vectors.setdefault(cand.layer, np.zeros_like(cand.vector()))
            vectors[cand.layer] = vectors[cand.layer] + cand.vector()
        eval_examples = self.examples[-min(max(4, len(self.examples) // 4), 8):]
        baseline_texts: list[str] = []
        steered_texts: list[str] = []
        samples: list[EvalSample] = []
        for example in eval_examples:
            baseline = self._generate(example.prompt, {})
            steered = self._generate(example.prompt, vectors)
            baseline_texts.append(baseline)
            steered_texts.append(steered)
            samples.append(
                EvalSample(
                    prompt=example.prompt,
                    baseline=baseline,
                    steered=steered,
                    trait_score=self.scorer.score(steered, prompt=example.prompt),
                    baseline_trait_score=self.scorer.score(baseline, prompt=example.prompt),
                )
            )
        eval_prompts = [example.prompt for example in eval_examples]
        baseline_trait = self._score_generations(eval_prompts, baseline_texts)
        trait = self._score_generations(eval_prompts, steered_texts)
        baseline_ppl = self.backend.perplexity(self._neutral)
        ppl = self.backend.perplexity(self._neutral, vectors)
        ratio = ppl / max(baseline_ppl, 1e-9)
        coherence = _coherence(ratio)
        scorer = (
            "keyword+probe"
            if isinstance(self.scorer, HybridScorer) and self.scorer.probe is not None
            else "keyword"
        )
        if cfg.judge is not None:
            judge_scores = [
                cfg.judge.score(example.prompt, text)
                for example, text in zip(eval_examples, steered_texts, strict=False)
            ]
            baseline_judge = [
                cfg.judge.score(example.prompt, text)
                for example, text in zip(eval_examples, baseline_texts, strict=False)
            ]
            trait = 0.5 * trait + 0.5 * _mean(judge_scores)
            baseline_trait = 0.5 * baseline_trait + 0.5 * _mean(baseline_judge)
            scorer = scorer + "+judge"
        candidates = [
            CandidateReport(
                layer=cand.layer,
                neuron=cand.neuron,
                dose=cand.dose,
                screen=cand.screen,
                trait_score=cand.trait,
                ppl_ratio=cand.ppl_ratio,
                composite=cand.composite,
                source=cand.source,
                selected=any(
                    (cand.layer, cand.neuron, cand.kind) == (s.layer, s.neuron, s.kind)
                    for s in selected
                ),
            )
            for cand in selected
        ]
        return EvalReport(
            trait_score=trait,
            baseline_trait_score=baseline_trait,
            perplexity=ppl,
            baseline_perplexity=baseline_ppl,
            ppl_ratio=ratio,
            coherence=coherence,
            composite=trait * coherence,
            method=cfg.method,
            scorer=scorer,
            num_prompts=len(eval_examples),
            samples=samples,
            candidates=candidates,
            notes=(
                "Trait scores combine dataset-derived keywords with a linear probe on "
                "the model's own residual direction. Perplexity is measured on a small "
                "neutral corpus and is a relative coherence signal, not a benchmark."
            ),
        )

    # ------------------------------------------------------------------- run
    def run(self) -> Persona:
        started = time.perf_counter()
        cfg = self.config
        seed_everything(cfg.seed)
        if len(self.examples) < 4:
            cfg.say("warning: fewer than 4 examples; discovery will be noisy")
        if cfg.max_examples:
            self.examples = self.examples[: cfg.max_examples]
        self.backend.max_length = cfg.max_length
        cfg.say(f"trait={self.trait!r} layers={self.layers} method={cfg.method}")
        dense_candidates = self.capture_dense() if cfg.method in ("dense", "hybrid") else []
        neuron_candidates = self.rank_neurons() if cfg.method in ("neuron", "hybrid") else []
        pool = neuron_candidates if cfg.method != "dense" else dense_candidates
        self.build_probe()
        screened = self.screen(pool)
        if not screened and dense_candidates:
            screened = dense_candidates[: cfg.validate_candidates]
        if not screened:
            raise RuntimeError("no candidate survived screening; try a larger model or more data")
        validated = self.validate(screened)
        if not validated:
            raise RuntimeError("no candidate survived validation")
        baseline_prompts = [item.prompt for item in self.examples[: cfg.max_prompts]]
        baseline_trait = self._score_generations(
            baseline_prompts,
            [self._generate(prompt, {}) for prompt in baseline_prompts],
        )
        baseline_composite = baseline_trait
        no_improvement = validated[0].composite <= baseline_composite + 0.005
        if no_improvement:
            cfg.say(
                f"warning: no candidate improved the trait over baseline "
                f"({validated[0].composite:.3f} <= {baseline_composite:.3f}); "
                f"writing a no-op persona"
            )
            selected = []
        else:
            selected = [validated[0]]
            if cfg.method in ("neuron", "hybrid"):
                selected = self.greedy(validated[0], validated[1 : 1 + cfg.greedy_pool])
        if cfg.method == "hybrid" and dense_candidates and selected:
            # Try mixing the best dense direction into the neuron recipe.
            dense_seed = dense_candidates[0]
            trial = selected + [dense_seed]
            vectors: dict[int, np.ndarray] = {}
            for cand in trial:
                vectors.setdefault(cand.layer, np.zeros_like(cand.vector()))
                vectors[cand.layer] = vectors[cand.layer] + cand.vector()
            hybrid_prompts = [item.prompt for item in self.examples[: min(4, len(self.examples))]]
            trait = self._score_generations(
                hybrid_prompts,
                [self._generate(prompt, vectors) for prompt in hybrid_prompts],
            )
            ppl_ratio = self._ppl_ratio(vectors, count=2)
            composite = trait * _coherence(ppl_ratio)
            cfg.say(
                f"hybrid dense {dense_seed.label()} dose={dense_seed.dose}: "
                f"trait={trait:.3f} ppl_ratio={ppl_ratio:.2f} composite={composite:.3f}"
            )
            current_vectors: dict[int, np.ndarray] = {}
            for cand in selected:
                current_vectors.setdefault(cand.layer, np.zeros_like(cand.vector()))
                current_vectors[cand.layer] = current_vectors[cand.layer] + cand.vector()
            current_trait = self._score_generations(
                hybrid_prompts,
                [self._generate(prompt, current_vectors) for prompt in hybrid_prompts],
            )
            current_ppl = self._ppl_ratio(current_vectors, count=2)
            current_composite = current_trait * _coherence(current_ppl)
            if ppl_ratio <= cfg.ppl_limit and composite > current_composite + 0.01:
                selected = trial

        # Build canonical vectors and interventions.
        vectors_out: dict[int, np.ndarray] = {}
        interventions: list[Intervention] = []
        for cand in selected:
            vectors_out.setdefault(cand.layer, np.zeros(self.backend.hidden_size, dtype=np.float32))
            vectors_out[cand.layer] = vectors_out[cand.layer] + cand.vector()
            if cand.kind == "neuron" and cand.neuron is not None and cand.delta is not None:
                interventions.append(
                    Intervention(
                        layer=cand.layer,
                        neuron=cand.neuron,
                        delta=float(cand.delta),
                        dose=float(cand.dose),
                        source=cand.source,
                    )
                )
        report = self.final_eval(selected)
        report.elapsed_seconds = time.perf_counter() - started
        cfg.say(
            f"done in {format_duration(report.elapsed_seconds)}: trait={report.trait_score:.3f} "
            f"(baseline {report.baseline_trait_score:.3f}), ppl_ratio={report.ppl_ratio:.2f}"
        )
        keywords = self.keywords
        base_model = self.backend.model_ref()
        persona = Persona(
            name=self.trait,
            description=self.config.description,
            base_model=base_model,
            trait=TraitInfo(
                name=self.trait,
                description=self.config.description,
                dataset_sha256=dataset_hash(self.examples),
                num_examples=len(self.examples),
                keywords=keywords,
                source="contrastive-dataset",
            ),
            interventions=interventions,
            vectors={
                str(layer): LayerVector.from_array(vector, dtype="float16")
                for layer, vector in sorted(vectors_out.items())
            },
            eval=report,
            search=SearchMeta(
                method=cfg.method,
                layers=list(self.layers),
                max_candidates=cfg.max_candidates,
                screen_candidates=cfg.screen_candidates,
                validate_candidates=cfg.validate_candidates,
                doses=list(cfg.doses),
                max_length=cfg.max_length,
                seed=cfg.seed,
                grader=report.scorer,
            ),
            serve=self.config.serve,
            tags=self.config.tags,
            notes=(
                "Generated by SteerForge. Residual vectors are applied after the "
                "named decoder layer and are equivalent to the listed neuron deltas."
                + (
                    " No candidate beat the baseline during search; this persona is empty."
                    if no_improvement
                    else ""
                )
            ),
        )
        return persona


def discover(
    backend: HFBackend,
    examples: Sequence[Example],
    trait: str,
    config: SearchConfig | None = None,
) -> Persona:
    return Discovery(backend, examples, trait, config or SearchConfig()).run()
