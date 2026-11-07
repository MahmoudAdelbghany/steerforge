"""SteerForge command line interface."""

from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .dataset import (
    DEFAULT_PROMPTS,
    TRAIT_PRESETS,
    TraitSpec,
    dataset_hash,
    derive_keywords,
    load_dataset,
    load_prompts,
    save_dataset,
    synthesize_dataset,
)
from .export import write_control_vector, write_easysteer_spec
from .launcher import (
    build_llamacpp_command,
    build_vllm_command,
    control_vector_cache,
    detect_backend,
    easy_steer_available,
    find_llama_server,
    normalize_backend,
    run_process,
    serve_config_from_command,
)
from .persona import ServeConfig, load_persona, save_persona
from .search import JudgeConfig, SearchConfig, discover
from .utils import human_int, safe_name

app = typer.Typer(
    name="steerforge",
    help="Discover, package, and serve activation-steered personas.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"steerforge {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """Unsloth-style persona discovery for local inference servers."""


def _parse_doses(value: str) -> tuple[float, ...]:
    doses = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not doses:
        raise typer.BadParameter("at least one dose is required")
    return doses


def _parse_layers(value: str) -> str:
    return value.strip() or "auto"


def _backend_or_die(name: str | None, command: str | None) -> str:
    if command:
        detected = normalize_backend(name) or detect_backend(shlex.split(command))
        if detected:
            return detected
    normalized = normalize_backend(name)
    if normalized:
        return normalized
    raise typer.BadParameter("pass --backend (llama.cpp, transformers, vllm) or --command")


def _load_backend(args) -> object:
    from .hf import HFBackend

    try:
        return HFBackend(
            args.model,
            revision=args.revision,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            load_in_4bit=args.load_in_4bit,
            verbose=True,
        )
    except Exception as exc:
        console.print(f"[red]failed to load model:[/red] {exc}")
        raise typer.Exit(1) from exc


def _generate_fn(backend, system: str, max_tokens: int):
    def run(prompts, sys_prompt, tokens):
        return [
            backend.generate(
                prompt,
                system=sys_prompt,
                max_new_tokens=tokens,
                temperature=0.9,
                repetition_penalty=1.05,
            )
            for prompt in prompts
        ]

    return run


@app.command("doctor")
def doctor() -> None:
    """Check the local toolchain."""
    table = Table(title="SteerForge doctor")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Details")

    def add(name: str, ok: bool, detail: str = "") -> None:
        table.add_row(name, "[green]ok[/green]" if ok else "[yellow]missing[/yellow]", detail)

    add("python", True, sys.version.split()[0])
    for module in ("torch", "transformers", "accelerate", "gguf", "fastapi", "uvicorn", "yaml"):
        spec = importlib.util.find_spec(module)
        detail = ""
        if spec:
            try:
                imported = __import__(module)
                detail = getattr(imported, "__version__", "")
            except Exception:
                detail = "import failed"
        add(module, spec is not None, detail)
    try:
        import torch

        add("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
    except Exception:
        add("cuda", False, "torch not installed")
    server = find_llama_server()
    add("llama-server", bool(server), server or "not on PATH")
    add("vllm", importlib.util.find_spec("vllm") is not None, "steering requires EasySteer fork")
    add("steerforge", True, __version__)
    console.print(table)
    console.print(
        Panel(
            "Discovery needs [bold]torch + transformers[/bold]. Serving needs either "
            "[bold]llama.cpp[/bold] (control-vector GGUF) or the built-in Transformers server.\n"
            "Install extras: [cyan]pip install 'steerforge\\[hf,serve]'[/cyan]",
            title="next",
        )
    )


@app.command("dataset")
def dataset_command(
    action: str = typer.Argument(..., help="make | stats"),
    spec_path: Path | None = typer.Option(None, "--spec", help="Trait YAML spec."),
    trait: str | None = typer.Option(None, "--trait", help="Preset name (evil, sarcastic, pirate)."),
    prompts: Path | None = typer.Option(None, "--prompts", help="Prompt file: JSONL or one per line."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output JSONL dataset."),
    dataset: Path | None = typer.Option(None, "--dataset", help="Dataset to inspect for stats."),
    model: str | None = typer.Option(None, "--model", help="Model used to synthesize responses."),
    tokenizer: str | None = typer.Option(None, "--tokenizer", help="Override tokenizer id/path."),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    threads: int | None = typer.Option(None, "--threads", help="CPU threads (default: 4)."),
    revision: str | None = typer.Option(None, "--revision"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
    max_examples: int = typer.Option(64, "--max-examples"),
    max_new_tokens: int = typer.Option(160, "--max-new-tokens"),
) -> None:
    """Build or inspect a contrastive persona dataset."""
    if action == "stats":
        if not dataset:
            raise typer.BadParameter("--dataset is required for stats")
        examples = load_dataset(dataset)
        keywords = derive_keywords(examples)
        table = Table(title=str(dataset))
        table.add_column("field")
        table.add_column("value")
        table.add_row("examples", str(len(examples)))
        table.add_row("sha256", dataset_hash(examples)[:16])
        table.add_row("keywords", ", ".join(keywords[:20]))
        table.add_row("avg positive chars", str(int(sum(len(e.positive) for e in examples) / len(examples))))
        table.add_row("avg negative chars", str(int(sum(len(e.negative) for e in examples) / len(examples))))
        console.print(table)
        return
    if action != "make":
        raise typer.BadParameter("action must be 'make' or 'stats'")
    if not out:
        raise typer.BadParameter("--out is required")
    if spec_path:
        trait_spec, base = TraitSpec.load(spec_path)
        prompts_path = prompts or trait_spec.prompt_path(base)
    else:
        if not trait:
            raise typer.BadParameter("pass --spec or --trait")
        if trait not in TRAIT_PRESETS:
            raise typer.BadParameter(f"unknown preset {trait!r}; choose from {', '.join(TRAIT_PRESETS)}")
        trait_spec = TRAIT_PRESETS[trait]
        prompts_path = prompts
    prompt_list = load_prompts(prompts_path) if prompts_path else list(DEFAULT_PROMPTS)
    if not model:
        raise typer.BadParameter("--model is required to synthesize a dataset")
    from .hf import HFBackend

    console.print(f"loading [bold]{model}[/bold] for synthesis…")
    backend = HFBackend(
        model,
        revision=revision,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        tokenizer_id=tokenizer,
        threads=threads,
        verbose=True,
    )
    try:
        examples = synthesize_dataset(
            prompt_list,
            trait=trait_spec,
            generate_fn=lambda ps, system, tokens: [
                backend.generate(
                    p,
                    system=system,
                    max_new_tokens=tokens,
                    temperature=trait_spec.temperature,
                    repetition_penalty=1.05,
                )
                for p in ps
            ],
            max_examples=max_examples,
        )
    finally:
        backend.close()
    save_dataset(examples, out)
    console.print(f"wrote [bold]{len(examples)}[/bold] examples to [cyan]{out}[/cyan]")


@app.command("discover")
def discover_command(
    model: str = typer.Option(..., "--model", "-m", help="HF model id or local path."),
    trait: str = typer.Option(..., "--trait", help="Persona name, e.g. evil."),
    dataset: Path | None = typer.Option(None, "--dataset", help="Contrastive JSONL dataset."),
    spec_path: Path | None = typer.Option(None, "--spec", help="Trait YAML spec; can synthesize the dataset."),
    prompts: Path | None = typer.Option(None, "--prompts", help="Prompt file for synthesis."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Output .persona path."),
    method: str = typer.Option("hybrid", "--method", help="neuron | dense | hybrid"),
    layers: str = typer.Option("auto", "--layers", help="auto, all, 8-24, or 8,14,20"),
    max_candidates: int = typer.Option(64, "--max-candidates"),
    screen_candidates: int = typer.Option(32, "--screen-candidates"),
    validate_candidates: int = typer.Option(10, "--validate-candidates"),
    max_examples: int | None = typer.Option(None, "--max-examples"),
    grad_examples: int = typer.Option(16, "--grad-examples"),
    max_length: int = typer.Option(384, "--max-length"),
    max_prompts: int = typer.Option(6, "--max-prompts"),
    max_new_tokens: int = typer.Option(56, "--max-new-tokens"),
    doses: str = typer.Option("0.5,0.75,1.0", "--doses"),
    ppl_limit: float = typer.Option(2.5, "--ppl-limit"),
    max_neurons: int = typer.Option(4, "--max-neurons"),
    keywords: str | None = typer.Option(None, "--keywords", help="Comma-separated trait keywords."),
    judge_url: str | None = typer.Option(None, "--judge-url", help="OpenAI-compatible base URL."),
    judge_model: str | None = typer.Option(None, "--judge-model"),
    judge_key: str | None = typer.Option(None, "--judge-key"),
    serve_command: str | None = typer.Option(None, "--serve-command", help="Existing server command to remember."),
    serve_model: str | None = typer.Option(None, "--serve-model", help="GGUF path for llama.cpp serving."),
    tokenizer: str | None = typer.Option(None, "--tokenizer", help="Override tokenizer id/path."),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    threads: int | None = typer.Option(None, "--threads", help="CPU threads (default: 4)."),
    revision: str | None = typer.Option(None, "--revision"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
    load_in_4bit: bool = typer.Option(False, "--load-in-4bit"),
    seed: int = typer.Option(0, "--seed"),
    report: Path | None = typer.Option(None, "--report", help="Write a JSON search report."),
) -> None:
    """Search a model for a persona and write a .persona file."""
    from .hf import HFBackend

    if method not in ("neuron", "dense", "hybrid"):
        raise typer.BadParameter("--method must be neuron, dense, or hybrid")
    out_path = out or Path(f"{safe_name(trait)}.persona")
    backend = HFBackend(
        model,
        revision=revision,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        load_in_4bit=load_in_4bit,
        tokenizer_id=tokenizer,
        threads=threads,
        verbose=True,
    )
    try:
        if dataset:
            examples = load_dataset(dataset)
            trait_description = ""
            spec_tags: list[str] = []
            if spec_path:
                trait_spec, _ = TraitSpec.load(spec_path)
                trait = trait or trait_spec.name
                trait_description = trait_spec.description
                spec_tags = list(trait_spec.tags)
        elif spec_path:
            trait_spec, base = TraitSpec.load(spec_path)
            trait = trait_spec.name
            trait_description = trait_spec.description
            spec_tags = list(trait_spec.tags)
            prompts_path = trait_spec.prompt_path(base)
            if prompts:
                prompt_list = load_prompts(prompts)
            elif prompts_path:
                prompt_list = load_prompts(prompts_path)
            else:
                prompt_list = list(DEFAULT_PROMPTS)
            console.print(f"synthesizing [bold]{len(prompt_list)}[/bold] contrastive examples…")
            examples = synthesize_dataset(
                prompt_list,
                trait=trait_spec,
                generate_fn=lambda ps, system, tokens: [
                    backend.generate(
                        p,
                        system=system,
                        max_new_tokens=tokens,
                        temperature=trait_spec.temperature,
                        repetition_penalty=1.05,
                    )
                    for p in ps
                ],
                max_examples=max_examples,
            )
        elif trait in TRAIT_PRESETS:
            trait_spec = TRAIT_PRESETS[trait]
            trait_description = trait_spec.description
            spec_tags = list(trait_spec.tags)
            prompt_list = load_prompts(prompts) if prompts else list(DEFAULT_PROMPTS)
            console.print(f"synthesizing [bold]{len(prompt_list)}[/bold] contrastive examples…")
            examples = synthesize_dataset(
                prompt_list,
                trait=trait_spec,
                generate_fn=lambda ps, system, tokens: [
                    backend.generate(
                        p,
                        system=system,
                        max_new_tokens=tokens,
                        temperature=trait_spec.temperature,
                        repetition_penalty=1.05,
                    )
                    for p in ps
                ],
                max_examples=max_examples,
            )
        else:
            raise typer.BadParameter("pass --dataset or --spec, or use a built-in --trait preset")
        config_keywords = [part.strip() for part in (keywords or "").split(",") if part.strip()]
        if spec_path:
            spec_keywords = [item for item in trait_spec.keywords if item not in config_keywords]
            config_keywords = spec_keywords + config_keywords
        judge = None
        if judge_url and judge_model:
            judge = JudgeConfig(base_url=judge_url, model=judge_model, api_key=judge_key)
        elif judge_url or judge_model:
            raise typer.BadParameter("--judge-url and --judge-model must be used together")
        serve = None
        if serve_command:
            serve = serve_config_from_command(serve_command)
        elif serve_model:
            serve = ServeConfig(backend="llama.cpp", model=serve_model, command=["llama-server", "-m", serve_model])
        config = SearchConfig(
            method=method,  # type: ignore[arg-type]
            layers=_parse_layers(layers),
            max_candidates=max_candidates,
            screen_candidates=screen_candidates,
            validate_candidates=validate_candidates,
            max_examples=max_examples,
            grad_max_examples=grad_examples,
            max_length=max_length,
            max_prompts=max_prompts,
            max_new_tokens=max_new_tokens,
            doses=_parse_doses(doses),
            ppl_limit=ppl_limit,
            max_neurons=max_neurons,
            keywords=config_keywords,
            judge=judge,
            seed=seed,
            description=trait_description,
            tags=spec_tags,
            serve=serve,
            log=lambda message: console.print(f"[dim]{message}[/dim]"),
        )
        persona = discover(backend, examples, trait, config)
    finally:
        backend.close()
    saved = save_persona(persona, out_path)
    console.print(
        Panel(
            f"[bold]{saved}[/bold]\n"
            f"trait score {persona.eval.trait_score:.3f} "
            f"(baseline {persona.eval.baseline_trait_score:.3f})  ·  "
            f"ppl ratio {persona.eval.ppl_ratio:.2f}\n"
            f"layers: {', '.join(str(layer) for layer in persona.vector_layers()) or 'none'}  ·  "
            f"{len(persona.interventions)} neuron edit(s)",
            title="persona written",
        )
    )
    if not persona.vectors:
        console.print(
            "[yellow]warning:[/yellow] no candidate beat the baseline; this is a "
            "no-op persona. Add data, a judge, or search a larger model."
        )
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(persona.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        console.print(f"report: [cyan]{report}[/cyan]")


@app.command("show")
def show_command(
    persona_path: Path = typer.Argument(..., exists=True, readable=True),
    samples: int = typer.Option(3, "--samples"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect a .persona file."""
    persona = load_persona(persona_path)
    if json_out:
        console.print_json(data=persona.model_dump(mode="json"))
        return
    table = Table(title=f"{persona.name} · {persona.base_model.id}")
    table.add_column("field")
    table.add_column("value")
    rows = [
        ("created", persona.created_at),
        ("architecture", persona.base_model.architecture or "?"),
        ("layers", str(persona.base_model.num_layers)),
        ("hidden", str(persona.base_model.hidden_size)),
        ("parameters", human_int(persona.base_model.num_parameters)),
        ("trait score", f"{persona.eval.trait_score:.3f}" if persona.eval.trait_score is not None else "?"),
        ("baseline", f"{persona.eval.baseline_trait_score:.3f}" if persona.eval.baseline_trait_score is not None else "?"),
        ("ppl", f"{persona.eval.perplexity:.2f}" if persona.eval.perplexity else "?"),
        ("ppl ratio", f"{persona.eval.ppl_ratio:.2f}" if persona.eval.ppl_ratio else "?"),
        ("coherence", f"{persona.eval.coherence:.3f}" if persona.eval.coherence is not None else "?"),
        ("method", persona.eval.method or "?"),
        ("scorer", persona.eval.scorer or "?"),
        (
            "search",
            f"{persona.search.method} · layers {persona.search.layers} · seed {persona.search.seed}"
            if persona.search
            else "—",
        ),
        ("keywords", ", ".join(persona.trait.keywords[:14])),
        ("serve", f"{persona.serve.backend}: {shlex.join(persona.serve.command)}" if persona.serve else "—"),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)

    if persona.interventions:
        itable = Table(title="interventions")
        for column in ("layer", "neuron", "delta", "dose", "source"):
            itable.add_column(column)
        for item in persona.interventions:
            itable.add_row(
                str(item.layer),
                str(item.neuron),
                f"{item.delta:+.4f}",
                f"{item.dose}" if item.dose is not None else "—",
                item.source or "—",
            )
        console.print(itable)

    if persona.eval.samples and samples:
        for sample in persona.eval.samples[:samples]:
            console.print(Panel(sample.prompt, title="prompt", border_style="blue"))
            console.print(Panel(sample.baseline or "—", title="baseline", border_style="dim"))
            console.print(Panel(sample.steered or "—", title="steered", border_style="red"))


@app.command("export")
def export_command(
    persona_path: Path = typer.Argument(..., exists=True, readable=True),
    out: Path = typer.Option(..., "--out", "-o", help="Output path (.gguf)."),
    scale: float = typer.Option(1.0, "--scale"),
) -> None:
    """Export a persona as a llama.cpp control-vector GGUF."""
    persona = load_persona(persona_path)
    target = write_control_vector(persona, out, scale=scale)
    console.print(f"wrote [cyan]{target}[/cyan] ({target.stat().st_size / 1024:.1f} KiB)")


@app.command("apply")
def apply_command(
    persona_path: Path = typer.Argument(..., exists=True, readable=True),
    prompt: str = typer.Option(..., "--prompt", "-p"),
    model: str | None = typer.Option(None, "--model", help="Override base model."),
    system: str | None = typer.Option(None, "--system"),
    max_new_tokens: int = typer.Option(160, "--max-new-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    baseline: bool = typer.Option(False, "--baseline", help="Also print the unsteered answer."),
    tokenizer: str | None = typer.Option(None, "--tokenizer"),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    threads: int | None = typer.Option(None, "--threads", help="CPU threads (default: 4)."),
    revision: str | None = typer.Option(None, "--revision"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
) -> None:
    """Run one prompt through a persona (useful for smoke tests)."""
    from .hf import HFBackend

    persona = load_persona(persona_path)
    if not persona.vectors:
        console.print("[yellow]persona has no vectors; this will be the baseline model[/yellow]")
    backend = HFBackend(
        model or persona.base_model.id,
        revision=revision or persona.base_model.revision,
        device=device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        tokenizer_id=tokenizer,
        threads=threads,
        verbose=True,
    )
    try:
        if baseline:
            console.print(Panel(backend.generate(prompt, system=system, max_new_tokens=max_new_tokens, temperature=temperature), title="baseline"))
        text = backend.generate(
            prompt,
            system=system,
            vectors=persona.all_vectors(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        console.print(Panel(text, title=f"{persona.name} (steered)"))
    finally:
        backend.close()


@app.command("serve")
def serve_command(
    persona_path: Path = typer.Argument(..., exists=True, readable=True),
    backend_name: str | None = typer.Option(None, "--backend", help="llama.cpp | transformers | vllm"),
    model: str | None = typer.Option(None, "--model", help="Override model path/id."),
    command: str | None = typer.Option(None, "--command", help="Existing server command."),
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    scale: float = typer.Option(1.0, "--scale", help="Multiply all persona vectors."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the final command and exit."),
    no_steer: bool = typer.Option(False, "--no-steer", help="Launch the baseline server without the persona."),
    server_arg: list[str] = typer.Option([], "--server-arg", help="Extra server argument; repeatable."),
    tokenizer: str | None = typer.Option(None, "--tokenizer"),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    threads: int | None = typer.Option(None, "--threads", help="CPU threads (default: 4)."),
    revision: str | None = typer.Option(None, "--revision"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
    api_key: str | None = typer.Option(None, "--api-key", help="Optional bearer token for the Transformers server."),
) -> None:
    """Launch an inference server with the persona already applied."""
    persona = load_persona(persona_path)
    effective = persona if scale == 1.0 else persona.scaled(scale)
    if not no_steer and not effective.vectors:
        console.print("[red]persona contains no vectors[/red]")
        raise typer.Exit(1)

    existing = shlex.split(command) if command else (persona.serve.command if persona.serve else None)
    backend = normalize_backend(backend_name) or (persona.serve.backend if persona.serve else None) or detect_backend(existing)
    if not backend:
        raise typer.BadParameter("pass --backend or --command")
    resolved_host = host or (persona.serve.host if persona.serve else None)
    resolved_port = port or (persona.serve.port if persona.serve else None)

    if backend == "llama.cpp":
        if existing and existing[0].endswith("llama-server") is False and "llama-server" in " ".join(existing):
            # A wrapped command such as `sh -c "... llama-server ..."` cannot be
            # safely rewritten; still try to append the control vector.
            pass
        cv_path = control_vector_cache(effective, scale=1.0)
        if not no_steer:
            write_control_vector(effective, cv_path, scale=1.0)
        final = build_llamacpp_command(
            effective,
            control_vector=cv_path,
            command=existing,
            model=model,
            host=resolved_host,
            port=resolved_port,
            scale=1.0,
            extra_args=server_arg,
            no_steer=no_steer,
        )
        console.print(Panel(shlex.join(final), title="llama.cpp"))
        raise typer.Exit(run_process(final, dry_run=dry_run))

    if backend == "transformers":
        from .hf import HFBackend
        from .openai_server import create_app

        model_id = model or (persona.serve.model if persona.serve else None) or persona.base_model.id
        console.print(f"loading [bold]{model_id}[/bold] with {len(effective.vectors)} steered layer(s)")
        hf = HFBackend(
            model_id,
            revision=revision or persona.base_model.revision,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tokenizer_id=tokenizer,
            threads=threads,
            verbose=True,
        )
        active = effective if not no_steer else effective.model_copy(update={"vectors": {}})
        web = create_app(hf, active, api_key=api_key)
        import uvicorn

        uvicorn.run(
            web,
            host=resolved_host or "127.0.0.1",
            port=resolved_port or 8080,
            log_level="info",
        )
        return

    if backend == "vllm":
        if not no_steer and not easy_steer_available():
            console.print(
                Panel(
                    "vLLM mainline does not expose activation steering. Install the "
                    "EasySteer vLLM fork, or serve with --backend transformers / llama.cpp.\n"
                    "You can still launch a baseline with --no-steer.",
                    title="vLLM needs EasySteer",
                    border_style="yellow",
                )
            )
            raise typer.Exit(2)
        model_id = model or (persona.serve.model if persona.serve else None) or persona.base_model.id
        base_command = existing or ["vllm", "serve", model_id]
        spec_path = None
        if not no_steer:
            cv_path = control_vector_cache(effective)
            write_control_vector(effective, cv_path, scale=1.0)
            spec_path = cv_path.with_suffix(".spec.json")
            write_easysteer_spec(
                effective,
                cv_path,
                spec_path,
                layers=[layer for layer in effective.vector_layers() if layer >= 1],
            )
        final = build_vllm_command(
            effective,
            steering_config=spec_path,
            command=base_command,
            host=resolved_host,
            port=resolved_port,
            extra_args=server_arg,
            no_steer=no_steer,
        )
        console.print(Panel(shlex.join(final), title="vLLM / EasySteer"))
        raise typer.Exit(run_process(final, dry_run=dry_run))

    raise typer.BadParameter(f"unsupported backend: {backend}")


if __name__ == "__main__":  # pragma: no cover
    app()
