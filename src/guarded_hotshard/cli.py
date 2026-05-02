"""Command-line interface.

    ghs demo    --backend http://localhost:8001 --model qwen2.5-3b-instruct
    ghs serve   --backend http://localhost:8001 --port 8000 --mode protected_lane
    ghs modes
    ghs version
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from guarded_hotshard._version import __version__
from guarded_hotshard.modes import _FACTORIES as _ALL_MODES  # noqa: PLC2701
from guarded_hotshard.modes import MODES, make_mode

app = typer.Typer(
    name="ghs",
    help="guarded-hotshard - tenant-aware request scheduling for LLM inference.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"guarded-hotshard {__version__}")


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
@app.command()
def modes() -> None:
    """List built-in scheduling modes and their descriptions."""
    table = Table(title="Built-in scheduling modes", show_header=True, header_style="bold cyan")
    table.add_column("name", style="cyan")
    table.add_column("eviction", justify="right")
    table.add_column("tmr", justify="right")
    table.add_column("description")
    for name in MODES:
        m = make_mode(name)
        table.add_row(
            name,
            f"{m.eviction_frac:.2f}",
            f"{m.tmr_frac:.2f}",
            m.description,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
@app.command()
def demo(
    backend: str = typer.Option(..., "--backend", "-b", help="OpenAI-compatible base URL, e.g. http://localhost:8001"),
    model: str = typer.Option(..., "--model", "-m", help="Model name to send to the backend"),
    api_key: str | None = typer.Option(None, "--api-key", "-k", envvar="OPENAI_API_KEY", help="Bearer token, if the backend needs one"),
    n_requests: int = typer.Option(60, "--requests", "-n", help="Total requests across all tenants"),
    n_tenants: int = typer.Option(5, "--tenants", "-t", help="Number of tenants (Zipf-skewed)"),
    seed: int = typer.Option(42, "--seed", help="Workload seed"),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Concurrent requests against the backend"),
    max_tokens: int = typer.Option(32, "--max-tokens", help="max_tokens per chat completion"),
    out: Path = typer.Option(Path("demo_results"), "--out", "-o", help="Output directory"),
    only: str | None = typer.Option(None, "--only", help="Comma-separated subset of modes to run"),
    hourly_cost: float = typer.Option(4.0, "--hourly-cost", help="Backend GPU $/hr for cost modeling"),
    adversarial: bool = typer.Option(False, "--adversarial", help="Inject a replica-storm tenant"),
) -> None:
    """Run the full benchmark against any OpenAI-compatible backend."""
    from guarded_hotshard.demo import render_summary, run_demo_async

    mode_subset = [m.strip() for m in only.split(",")] if only else None
    if mode_subset:
        for m in mode_subset:
            if m not in _ALL_MODES:
                console.print(f"[red]Unknown mode: {m}[/red] (valid: {MODES})")
                raise typer.Exit(code=2)

    async def _go():
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            tasks: dict[str, int] = {}
            mode_list = mode_subset or MODES
            for m in mode_list:
                tasks[m] = progress.add_task(f"running {m}", total=n_requests)

            def cb(mode_name: str, done: int, total: int) -> None:
                progress.update(tasks[mode_name], completed=done, total=total)

            return await run_demo_async(
                backend_url=backend,
                model=model,
                api_key=api_key,
                n_requests=n_requests,
                n_tenants=n_tenants,
                seed=seed,
                concurrency=concurrency,
                max_tokens=max_tokens,
                out_dir=out,
                modes=mode_subset,
                hourly_cost_usd=hourly_cost,
                adversarial=adversarial,
                progress=cb,
            )

    try:
        result = asyncio.run(_go())
    except KeyboardInterrupt:
        console.print("[yellow]aborted[/yellow]")
        raise typer.Exit(code=130) from None
    except Exception as e:
        console.print(f"[red]Demo failed:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print()
    console.print(render_summary(result))
    console.print()
    console.print(f"[green]Saved:[/green] {result['json']}")
    console.print(f"[green]Saved:[/green] {result['csv']}")
    if result.get("plot"):
        console.print(f"[green]Saved:[/green] {result['plot']}")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
@app.command()
def serve(
    backend: str = typer.Option(..., "--backend", "-b", help="OpenAI-compatible base URL to forward to"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    mode: str = typer.Option("balanced", "--mode", "-m", help=f"One of: {', '.join(MODES)}"),
    critical_users: str | None = typer.Option(None, "--critical-users", help="Comma-separated user ids to treat as critical"),
    concurrency: int = typer.Option(8, "--concurrency", "-c"),
    api_key: str | None = typer.Option(None, "--api-key", "-k", envvar="OPENAI_API_KEY"),
    log_level: str = typer.Option("info", "--log-level"),
) -> None:
    """Serve an OpenAI-compatible proxy with guarded scheduling."""
    if mode not in _ALL_MODES:
        console.print(f"[red]Unknown mode: {mode}[/red] (valid: {MODES})")
        raise typer.Exit(code=2)
    try:
        from guarded_hotshard.proxy import run as proxy_run
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e

    crit = [u.strip() for u in critical_users.split(",")] if critical_users else None
    console.print(
        f"[bold green]ghs serve[/bold green]  "
        f"mode=[cyan]{mode}[/cyan]  backend=[cyan]{backend}[/cyan]  "
        f"port=[cyan]{port}[/cyan]  concurrency=[cyan]{concurrency}[/cyan]"
    )
    if crit:
        console.print(f"  critical users: [magenta]{', '.join(crit)}[/magenta]")
    proxy_run(
        backend,
        host=host,
        port=port,
        mode=mode,
        critical_users=crit,
        concurrency=concurrency,
        api_key=api_key,
        log_level=log_level,
    )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
