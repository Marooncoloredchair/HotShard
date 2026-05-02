"""End-to-end demo runner. The cold-DM artifact.

Generates a Zipf-skewed multi-tenant chat workload, runs every mode against
a real OpenAI-compatible backend, prints a per-tenant p99 table + a Pareto
chart, and writes a JSON results file.

The demo deliberately does NOT include a model. You point it at a backend
(vLLM, Ollama, OpenAI, Together, etc.) and it does the rest. See
`examples/quickstart.py` for the smallest example.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import httpx

from guarded_hotshard.metrics import (
    output_fidelity,
    per_tenant_p99,
    summary_row,
)
from guarded_hotshard.modes import MODES, make_mode
from guarded_hotshard.scheduler import GuardedScheduler
from guarded_hotshard.workload import make_workload


# ---------------------------------------------------------------------------
# Backend client (OpenAI-compatible)
# ---------------------------------------------------------------------------
class OpenAIBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 32,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def chat(self, client: httpx.AsyncClient, req: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": req["system"]},
                {"role": "user", "content": req["user"]},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "user": str(req["tenant"]),
        }
        t0 = time.time()
        try:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                headers=self._headers(),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            n_out = int(usage.get("completion_tokens", len(text.split())))
            return {
                "ok": True,
                "text": text,
                "n_out_tokens": n_out,
                "wall": time.time() - t0,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "wall": time.time() - t0, "text": "", "n_out_tokens": 0}


# ---------------------------------------------------------------------------
# Run one mode against the backend
# ---------------------------------------------------------------------------
async def _run_mode(
    mode_name: str,
    requests: list[dict[str, Any]],
    backend: OpenAIBackend,
    *,
    concurrency: int,
    critical_tenants: set[Hashable] | None = None,
    progress=None,
) -> dict[str, Any]:
    """Schedule offline (sort + admit + tmr), then run with backend concurrency limit."""
    mode = make_mode(mode_name)
    sched = GuardedScheduler(mode=mode, critical_tenants=critical_tenants or set())
    plan = sched.schedule_batch(requests)
    admitted = plan["admitted"]
    evicted = plan["evicted"]
    tmr_set = plan["tmr_set"]

    sem = asyncio.Semaphore(concurrency)
    completion_order: list[float] = []
    results: list[dict[str, Any]] = []

    # Trace t0 = real wall clock; we measure both queue+gen and pure gen
    trace_start = time.time()
    in_flight = 0
    high_water = 0

    async with httpx.AsyncClient() as client:

        async def run_one(idx: int, sr) -> None:
            nonlocal in_flight, high_water
            async with sem:
                in_flight += 1
                high_water = max(high_water, in_flight)
                req = sr.request
                arrival_real = trace_start + req["arrival_time"]
                # Wait until the request's "arrival time" before sending.
                # This recreates the staggered-arrival pattern that makes
                # priority scheduling actually matter.
                wait = arrival_real - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                send_t = time.time()
                primary = backend.chat(client, req)
                if sr.tmr:
                    tmr = backend.chat(client, req)
                    primary_res, tmr_res = await asyncio.gather(primary, tmr)
                else:
                    primary_res = await primary
                    tmr_res = None
                done_t = time.time()
                wall_lat = done_t - arrival_real
                gen_lat = done_t - send_t
                results.append(
                    {
                        "id": req["id"],
                        "tenant": req["tenant"],
                        "is_critical": sr.is_critical,
                        "is_storm": req.get("is_storm", False),
                        "is_pinned": sr.pinned,
                        "is_hot": sr.hot,
                        "F_risk": sr.F_risk,
                        "priority": sr.priority,
                        "tmr": sr.tmr,
                        "wall_latency": wall_lat,        # queue + gen, what users feel
                        "gen_latency": gen_lat,          # backend-only
                        "arrival_time": req["arrival_time"],
                        "n_out_tokens": primary_res["n_out_tokens"],
                        "output": primary_res.get("text", ""),
                        "ok": primary_res.get("ok", False),
                        "tmr_output": tmr_res.get("text", "") if tmr_res else None,
                    }
                )
                completion_order.append(done_t - trace_start)
                in_flight -= 1

        tasks = [asyncio.create_task(run_one(i, sr)) for i, sr in enumerate(admitted)]
        if progress is not None:
            done_count = 0
            total = len(tasks)
            for fut in asyncio.as_completed(tasks):
                await fut
                done_count += 1
                progress(mode_name, done_count, total)
        else:
            await asyncio.gather(*tasks)

    total_wall = max(completion_order) if completion_order else 0.0
    total_tokens = sum(r["n_out_tokens"] for r in results)

    return {
        "mode": mode_name,
        "n_admitted": len(admitted),
        "n_evicted": len(evicted),
        "n_tmr": len(tmr_set),
        "high_water_concurrency": high_water,
        "total_wall_seconds": total_wall,
        "n_tokens": total_tokens,
        "tokens_per_sec": total_tokens / max(total_wall, 1e-6),
        "per_request": results,
        "evicted_ids": [s.request_id for s in evicted],
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
async def run_demo_async(
    *,
    backend_url: str,
    model: str,
    api_key: str | None = None,
    n_requests: int = 60,
    n_tenants: int = 5,
    seed: int = 42,
    concurrency: int = 4,
    max_tokens: int = 32,
    out_dir: str | Path = "demo_results",
    modes: list[str] | None = None,
    hourly_cost_usd: float = 4.0,
    progress=None,
    adversarial: bool = False,
) -> dict[str, Any]:
    """The full demo. Returns the result dict that gets saved to JSON."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    backend = OpenAIBackend(backend_url, model, api_key=api_key, max_tokens=max_tokens)
    workload = make_workload(
        n_requests=n_requests,
        n_tenants=n_tenants,
        seed=seed,
        adversarial=adversarial,
    )
    mode_names = modes or MODES

    by_mode: dict[str, dict[str, Any]] = {}
    for name in mode_names:
        run = await _run_mode(
            name,
            workload,
            backend,
            concurrency=concurrency,
            critical_tenants={0},
            progress=progress,
        )
        by_mode[name] = run

    # Build fidelity vs baseline if baseline ran
    fidelity_by_mode: dict[str, float | None] = {}
    if "baseline" in by_mode:
        base_outputs: dict[Hashable, str] = {
            r["id"]: r["output"] for r in by_mode["baseline"]["per_request"]
        }
        for name, run in by_mode.items():
            if name == "baseline":
                fidelity_by_mode[name] = None
                continue
            cand = {r["id"]: r["output"] for r in run["per_request"]}
            fidelity_by_mode[name] = output_fidelity(base_outputs, cand)
    else:
        for name in by_mode:
            fidelity_by_mode[name] = None

    # Build summary table
    rows = []
    for name in mode_names:
        run = by_mode[name]
        rows.append(
            summary_row(
                mode=name,
                completed=len(run["per_request"]),
                evicted=run["n_evicted"],
                tmr=run["n_tmr"],
                per_request=run["per_request"],
                total_wall_seconds=run["total_wall_seconds"],
                total_tokens=run["n_tokens"],
                fidelity=fidelity_by_mode[name],
                hourly_cost_usd=hourly_cost_usd,
            )
        )

    # Per-tenant p99
    per_tenant: dict[str, dict[Hashable, float]] = {
        name: per_tenant_p99(run["per_request"]) for name, run in by_mode.items()
    }

    result = {
        "config": {
            "backend_url": backend_url,
            "model": model,
            "n_requests": n_requests,
            "n_tenants": n_tenants,
            "seed": seed,
            "concurrency": concurrency,
            "max_tokens": max_tokens,
            "hourly_cost_usd": hourly_cost_usd,
            "adversarial": adversarial,
            "modes": mode_names,
        },
        "summary": rows,
        "per_tenant_p99": {
            name: {str(k): float(v) for k, v in pt.items()} for name, pt in per_tenant.items()
        },
        "by_mode": {
            name: {
                "n_admitted": run["n_admitted"],
                "n_evicted": run["n_evicted"],
                "n_tmr": run["n_tmr"],
                "total_wall_seconds": run["total_wall_seconds"],
                "n_tokens": run["n_tokens"],
                "evicted_ids": [int(x) if isinstance(x, (int, float)) else str(x) for x in run["evicted_ids"]],
            }
            for name, run in by_mode.items()
        },
    }

    json_path = out_path / "demo_results.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))

    # CSV per-request
    csv_path = out_path / "demo_per_request.csv"
    csv_lines = ["mode,id,tenant,is_critical,is_pinned,is_hot,tmr,priority,wall_latency,n_out_tokens"]
    for name, run in by_mode.items():
        for r in run["per_request"]:
            csv_lines.append(
                f"{name},{r['id']},{r['tenant']},{int(r['is_critical'])},"
                f"{int(r['is_pinned'])},{int(r['is_hot'])},{int(r['tmr'])},"
                f"{r['priority']:.3f},{r['wall_latency']:.4f},{r['n_out_tokens']}"
            )
    csv_path.write_text("\n".join(csv_lines))

    # Optional Pareto plot
    try:
        _save_pareto(rows, out_path / "demo_pareto.png")
        result["plot"] = str(out_path / "demo_pareto.png")
    except Exception:  # pragma: no cover
        pass

    html_path = write_demo_html(result, out_path)
    result["report"] = str(html_path)

    result["json"] = str(json_path)
    result["csv"] = str(csv_path)
    return result


def write_demo_html(result: dict[str, Any], out_dir: Path) -> Path:
    """Write ``report.html`` next to JSON/CSV with tables + optional Pareto image."""
    import html as html_module

    out_dir = Path(out_dir)
    cfg = result["config"]
    pareto_name = "demo_pareto.png"
    pareto_ok = (out_dir / pareto_name).exists()

    rows_html = []
    for r in result["summary"]:
        fid = "-" if r["fidelity"] is None else f"{r['fidelity']:.3f}"
        rows_html.append(
            "<tr>"
            f"<td>{html_module.escape(str(r['mode']))}</td>"
            f"<td class='num'>{r['completed']}</td>"
            f"<td class='num'>{r['evicted']}</td>"
            f"<td class='num'>{r['tmr']}</td>"
            f"<td class='num'>{r['p50_lat']:.3f}</td>"
            f"<td class='num'>{r['p99_lat']:.3f}</td>"
            f"<td class='num'>{r['tok/s']:.1f}</td>"
            f"<td class='num'>{html_module.escape(str(fid))}</td>"
            f"<td class='num'>{r['fairness']:.3f}</td>"
            f"<td class='num'>{r['$/M_tok']:.3f}</td>"
            "</tr>"
        )

    pt = result.get("per_tenant_p99") or {}
    tenant_ids = sorted({t for d in pt.values() for t in d}, key=lambda x: (str(x),))
    pt_header = "".join(
        f"<th>{html_module.escape(str(m))}</th>" for m in pt.keys()
    )
    pt_rows = []
    for tid in tenant_ids:
        cells = "".join(
            f"<td class='num'>{pt[m].get(str(tid), 0):.3f}</td>" for m in pt.keys()
        )
        pt_rows.append(f"<tr><th>{html_module.escape(str(tid))}</th>{cells}</tr>")

    img_block = (
        f'<figure class="pareto"><img src="{html_module.escape(pareto_name)}" '
        f'alt="Cost vs p99 Pareto frontier"/><figcaption>Modeled cost vs p99 latency</figcaption></figure>'
        if pareto_ok
        else "<p><em>Pareto chart not generated (install matplotlib for demo extra).</em></p>"
    )

    body = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>guarded-hotshard demo report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1200px; color: #1a1a1a; }}
h1 {{ font-size: 1.4rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
th {{ background: #f4f4f4; }}
td.num, th.num {{ text-align: right; }}
figure.pareto img {{ max-width: 100%; height: auto; }}
.meta {{ color: #444; font-size: 0.9rem; margin-bottom: 1.5rem; }}
</style></head><body>
<h1>guarded-hotshard demo report</h1>
<div class="meta">
Backend: {html_module.escape(str(cfg['backend_url']))}<br/>
Model: {html_module.escape(str(cfg['model']))}<br/>
Workload: {cfg['n_requests']} requests, {cfg['n_tenants']} tenants, seed {cfg['seed']}<br/>
Concurrency: {cfg['concurrency']}, max_tokens: {cfg['max_tokens']}, hourly_cost: ${cfg['hourly_cost_usd']}/GPU-hr (proxy)
</div>

<h2>Summary by mode</h2>
<table>
<thead><tr>
<th>mode</th><th class="num">completed</th><th class="num">evicted</th><th class="num">tmr</th>
<th class="num">p50&nbsp;s</th><th class="num">p99&nbsp;s</th><th class="num">tok/s</th>
<th class="num">fidelity</th><th class="num">fairness</th><th class="num">$/M&nbsp;tok</th>
</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>

<h2>Per-tenant p99 wall latency (s)</h2>
<table>
<thead><tr><th>tenant</th>{pt_header}</tr></thead>
<tbody>{''.join(pt_rows)}</tbody>
</table>

<h2>Pareto frontier</h2>
{img_block}
<p><small>Raw data: <code>demo_results.json</code>, <code>demo_per_request.csv</code></small></p>
</body></html>"""

    path = out_dir / "report.html"
    path.write_text(body, encoding="utf-8")
    return path


def _save_pareto(rows: list[dict[str, Any]], png_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = [r["$/M_tok"] for r in rows]
    y = [r["p99_lat"] for r in rows]
    labels = [r["mode"] for r in rows]
    ax.scatter(x, y, s=80)
    for xi, yi, lab in zip(x, y, labels, strict=True):
        ax.annotate(lab, (xi, yi), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Modeled cost ($/M tokens)")
    ax.set_ylabel("p99 latency (s)")
    ax.set_title("guarded-hotshard - cost vs p99 latency Pareto frontier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def render_summary(result: dict[str, Any]) -> str:
    """Render the human-readable summary string (mirrors the notebook output)."""
    lines: list[str] = []
    cfg = result["config"]
    lines.append(
        f"Backend: {cfg['backend_url']}  |  Model: {cfg['model']}  |  "
        f"Workload: {cfg['n_requests']} requests across {cfg['n_tenants']} tenants  |  "
        f"Seed: {cfg['seed']}"
    )
    lines.append("")

    header = (
        f"{'mode':>16}  {'completed':>9}  {'evicted':>7}  {'tmr':>4}  "
        f"{'p50_lat':>8}  {'p99_lat':>8}  {'tok/s':>7}  {'fidelity':>9}  "
        f"{'fairness':>8}  {'$/M_tok':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in result["summary"]:
        fid = "-" if r["fidelity"] is None else f"{r['fidelity']:.3f}"
        lines.append(
            f"{r['mode']:>16}  {r['completed']:>9}  {r['evicted']:>7}  {r['tmr']:>4}  "
            f"{r['p50_lat']:>8.3f}  {r['p99_lat']:>8.3f}  {r['tok/s']:>7.1f}  "
            f"{fid:>9}  {r['fairness']:>8.3f}  $ {r['$/M_tok']:>8.3f}"
        )

    lines.append("")
    lines.append("Per-tenant p99 (lower = faster for that tenant)")
    if result["per_tenant_p99"]:
        modes = list(result["per_tenant_p99"].keys())
        tenants = sorted({t for d in result["per_tenant_p99"].values() for t in d})
        head = "  tenant   " + "  ".join(f"{m:>14}" for m in modes)
        lines.append(head)
        for t in tenants:
            row = f"  {t:>6}   " + "  ".join(
                f"{result['per_tenant_p99'][m].get(t, 0):>14.3f}" for m in modes
            )
            lines.append(row)

    # Headline number: T0 p99 reduction vs baseline.
    if "baseline" in result["per_tenant_p99"]:
        lines.append("")
        lines.append("T0 (premium tenant) p99 latency: HCF mode vs baseline")
        b = result["per_tenant_p99"]["baseline"].get("0", 0)
        for name, pt in result["per_tenant_p99"].items():
            if name == "baseline":
                continue
            v = pt.get("0", 0)
            if b > 0:
                pct = 100.0 * (b - v) / b
                tag = "FASTER" if pct > 0 else "slower"
                lines.append(f"  {name:>16}  T0 p99: {v:6.3f}s  ({pct:+.1f}% {tag} than baseline)")
    return "\n".join(lines)
