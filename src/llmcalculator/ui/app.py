"""A local web app, for people who do not want a terminal.

Runs entirely on the standard library: no Flask, no Node, no build step. The
page is a single self-contained HTML file served from disk; everything it
needs comes from the JSON API below, so nothing is fetched from the internet
and the whole thing works offline.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .. import compare, workloads
from ..estimate import (
    Verdict, estimate as estimate_one, max_model_size, recommended_quant, sweep,
)
from ..hardware import detect
from ..hardware.base import HardwareProfile
from ..models import capabilities as caps
from ..models import catalog, hub

_PAGE = Path(__file__).with_name("app.html")
_hardware: Optional[HardwareProfile] = None


def hardware() -> HardwareProfile:
    """Detect once and reuse; probing vendor tools is slow."""
    global _hardware
    if _hardware is None:
        _hardware = detect()
    return _hardware


# --- API ------------------------------------------------------------------

def api_hardware() -> dict:
    hw = hardware()
    # What the machine can do, per workload - not to be confused with what a
    # model can do, which is `api_capabilities` below.
    can_do = []
    for wl in workloads.ALL:
        ctx = 8192 if wl.key == "inference" else 2048
        best = compare.largest_that_fits(hw, wl, context=ctx)
        can_do.append({
            "key": wl.key,
            "label": wl.label,
            "description": wl.description,
            "max_params_b": round(max_model_size(hw, wl, context=ctx), 1),
            "precision": wl.default_base_quant,
            "largest": best.model.name if best else None,
            "largest_gb": round(best.total_gb, 1) if best else None,
        })
    return {
        "cpu": {"name": hw.cpu.name, "cores": hw.cpu.cores, "arch": hw.cpu.arch},
        "ram_gb": round(hw.ram_gb, 1),
        "disk_free_gb": round(hw.disk_free_gb, 1),
        "platform": hw.platform,
        "has_gpu": hw.has_gpu,
        "gpus": [{"name": a.name, "vendor": a.vendor, "memory_gb": round(a.memory_gb, 1),
                  "bandwidth_gbs": a.bandwidth_gbs, "unified": a.unified_memory}
                 for a in hw.accelerators if a.vendor != "cpu"],
        "budget_gb": round(hw.budget_bytes("auto") / (1024 ** 3), 1),
        "capabilities": can_do,
        "notes": hw.notes,
    }


def api_models(query: str = "", workload: str = "inference", context: int = 8192,
               capability: str = "") -> dict:
    hw = hardware()
    wl = workloads.get(workload)
    ctx = context if wl.key == "inference" else min(context, 2048)
    models = catalog.search(query) if query else catalog.all_models()
    if capability:
        models = [m for m in models if m.has_capability(capability)]
    out = []
    for m in models:
        est = recommended_quant(m, hw, wl, context=ctx)
        out.append({
            "name": m.name,
            "params_b": round(m.params_b, 1),
            "active_b": round(m.active_params / 1e9, 1) if m.is_moe else None,
            "family": m.family,
            "tags": list(m.tags),
            "capabilities": [t.label for t in m.support().capabilities],
            "hf_id": m.hf_id,
            "max_context": m.max_context,
            "verdict": est.verdict,
            "label": est.label(),
            "quant": est.quant_name,
            "gb": round(est.total_gb, 1),
            "utilization": round(est.utilization, 3),
            "tokens_per_sec": round(est.tokens_per_sec, 0),
        })
    return {"models": out, "count": len(out),
            "capability_counts": catalog.capability_counts()}


def api_detail(name: str, workload: str = "inference", context: int = 8192,
               quant: Optional[str] = None) -> dict:
    hw = hardware()
    wl = workloads.get(workload)
    model = catalog.get(name)
    ctx = min(context, model.max_context)
    if quant:
        est = estimate_one(model, hw, wl, quant, context=ctx)
    else:
        est = recommended_quant(model, hw, wl, context=ctx)

    data = est.as_dict()
    data["describe"] = model.describe()
    data["hf_id"] = model.hf_id
    data["label"] = est.label()
    # Everything true about the model regardless of this machine: architecture,
    # what it is for, how it ships, and what will load it.
    data["spec"] = model.as_dict()
    data["architecture_items"] = [{"label": k, "value": v}
                                  for k, v in model.architecture_items()]
    data["breakdown"] = [{"label": k, "gb": round(v, 2)} for k, v in est.breakdown.items_gb()]
    data["workloads"] = [
        {"key": e.workload.key, "label": e.workload.label, "gb": round(e.total_gb, 1),
         "verdict": e.verdict, "label_text": e.label(), "quant": e.quant_name}
        for e in compare.workload_table(model, hw, context=min(ctx, 4096))
    ]
    data["quants"] = [
        {"name": e.quant_name, "gb": round(e.total_gb, 1), "verdict": e.verdict,
         "tokens_per_sec": round(e.tokens_per_sec, 0), "quality": round(e.quality * 100),
         "utilization": round(e.utilization, 3)}
        for e in compare.quant_table(model, hw, wl, context=ctx)
    ]
    data["contexts"] = [
        {"context": e.context, "gb": round(e.total_gb, 1), "verdict": e.verdict,
         "kv_gb": round(e.breakdown.kv_cache / (1024 ** 3), 2)}
        for e in sweep(model, hw, quant_name=est.quant_name)
    ]
    return data


def api_hub_search(query: str = "", workload: str = "inference",
                   context: int = 8192, limit: int = 20,
                   include_gguf: bool = False) -> dict:
    """Search the whole Hugging Face Hub and size every hit."""
    if not query.strip():
        return {"results": [], "count": 0, "query": query}
    hw = hardware()
    wl = workloads.get(workload)
    ctx = context if wl.key == "inference" else min(context, 2048)
    try:
        hits = hub.search_resolved(query, limit=limit, include_gguf=include_gguf)
    except RuntimeError as exc:
        return {"error": str(exc), "results": [], "count": 0}

    out = []
    for r in hits:
        row = {"id": r.hub.id, "downloads": r.hub.downloads, "likes": r.hub.likes,
               "gated": r.hub.gated, "resolved": r.resolved, "error": r.error}
        if r.resolved:
            est = recommended_quant(r.spec, hw, wl, context=ctx)
            row.update({
                "params_b": round(r.spec.params_b, 1),
                "active_b": round(r.spec.active_params / 1e9, 1) if r.spec.is_moe else None,
                "quant": est.quant_name, "gb": round(est.total_gb, 1),
                "verdict": est.verdict, "label": est.label(),
                "utilization": round(est.utilization, 3),
                "tokens_per_sec": round(est.tokens_per_sec, 0),
                "max_context": r.spec.max_context,
                "capabilities": [t.label for t in r.spec.support().capabilities],
                "formats": [t.label for t in r.spec.support().formats],
            })
        out.append(row)
    return {"results": out, "count": len(out), "query": query}


def api_recommend(workload: str = "inference", context: int = 8192, limit: int = 8) -> dict:
    hw = hardware()
    recs = compare.recommend(hw, workloads.get(workload), context=context, limit=limit)
    return {"recommendations": [
        {"name": r.estimate.model.name, "params_b": round(r.estimate.model.params_b, 1),
         "quant": r.estimate.quant_name, "gb": round(r.estimate.total_gb, 1),
         "tokens_per_sec": round(r.estimate.tokens_per_sec, 0),
         "verdict": r.estimate.verdict, "reason": r.reason, "score": round(r.score, 1)}
        for r in recs]}


def api_compare(names: list, workload: str = "inference", context: int = 8192) -> dict:
    hw = hardware()
    ests = compare.compare(names, hw, workloads.get(workload), context=context)
    return {"results": [
        {"name": e.model.name, "params_b": round(e.model.params_b, 1), "quant": e.quant_name,
         "gb": round(e.total_gb, 1), "verdict": e.verdict, "label": e.label(),
         "utilization": round(e.utilization, 3),
         "tokens_per_sec": round(e.tokens_per_sec, 0)}
        for e in ests]}


def api_capabilities() -> dict:
    """The trait vocabulary, so the page can explain a chip rather than just
    show it."""
    return {
        "capabilities": [t.as_dict() for t in caps.known("capability")],
        "formats": [t.as_dict() for t in caps.known("format")],
        "runtimes": [t.as_dict() for t in caps.known("runtime")],
        "counts": catalog.capability_counts(),
    }


ROUTES = {
    "/api/hardware": lambda q: api_hardware(),
    "/api/capabilities": lambda q: api_capabilities(),
    "/api/models": lambda q: api_models(_one(q, "q", ""), _one(q, "workload", "inference"),
                                        int(_one(q, "context", "8192")),
                                        _one(q, "capability", "")),
    "/api/detail": lambda q: api_detail(_one(q, "name"), _one(q, "workload", "inference"),
                                        int(_one(q, "context", "8192")),
                                        _one(q, "quant", "") or None),
    "/api/hub": lambda q: api_hub_search(_one(q, "q", ""), _one(q, "workload", "inference"),
                                         int(_one(q, "context", "8192")),
                                         int(_one(q, "limit", "20")),
                                         _one(q, "gguf", "") == "1"),
    "/api/recommend": lambda q: api_recommend(_one(q, "workload", "inference"),
                                              int(_one(q, "context", "8192")),
                                              int(_one(q, "limit", "8"))),
    "/api/compare": lambda q: api_compare([n for n in _one(q, "names", "").split(",") if n],
                                          _one(q, "workload", "inference"),
                                          int(_one(q, "context", "8192"))),
}


def _one(query: dict, key: str, default: str = "") -> str:
    vals = query.get(key)
    return vals[0] if vals else default


# --- server ---------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "llmcalculator"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._send_page()
            return
        if path in ROUTES:
            try:
                payload = ROUTES[path](query)
            except (KeyError, ValueError, RuntimeError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": "internal: {}".format(exc)}, status=500)
                return
            self._send_json(payload)
            return
        self._send_json({"error": "not found"}, status=404)

    def _send_page(self) -> None:
        try:
            body = _PAGE.read_bytes()
        except OSError:
            self._send_json({"error": "app.html missing from the installation"}, status=500)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args) -> None:
        """Stay quiet; the terminal belongs to the user, not the access log."""
        return


def _free_port(host: str, preferred: int) -> int:
    """Use the preferred port, or the next free one if it is taken."""
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port in range {}-{}".format(preferred, preferred + 20))


def serve(host: str = "127.0.0.1", port: int = 8770, open_browser: bool = True) -> int:
    """Start the local app and, unless told otherwise, open a browser at it."""
    port = _free_port(host, port)
    url = "http://{}:{}/".format(host, port)

    hardware()  # warm the cache before the first request arrives

    server = ThreadingHTTPServer((host, port), Handler)
    print("llmcalculator is running at {}".format(url))
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0
