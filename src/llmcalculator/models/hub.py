"""Search the Hugging Face Hub and size what you find.

The bundled catalog covers what people commonly run locally, which is a few
dozen models against the Hub's millions. This module closes that gap: it
searches the Hub, reads each result's real `config.json`, and hands back
`ModelSpec` objects the estimator can size like any catalogued model.

Results are cached on disk, so a repeated search is instant and works offline.
Only the standard library is used - no `huggingface_hub` dependency.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import capabilities as caps
from .catalog import _spec_from_config
from .spec import ModelSpec

API = "https://huggingface.co/api/models"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/config.json"
USER_AGENT = "llmcalculator"

CACHE_TTL = 7 * 24 * 3600  # a week; architectures do not change under a tag


# --- cache ----------------------------------------------------------------

def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    d = Path(base) / "llmcalculator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)
    return cache_dir() / (safe + ".json")


def _cache_read(key: str, ttl: float = CACHE_TTL) -> Optional[dict]:
    p = _cache_path(key)
    try:
        if time.time() - p.stat().st_mtime > ttl:
            return None
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _cache_write(key: str, payload: dict) -> None:
    try:
        _cache_path(key).write_text(json.dumps(payload))
    except OSError:
        pass  # a cache failure must never break a lookup


def clear_cache() -> int:
    """Delete every cached response. Returns how many files were removed."""
    n = 0
    for f in cache_dir().glob("*.json"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n


# --- http -----------------------------------------------------------------

def _token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tok = _token()
    if tok:
        req.add_header("Authorization", "Bearer {}".format(tok))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# --- search ---------------------------------------------------------------

@dataclass
class HubModel:
    """A search hit, before its architecture has been read."""

    id: str
    downloads: int = 0
    likes: int = 0
    gated: bool = False
    library: str = ""
    tags: Tuple[str, ...] = ()
    pipeline: str = ""

    @property
    def is_gguf(self) -> bool:
        return "gguf" in self.library.lower() or "gguf" in " ".join(self.tags).lower()

    @property
    def author(self) -> str:
        return self.id.split("/")[0] if "/" in self.id else ""


def search(query: str, limit: int = 20, sort: str = "downloads",
           task: str = "text-generation", include_gguf: bool = False,
           use_cache: bool = True) -> List[HubModel]:
    """Search the Hub for models matching `query`.

    Sorted by downloads by default, which is a decent proxy for "the model
    someone actually meant" when a name is ambiguous.
    """
    params = {
        "search": query,
        "sort": sort,
        "direction": "-1",
        # Over-fetch, because GGUF-only and config-less repos get filtered out.
        "limit": str(max(limit * 4, 40)),
    }
    if task:
        params["filter"] = task
    url = API + "?" + urllib.parse.urlencode(params)

    key = "search_" + urllib.parse.urlencode(params)
    raw = _cache_read(key, ttl=3600) if use_cache else None
    if raw is None:
        try:
            raw = {"results": json.loads(_get(url).decode("utf-8"))}
        except urllib.error.HTTPError as exc:
            raise RuntimeError("Hugging Face search failed: HTTP {}".format(exc.code)) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Hugging Face: {}. Search needs a network "
                "connection; `llmcalculator models` works offline.".format(exc.reason)) from exc
        except ValueError as exc:
            raise RuntimeError("Hugging Face returned malformed JSON") from exc
        if use_cache:
            _cache_write(key, raw)

    out = []
    for item in raw.get("results", []):
        hm = HubModel(
            id=item.get("id") or item.get("modelId", ""),
            downloads=item.get("downloads") or 0,
            likes=item.get("likes") or 0,
            gated=bool(item.get("gated")),
            library=item.get("library_name") or "",
            tags=tuple(item.get("tags") or ()),
            pipeline=item.get("pipeline_tag") or "",
        )
        if not hm.id:
            continue
        if hm.is_gguf and not include_gguf:
            continue
        out.append(hm)
    return out[:limit]


# --- resolving architectures ---------------------------------------------

def resolve(repo_id: str, use_cache: bool = True) -> Optional[ModelSpec]:
    """Read a repo's config.json and build a ModelSpec. None if unavailable."""
    key = "config_" + repo_id
    cfg = _cache_read(key) if use_cache else None
    if cfg is None:
        try:
            cfg = json.loads(_get(RESOLVE.format(repo=repo_id)).decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError):
            return None
        if use_cache:
            _cache_write(key, cfg)
    try:
        return _spec_from_config(repo_id, cfg)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def resolve_many(repo_ids: List[str], workers: int = 8,
                 use_cache: bool = True) -> Dict[str, Optional[ModelSpec]]:
    """Resolve several repos at once. Network-bound, so threads are the right tool."""
    out: Dict[str, Optional[ModelSpec]] = {}
    if not repo_ids:
        return out
    with ThreadPoolExecutor(max_workers=min(workers, len(repo_ids))) as pool:
        futures = {pool.submit(resolve, r, use_cache): r for r in repo_ids}
        for fut, repo in futures.items():
            try:
                out[repo] = fut.result()
            except Exception:
                out[repo] = None
    return out


def apply_hub_metadata(spec: ModelSpec, hub_model: "HubModel") -> ModelSpec:
    """Fold a repository's tags into a spec resolved from its config.json.

    A config file describes the architecture and nothing else. The repository
    tags are where the rest lives: which weight formats the repo actually
    publishes, what licence it carries, and what the author says the model is
    for. Both sources are needed for a complete picture.
    """
    tags = [str(t).lower() for t in hub_model.tags]
    if hub_model.library:
        tags.append(hub_model.library.lower())
    if hub_model.pipeline:
        tags.append(hub_model.pipeline.lower())

    for t in tags:
        if t.startswith("license:") and not spec.license:
            spec.license = t.split(":", 1)[1]

    # Strip the namespaced tags (license:mit, base_model:..., region:us) before
    # matching; only the bare topic tags mean anything to the trait registries.
    bare = [t for t in tags if ":" not in t]
    spec.capabilities = caps.infer_capabilities(spec, bare)
    spec.formats = caps.infer_formats(spec, bare)
    spec.runtimes = caps.infer_runtimes(spec, spec.formats)
    return spec


@dataclass
class HubResult:
    """A search hit with its architecture resolved."""

    hub: HubModel
    spec: Optional[ModelSpec] = None
    error: str = ""

    @property
    def resolved(self) -> bool:
        return self.spec is not None


def search_resolved(query: str, limit: int = 15, sort: str = "downloads",
                    include_gguf: bool = False, task: str = "text-generation",
                    use_cache: bool = True) -> List[HubResult]:
    """Search the Hub and resolve each hit's architecture.

    Hits whose config cannot be read - gated repos, adapter-only repos, and
    formats without a `config.json` - are returned with an explanation rather
    than silently dropped, so the count you see matches the count you asked for.
    """
    hits = search(query, limit=limit, sort=sort, task=task,
                  include_gguf=include_gguf, use_cache=use_cache)
    specs = resolve_many([h.id for h in hits], use_cache=use_cache)

    results = []
    for h in hits:
        spec = specs.get(h.id)
        if spec is not None:
            results.append(HubResult(hub=h, spec=apply_hub_metadata(spec, h)))
        else:
            reason = "gated - set HF_TOKEN" if h.gated else "no readable config.json"
            results.append(HubResult(hub=h, error=reason))
    return results


def trending(limit: int = 15, use_cache: bool = True) -> List[HubResult]:
    """What the Hub is currently trending, resolved and sizeable."""
    hits = search("", limit=limit, sort="trendingScore", use_cache=use_cache)
    specs = resolve_many([h.id for h in hits], use_cache=use_cache)
    out = []
    for h in hits:
        spec = specs.get(h.id)
        out.append(HubResult(hub=h,
                             spec=apply_hub_metadata(spec, h) if spec else None,
                             error="" if spec else "no readable config.json"))
    return out
