"""FastAPI job-based REST wrapper around browser_use.

Lets a remote service (e.g. the OpenClaw container) delegate autonomous web
tasks to a browser_use Agent that attaches to a *remote* Chrome over CDP.

Design notes
------------
- Jobs run as background asyncio tasks; the HTTP layer never blocks on agent.run().
- In-memory job store, single replica (no persistence, no shared state).
- Auth: Bearer token from env WEBTASK_API_KEY. Fail closed (503) if unset.
- Concurrency cap via asyncio.Semaphore (MAX_CONCURRENT_JOBS).
- Per-job overall timeout via asyncio.wait_for (JOB_TIMEOUT_SEC).
- Each job attaches a *fresh* BrowserSession to the same remote Chrome CDP url
  and closes it afterwards, to avoid concurrency issues from a shared session.

Verified browser_use public API (v0.12.6):
- from browser_use import Agent, BrowserSession, BrowserProfile, ChatOpenAI
- ChatOpenAI is a pydantic model with fields model / api_key / base_url
  (browser_use/llm/openai/chat.py:34,53,56) -> ChatOpenAI(model=, api_key=, base_url=).
- BrowserProfile has cdp_url, allowed_domains, prohibited_domains,
  block_ip_addresses, headless (browser_use/browser/profile.py:562,581,585,589,386).
- BrowserSession(browser_profile=BrowserProfile(...)) (browser_use/browser/session.py:235,116).
- BrowserSession.start() / .stop() / .close() (alias of stop) / .kill()
  (browser_use/browser/session.py:673,700,725,680).
- Agent(task=, llm=, browser_session=, use_vision=) (browser_use/agent/service.py:133-166);
  use_vision accepts bool | 'auto', default True.
- await agent.run(max_steps=...) -> AgentHistoryList (browser_use/agent/service.py:2483-2488).
- AgentHistoryList methods (browser_use/agent/views.py):
    .final_result() -> str | None        (717)
    .urls()         -> list[str | None]   (767)
    .number_of_steps() -> int             (883)
    .is_done() / .is_successful() / .has_errors() / .errors()  (725,732,740,707)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections import OrderedDict
from contextlib import suppress
from typing import Any, Literal

import ipaddress
import socket
import time

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

# browser_use imports are deferred into the worker so the module can be imported
# (and /health served) even in environments where the heavy deps misbehave.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("browser_use.api")

# ----------------------------------------------------------------------------
# Configuration (read once at import time; all overridable via env)
# ----------------------------------------------------------------------------

WEBTASK_API_KEY = os.environ.get("WEBTASK_API_KEY")
# Optional source-IP allowlist (comma-separated hostnames, e.g.
# "spr-clawbot.railway.internal"). When set, protected routes require BOTH a
# valid bearer token AND a peer IP that resolves to one of these hosts — so only
# the designated sibling service can drive browser-use. Railway's private network
# is per-project isolated, IPv6 ULA, no-NAT, so the peer IP is a reliable identity.
WEBTASK_ALLOWED_CLIENTS = [
    h.strip() for h in os.environ.get("WEBTASK_ALLOWED_CLIENTS", "").split(",") if h.strip()
]
_ALLOWED_TTL_SEC = 30.0
_allowed_cache: dict[str, Any] = {"ts": -1e9, "last_attempt": -1e9, "ips": frozenset()}
BROWSERUSE_MODEL = os.environ.get("BROWSERUSE_MODEL", "claude-opus-4-8")
JOB_TIMEOUT_SEC = int(os.environ.get("JOB_TIMEOUT_SEC", "600"))
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))
MAX_STORED_JOBS = 100
RESULT_TRUNCATE_CHARS = 6000
DEFAULT_MAX_STEPS = 40

# SSRF / internal-host hard blocklist. Any allowed_domain containing one of these
# substrings is rejected, and tasks that obviously target them are refused.
BLOCKED_HOST_SUBSTRINGS = (
    "railway.internal",
    "localhost",
    "127.0.0.1",
    "169.254.169.254",
    "::1",
)


def _env_truthy(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _cdp_url() -> str | None:
    """Build the remote CDP url, appending the auth token if configured."""
    url = os.environ.get("CHROME_CDP_URL")
    if not url:
        return None
    token = os.environ.get("CDP_AUTH_TOKEN")
    if token:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={token}"
    return url


# ----------------------------------------------------------------------------
# Job store
# ----------------------------------------------------------------------------

JobStatus = Literal["queued", "running", "done", "error", "cancelled", "timeout"]


class Job:
    __slots__ = (
        "id",
        "task",
        "max_steps",
        "allowed_domains",
        "use_vision",
        "status",
        "result",
        "steps",
        "urls",
        "error",
        "started_at",
        "finished_at",
        "asyncio_task",
    )

    def __init__(
        self,
        job_id: str,
        task: str,
        max_steps: int,
        allowed_domains: list[str] | None,
        use_vision: bool,
    ) -> None:
        self.id = job_id
        self.task = task
        self.max_steps = max_steps
        self.allowed_domains = allowed_domains
        self.use_vision = use_vision
        self.status: JobStatus = "queued"
        self.result: str | None = None
        self.steps: int | None = None
        self.urls: list[str] | None = None
        self.error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.asyncio_task: asyncio.Task[Any] | None = None

    def to_detail(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "result": self.result,
            "steps": self.steps,
            "urls": self.urls,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    def to_summary(self) -> dict[str, Any]:
        snippet = self.task if len(self.task) <= 120 else self.task[:117] + "..."
        return {"job_id": self.id, "status": self.status, "task": snippet}


# Ordered so we can evict the oldest beyond MAX_STORED_JOBS.
JOBS: "OrderedDict[str, Job]" = OrderedDict()
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_JOBS)


def _store_job(job: Job) -> None:
    JOBS[job.id] = job
    while len(JOBS) > MAX_STORED_JOBS:
        # Evict the oldest job that is no longer running/queued, else oldest.
        evict_id = None
        for jid, j in JOBS.items():
            if j.status not in ("queued", "running"):
                evict_id = jid
                break
        if evict_id is None:
            evict_id = next(iter(JOBS))
        JOBS.pop(evict_id, None)


# ----------------------------------------------------------------------------
# Auth (fail closed)
# ----------------------------------------------------------------------------


def _norm_ip(addr: str | None) -> str | None:
    if not addr:
        return addr
    try:
        return str(ipaddress.ip_address(addr.split("%", 1)[0]))
    except ValueError:
        return addr


def _resolve_allowed_ips(force: bool = False) -> frozenset:
    # force=True re-resolves immediately (rate-limited to 3s) so a spr-clawbot
    # redeploy to a new private IP isn't locked out for the full TTL. `ts` is
    # bumped only on a successful resolve, so DNS failures keep retrying instead
    # of freezing a stale set as fresh.
    if not WEBTASK_ALLOWED_CLIENTS:
        return frozenset()
    now = time.monotonic()
    if not force and (now - _allowed_cache["ts"] < _ALLOWED_TTL_SEC):
        return _allowed_cache["ips"]
    if force and (now - _allowed_cache["last_attempt"] < 3.0):
        return _allowed_cache["ips"]
    _allowed_cache["last_attempt"] = now
    ips = set()
    for host in WEBTASK_ALLOWED_CLIENTS:
        try:
            for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP):
                ips.add(_norm_ip(info[4][0]))
        except socket.gaierror as exc:
            logger.warning("WEBTASK_ALLOWED_CLIENTS: could not resolve %s: %s", host, exc)
    if ips:
        _allowed_cache["ips"] = frozenset(ips)
        _allowed_cache["ts"] = now
    return _allowed_cache["ips"]


async def require_auth(
    request: Request, authorization: str | None = Header(default=None)
) -> None:
    if not WEBTASK_API_KEY:
        logger.error("WEBTASK_API_KEY is not set; refusing protected request (fail closed).")
        raise HTTPException(status_code=503, detail="Server not configured: WEBTASK_API_KEY unset")
    expected = f"Bearer {WEBTASK_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
    # Defense-in-depth: when an allowlist is configured, the peer IP must also
    # resolve to a designated sibling (only spr-clawbot may drive the browsers).
    # NOTE: request.client.host is the raw socket peer — this service is private
    # (no public domain) and uvicorn runs without --proxy-headers, so it is NOT
    # derived from a spoofable X-Forwarded-For.
    if WEBTASK_ALLOWED_CLIENTS:
        peer = _norm_ip(request.client.host if request.client else None)
        allowed = peer and (peer in _resolve_allowed_ips() or peer in _resolve_allowed_ips(force=True))
        if not allowed:
            logger.warning("Rejected request from non-allowlisted peer: %s", peer)
            raise HTTPException(status_code=403, detail="Client not allowed")


# ----------------------------------------------------------------------------
# SSRF guard
# ----------------------------------------------------------------------------


def _hits_blocklist(value: str) -> str | None:
    low = value.lower()
    for needle in BLOCKED_HOST_SUBSTRINGS:
        if needle in low:
            return needle
    return None


def validate_request_against_blocklist(task: str, allowed_domains: list[str] | None) -> None:
    """Reject obviously-internal targets. Raises HTTPException(400) on hit."""
    if allowed_domains:
        for dom in allowed_domains:
            hit = _hits_blocklist(dom)
            if hit:
                raise HTTPException(
                    status_code=400,
                    detail=f"allowed_domains contains blocked internal host substring: {hit}",
                )
    task_hit = _hits_blocklist(task)
    if task_hit:
        raise HTTPException(
            status_code=400,
            detail=f"task targets a blocked internal host substring: {task_hit}",
        )


# ----------------------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------------------


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1)
    max_steps: int = Field(default=DEFAULT_MAX_STEPS, ge=1, le=500)
    allowed_domains: list[str] | None = None
    use_vision: bool | None = None


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------


async def _run_agent(job: Job) -> None:
    """Construct + run the agent for a single job. Updates job state in place."""
    # Imported lazily so module import / health checks don't require heavy deps.
    from browser_use import Agent, BrowserProfile, BrowserSession, ChatOpenAI

    cdp_url = _cdp_url()
    if not cdp_url:
        job.status = "error"
        job.error = "CHROME_CDP_URL is not set; cannot attach to remote browser"
        job.finished_at = time.time()
        return

    try:
        api_key = os.environ["ANT_PROXY_API_KEY"]
        base_url = os.environ["ANT_PROXY_URL"].rstrip("/") + "/v1"
    except KeyError as exc:
        job.status = "error"
        job.error = f"Missing required env var: {exc.args[0]}"
        job.finished_at = time.time()
        return

    # temperature=None is REQUIRED for the Anthropic models behind ANT-Proxy
    # (e.g. claude-opus-4-8): they reject the `temperature` param entirely with
    # HTTP 400 "`temperature` is deprecated for this model". ChatOpenAI defaults
    # temperature to 0.2 and only omits it from the request when it is None
    # (see browser_use/llm/openai/chat.py), so we must pass None explicitly.
    llm = ChatOpenAI(
        model=BROWSERUSE_MODEL,
        api_key=api_key,
        base_url=base_url,
        temperature=None,
        # ANT-Proxy fronts Anthropic (claude-opus-4-8) via an OpenAI-compatible
        # shim. OpenAI's strict `response_format: json_schema` is NOT reliably
        # honored on that path, so the agent intermittently "fails to produce
        # correct output format" and aborts mid-task. Harden structured output:
        #  - add_schema_to_system_prompt: also put the JSON schema in the prompt,
        #    so the model formats correctly even if the proxy drops response_format.
        #  - remove_min_items / remove_defaults: strip schema features Anthropic's
        #    tool-schema validation chokes on (documented provider-compat knobs).
        add_schema_to_system_prompt=True,
        remove_min_items_from_schema=True,
        remove_defaults_from_schema=True,
    )

    profile_kwargs: dict[str, Any] = {
        "cdp_url": cdp_url,
        "headless": True,
        # Block raw IP navigation (covers 127.0.0.1, 169.254.x, ::1, etc.) as a
        # defence-in-depth SSRF measure on top of the request-level blocklist.
        "block_ip_addresses": True,
        # Booking widgets (BokaBord, Caspeco, TheFork, ...) are usually embedded
        # in a CROSS-ORIGIN iframe. Keep OOPIF traversal on explicitly so a future
        # browser-use default flip can't silently break booking-form filling.
        # NOTE: this alone is insufficient when the remote Chrome runs with Site
        # Isolation — that iframe stays out-of-process and its DOM is unreachable.
        # The decisive fix is CHROME_EXTRA_ARGS=--disable-features=IsolateOrigins,
        # site-per-process on the chrome-browseruse container (see chrome-novnc-cdp
        # 5-chromium.conf). These two work together.
        "cross_origin_iframes": True,
        # Raise the per-page iframe cap so ad/tracking iframes can't crowd out the
        # real booking widget before it is serialized.
        "max_iframes": 200,
        # paint_order_filtering is experimental and can suppress clickable indices
        # on cards that are visible but judged occluded — turn it off so booking
        # widget cards stay clickable.
        "paint_order_filtering": False,
    }
    if job.allowed_domains:
        profile_kwargs["allowed_domains"] = job.allowed_domains
    profile = BrowserProfile(**profile_kwargs)

    session = BrowserSession(browser_profile=profile)
    try:
        await session.start()
        agent = Agent(
            task=job.task,
            llm=llm,
            browser_session=session,
            use_vision=job.use_vision,
        )
        history = await agent.run(max_steps=job.max_steps)

        final = history.final_result()
        if final and len(final) > RESULT_TRUNCATE_CHARS:
            final = final[:RESULT_TRUNCATE_CHARS] + "...[truncated]"
        job.result = final
        job.steps = history.number_of_steps()
        # urls() returns list[str | None]; drop Nones and de-dup preserving order.
        seen: set[str] = set()
        urls: list[str] = []
        for u in history.urls():
            if u and u not in seen:
                seen.add(u)
                urls.append(u)
        job.urls = urls
        job.status = "done"
        if job.result is None and history.has_errors():
            job.error = "; ".join(e for e in history.errors() if e) or "agent reported errors"
    except Exception as exc:  # noqa: BLE001 - record any agent failure as error
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        logger.exception("Agent run failed for job %s", job.id)
    finally:
        with suppress(Exception):
            await session.kill()


async def _job_runner(job: Job) -> None:
    """Wrap a job with the concurrency semaphore + overall timeout."""
    async with SEMAPHORE:
        job.status = "running"
        job.started_at = time.time()
        try:
            await asyncio.wait_for(_run_agent(job), timeout=JOB_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            job.status = "timeout"
            job.error = f"Job exceeded JOB_TIMEOUT_SEC={JOB_TIMEOUT_SEC}s"
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "Job cancelled"
            raise
        finally:
            if job.finished_at is None:
                job.finished_at = time.time()


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

app = FastAPI(title="browser-use webtask API", version="0.12.6")


@app.get("/health")
async def health() -> dict[str, Any]:
    """Open (unauthenticated) health check. Reports CDP reachability."""
    cdp_reachable = False
    cdp_url = _cdp_url()
    if cdp_url:
        # Probe the CDP HTTP version endpoint (token, if any, is already on the url).
        base = cdp_url.split("?", 1)[0].rstrip("/")
        query = ("?" + cdp_url.split("?", 1)[1]) if "?" in cdp_url else ""
        probe = f"{base}/json/version{query}"
        with suppress(Exception):
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(probe)
                cdp_reachable = resp.status_code == 200
    return {"ok": True, "model": BROWSERUSE_MODEL, "cdp": cdp_reachable}


@app.post("/run", dependencies=[Depends(require_auth)])
async def run(req: RunRequest) -> dict[str, str]:
    validate_request_against_blocklist(req.task, req.allowed_domains)

    use_vision = req.use_vision if req.use_vision is not None else _env_truthy(
        os.environ.get("USE_VISION"), True
    )

    job_id = uuid.uuid4().hex
    job = Job(
        job_id=job_id,
        task=req.task,
        max_steps=req.max_steps,
        allowed_domains=req.allowed_domains,
        use_vision=use_vision,
    )
    _store_job(job)
    job.asyncio_task = asyncio.create_task(_job_runner(job))
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs", dependencies=[Depends(require_auth)])
async def list_jobs() -> dict[str, Any]:
    # Most-recent first, capped to last 50.
    jobs = list(JOBS.values())[-50:][::-1]
    return {"jobs": [j.to_summary() for j in jobs]}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_auth)])
async def get_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_detail()


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_auth)])
async def cancel_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status in ("done", "error", "cancelled", "timeout"):
        return {"job_id": job_id, "status": job.status}
    if job.asyncio_task is not None and not job.asyncio_task.done():
        job.asyncio_task.cancel()
    job.status = "cancelled"
    if job.finished_at is None:
        job.finished_at = time.time()
    return {"job_id": job_id, "status": "cancelled"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="::", port=int(os.environ.get("PORT", "8000")))
