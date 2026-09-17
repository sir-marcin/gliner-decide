#!/usr/bin/env python
"""FastAPI server that keeps GLiNER2 models loaded in memory and ranks
decision choices over HTTP.

Usage:
    python server.py                      # bind $GLINER_HOST:$GLINER_PORT
    uvicorn server:app --port 8765

Requests are not served in parallel: one daemon thread does every model load
and every forward pass, because MPS is not safe for concurrent inference.
Handlers put a job on a bounded FIFO queue and await its future, so a slow
request never pins a threadpool thread and /health stays instant. Jobs waiting
for the same, already-loaded model are drained into one batched forward pass.
POST /api/decide/batch asks several questions about one passage. It is ONE job
and one forward pass: gliner2 takes several classification tasks in one schema,
so the passage is encoded once however many decisions ride on it. That sharing
is not free of consequences -- see decide_batch() in common.py and the handler
below.

Run ONE process. `uvicorn --workers N` would load N copies of the model and
have them fight over a single GPU.

Environment:
    GLINER_HOST            (default 0.0.0.0)
    GLINER_PORT            (default 8765)
    GLINER_MODEL           (default base)   alias or full HF repo id, preloaded at startup
    GLINER_DEVICE          (default auto)   auto|mps|cuda|cpu
    GLINER_CORS_ORIGINS    (default unset)  comma-separated list of allowed browser
                                            origins, e.g. "http://dev-box:3000". Unset
                                            or empty means no CORS middleware at all,
                                            so only same-origin pages (the bundled UI)
                                            can call the API.
    GLINER_MAX_QUEUE       (default 64)     jobs allowed to wait; overflow is a 503
                                            with Retry-After.
    GLINER_QUEUE_TIMEOUT   (default 60)     seconds a request waits for the worker
                                            to *pick it up* before giving up with a
                                            503. The clock stops once the job is
                                            running, so a long forward pass can no
                                            longer time out its own request; a
                                            separate 600s ceiling (a wedged worker)
                                            answers 504.
    GLINER_MAX_TOKENS      (default 8192)   subword tokens a passage may contain,
                                            counted with the model's tokenizer.
                                            The real cost bound: word and
                                            character caps do not bound scripts
                                            that tokenize to ~1 subword per char.
    GLINER_MAX_BATCH       (default 8)      max jobs coalesced into one forward pass.
                                            1 disables batching.
    GLINER_MAX_DECISIONS   (default 16)     max decisions in one POST /api/decide/batch
                                            body, and GLINER_MAX_TOTAL_CHOICES (120)
                                            choices across them. Both bound the cost
                                            of the single forward pass they share.
    GLINER_ALLOW_ANY_MODEL (default unset)  1 accepts arbitrary Hugging Face repo ids
                                            in `model`. Off by default because the
                                            single worker is shared: an unknown repo
                                            makes the server download ~1 GB, and with
                                            the hub unreachable it pins the worker for
                                            minutes of HF retries.
"""

import asyncio
import functools
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

print = functools.partial(print, flush=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from pydantic import BaseModel, Field, field_validator, model_validator

from common import (
    DEFAULT_BIND_HOST,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_TASK_LABEL,
    MAX_CHOICES,
    MAX_TEXT_CHARS,
    MAX_DECISIONS as DEFAULT_MAX_DECISIONS,
    MAX_TOKENS,
    MAX_TOTAL_CHOICES as DEFAULT_MAX_TOTAL_CHOICES,
    MIN_CHOICES,
    MODEL_ALIASES,
    InputTooLong,
    cached_max_words,
    check_input,
    clean_choices,
    decide_batch,
    load_model,
    max_words_for,
    resolve_device,
    resolve_model_name,
)

log = logging.getLogger("minos.server")

HOST = os.environ.get("GLINER_HOST", DEFAULT_BIND_HOST)
PORT = int(os.environ.get("GLINER_PORT", str(DEFAULT_PORT)))
MODEL = os.environ.get("GLINER_MODEL", DEFAULT_MODEL)
DEVICE = resolve_device(os.environ.get("GLINER_DEVICE", "auto"))
CORS_ORIGINS = [o.strip() for o in os.environ.get("GLINER_CORS_ORIGINS", "").split(",") if o.strip()]
MAX_QUEUE = int(os.environ.get("GLINER_MAX_QUEUE", "64"))
QUEUE_TIMEOUT = float(os.environ.get("GLINER_QUEUE_TIMEOUT", "60"))
MAX_BATCH = max(1, int(os.environ.get("GLINER_MAX_BATCH", "8")))
MAX_DECISIONS = max(1, int(os.environ.get("GLINER_MAX_DECISIONS", str(DEFAULT_MAX_DECISIONS))))
MAX_TOTAL_CHOICES = max(MIN_CHOICES, int(os.environ.get("GLINER_MAX_TOTAL_CHOICES",
                                                        str(DEFAULT_MAX_TOTAL_CHOICES))))
ALLOW_ANY_MODEL = os.environ.get("GLINER_ALLOW_ANY_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}

# Backstop for a worker that is wedged (a hung load, a driver fault) rather than
# merely slow: QUEUE_TIMEOUT now stops counting the moment the job starts
# running, so without this a dead worker would hang every client forever.
# Generous on purpose -- it must never fire on a legitimately slow forward pass.
INFERENCE_CEILING = 600.0

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# Padded word-slots (longest text in the batch x batch size) a single forward
# pass may occupy. The collator pads every text to the longest one and
# attention is quadratic in length, so mixing one long passage into a batch of
# short ones is ruinous: measured on MPS, a 1325-word passage costs 0.43s alone
# but 6.0s batched with 7 x 53-word passages. This cap keeps normal short
# requests batching freely while a long one runs alone.
BATCH_WORD_BUDGET = 2048

# Only the worker thread touches this: MPS is not safe for concurrent forward
# passes, and two requests for the same cold model must not both load it.
_models: dict[str, object] = {}


def get_model(repo: str):
    """Return (model, load_seconds). Worker thread only."""
    if repo in _models:
        return _models[repo], 0.0
    print(f"Loading {repo} on {DEVICE} ...")
    t0 = time.perf_counter()
    _models[repo] = load_model(repo, DEVICE)
    load_s = time.perf_counter() - t0
    print(f"Loaded {repo} in {load_s:.2f}s")
    return _models[repo], load_s


class QueueFull(Exception):
    """The pending queue is at MAX_QUEUE."""


class ModelLoadError(Exception):
    """The worker could not load the job's model (client-visible 400)."""


def _input_detail(exc: InputTooLong) -> list[dict]:
    """common.InputTooLong -> the list shape pydantic uses for 422s, so a client
    renders a limit rejection exactly like any other validation error."""
    return [{"loc": ["body", "text"], "msg": exc.message, "type": "value_error"}]


# eq=False so `==` is identity: _pending.remove(job) must drop *this* job, not
# the first one that happens to carry the same text and choices.
@dataclass(eq=False)
class Job:
    repo: str
    text: str
    # (task label, choices) per decision. A single-decision request has exactly
    # one, labelled DEFAULT_TASK_LABEL; /api/decide/batch has up to
    # MAX_DECISIONS, and they all share this job's one forward pass.
    tasks: list[tuple[str, list[str]]]
    future: Future = field(default_factory=Future)
    # Resolved by the worker the moment it commits to running this job, before
    # the model load and the forward pass. GLINER_QUEUE_TIMEOUT is waited on
    # *this*, not on `future`: the old code timed a request out while its own
    # inference was running, throwing away GPU work already spent on it.
    started: Future = field(default_factory=Future)
    enqueued_at: float = field(default_factory=time.perf_counter)
    # check_input()'s result, set by the handler when the model was already
    # loaded (so the limits were checked there); None means the worker still
    # has to do it.
    info: dict | None = None

    @property
    def schema_words(self) -> int:
        """Roughly how many words this job's schema adds to the encoder
        sequence, on top of the passage.

        The schema is not free padding: task labels and every choice are
        tokenized into the same sequence as the text, which is why a
        multi-decision job is wider than its word count suggests. Counted the
        cheap way (whitespace) because this only has to be good enough to keep
        BATCH_WORD_BUDGET honest."""
        return sum(len(label.split()) + sum(len(c.split()) for c in choices) + 1
                   for label, choices in self.tasks)

    @property
    def width(self) -> int:
        """Padded word-slots this job claims in a batch: passage plus schema.
        Only meaningful once `info` is set."""
        return self.info["words"] + self.schema_words


# Invariant: _pending and _busy are written ONLY while holding _cond (/health
# reads them unlocked and may be one job stale, deliberately -- it must never
# block). _pending holds at most MAX_QUEUE jobs plus the shutdown
# sentinel, which is appended last so queued work drains first. Exactly one
# consumer (the worker) waits on _cond, so a single notify() per append is
# enough. The condition replaces queue.Queue because the worker must inspect
# the head of the queue before committing to a batch, which Queue cannot do.
_STOP = None
_cond = threading.Condition()
_pending: deque = deque()
_busy = False

_stats_lock = threading.Lock()
_stats = {"served": 0, "rejected_busy": 0, "timed_out": 0, "batches": 0}
_worker_thread: threading.Thread | None = None


def _bump(key: str) -> None:
    with _stats_lock:
        _stats[key] += 1


def _enqueue(job: Job) -> None:
    with _cond:
        if len(_pending) >= MAX_QUEUE:
            raise QueueFull
        _pending.append(job)
        _cond.notify()


def _take_batch() -> list[Job] | None:
    """Block for work, then return the longest FIFO prefix that can share one
    forward pass: same repo, model already loaded, at most MAX_BATCH jobs.
    Returns None on the shutdown sentinel.

    Jobs whose client already gave up (future cancelled) are dropped here
    without costing a batch slot or a slice of the word budget -- a timeout
    storm used to fill batches with corpses and stall the live requests behind
    them."""
    global _busy
    with _cond:
        while True:
            while not _pending:
                _cond.wait()
            head = _pending.popleft()
            if head is _STOP:
                return None
            if not head.future.cancelled():
                break
        batch = [head]
        # A cold model is never batched: it is loaded inside this batch, so the
        # jobs behind it cannot yet be known to share it. info is None for a
        # job enqueued before its model finished loading -- unmeasured, so it
        # does not get to join a batch either.
        if head.repo in _models and head.info is not None:
            longest = head.width
            while len(batch) < MAX_BATCH and _pending:
                nxt = _pending[0]
                if nxt is _STOP:
                    break
                if nxt.future.cancelled():
                    _pending.popleft()
                    continue
                if nxt.repo != head.repo or nxt.info is None:
                    break
                widest = max(longest, nxt.width)
                if widest * (len(batch) + 1) > BATCH_WORD_BUDGET:
                    break
                longest = widest
                batch.append(_pending.popleft())
        _busy = True
        return batch


def _finish_batch() -> None:
    global _busy
    with _cond:
        _busy = False


def _run_batch(jobs: list[Job], started_at: float) -> None:
    repo = jobs[0].repo
    try:
        model, load_s = get_model(repo)
    except (RepositoryNotFoundError, HFValidationError, ValueError, OSError) as e:
        # HFValidationError is a ValueError, not an OSError: "a/b/c" and
        # "not a repo" used to escape as a 500 with a traceback. Log the real
        # thing server-side, hand the client a readable 400.
        log.exception("Could not load model %r", repo)
        err = ModelLoadError(f"Could not load model '{repo}': {type(e).__name__}. "
                             "Check the alias or Hugging Face repo id.")
        for job in jobs:
            job.future.set_exception(err)
        return

    # Checked here rather than in a validator: the limits depend on which model
    # the request resolved to (its word budget, its tokenizer), which the
    # handler only knows for an already-loaded model.
    runnable = []
    for job in jobs:
        if job.info is None:
            try:
                job.info = check_input(model, job.text, repo)
            except InputTooLong as e:
                job.future.set_exception(e)
                continue
        runnable.append(job)
    if not runnable:
        return

    limit = runnable[0].info["max_words"]
    t0 = time.perf_counter()
    results = decide_batch(model, [(j.text, j.tasks) for j in runnable], max_len=limit)
    inference_s = time.perf_counter() - t0
    _bump("batches")

    for job, scored in zip(runnable, results):
        job.future.set_result({
            # {task label: ranked}. One entry for a single-decision request,
            # one per decision for a batch.
            "scored": scored,
            "load_s": load_s,
            "inference_s": inference_s,
            "queued_s": started_at - job.enqueued_at,
            "batch_size": len(runnable),
            **job.info,
        })


def _worker() -> None:
    while True:
        jobs = _take_batch()
        if jobs is None:
            return
        try:
            # Skips jobs whose client already timed out and cancelled: no point
            # burning GPU time on a response nobody will read. Also flips the
            # future to RUNNING, so a later cancel() cannot race set_result().
            jobs = [j for j in jobs if j.future.set_running_or_notify_cancel()]
            for j in jobs:
                # Releases the handler's queue-timeout wait: from here on the
                # request is running, not waiting. A handler that timed out in
                # the same instant has already cancelled this future -- its
                # cancel() of `future` then loses the race above, so it falls
                # through and awaits the result like everyone else.
                if j.started.set_running_or_notify_cancel():
                    j.started.set_result(None)
            if jobs:
                _run_batch(jobs, time.perf_counter())
        except Exception as e:
            log.exception("Inference worker failed on a batch of %d", len(jobs))
            for job in jobs:
                if not job.future.done():
                    job.future.set_exception(e)
        finally:
            _finish_batch()


# Set when the startup preload fails, so /health can report "degraded" instead
# of the process dying (a non-zero exit turns launchd KeepAlive into a restart
# loop, and a server that serves every *other* model is still useful).
_preload_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _preload_error, _worker_thread
    repo = resolve_model_name(MODEL)
    try:
        get_model(repo)
    except Exception as e:
        _preload_error = (f"Could not preload default model '{repo}': {type(e).__name__}: "
                          f"{str(e).splitlines()[0] if str(e) else ''}")
        log.exception("Startup preload failed; serving in degraded mode")
        print(f"WARNING: {_preload_error}")
        print("Serving anyway (degraded): requests naming another model still work.")
    _worker_thread = threading.Thread(target=_worker, name="gliner-inference", daemon=True)
    _worker_thread.start()
    yield
    with _cond:
        _pending.append(_STOP)
        _cond.notify()
    _worker_thread.join(timeout=5)
    _models.clear()


DESCRIPTION = """
Minos: a zero-shot **decision helper** built on GLiNER2.5.

A *decision* is a text passage describing a situation plus a list of candidate
choices. The model scores how well each choice fits the situation and returns
them ranked best-first, along with the single `best` choice.

**Confidences are independent per-choice scores, not a probability
distribution.** Each choice is scored on its own against the passage, so the
values do not sum to 1: every choice can score high, or all of them low. Treat
them as "how well does this option fit?", not "what share of the decision does
this option hold?".

Models are cached in memory. The model named by `GLINER_MODEL` is preloaded at
startup; any other alias or Hugging Face repo id is loaded on first use (that
request pays the load cost, reported as `timing.load_s`) and stays cached.

**Input length is limited three ways**, all of them 422 rather than silent
truncation:

* at most 32768 characters, a plain size guard on the payload;
* no whitespace-free run longer than 256 characters (a 30k-character "word" or
  URL counts as one word but costs thousands of subword tokens);
* the model's word budget (`max_len` in its config; 4096 for the GLiNER2.5
  models) counted with gliner2's word splitter, in which punctuation marks
  count as words, **and** `GLINER_MAX_TOKENS` subword tokens counted with the
  model's own tokenizer. The subword count is the one that actually bounds
  inference cost -- scripts such as Chinese tokenize to roughly one subword per
  character, so a few hundred "words" can be tens of thousands of tokens.

Responses report `input.words` / `input.max_words` and `input.tokens` /
`input.max_tokens`; `max_tokens` is also in `GET /health`.

`POST /api/decide/batch` asks several questions about the *same* passage in one
call and answers them in **one forward pass**, using gliner2's own multi-task
schema: the passage is encoded once however many decisions ride on it. On a
~640-word passage, four decisions (ten choices in total) cost 0.14s that way
against 0.46s as four separate requests, and the gap widens with the passage.

**The decisions in a batch are not scored in isolation from one another.** Their
labels share one encoder sequence with the passage and attend to each other, so
a decision's confidences depend on which other decisions travelled with it and
differ from what `POST /api/decide` returns for that decision alone. Measured on
`base`: adding three unrelated decisions moved one choice from 0.0029 to 0.0346,
and over a sweep of the five sample passages 13 of 120 combinations changed
which choice ranked first. The same batch always gives the same answer, and
other clients' traffic never affects it -- but changing, adding or reordering
the decisions changes all of them. Group questions that belong together; ask
`POST /api/decide` when a decision must be scored on its own terms.

**One inference worker serves everybody.** Requests queue FIFO and are answered
in order; jobs waiting for the same loaded model are coalesced into one batched
forward pass. When the queue is full, or a request waits longer than
`GLINER_QUEUE_TIMEOUT` *for the worker to pick it up*, the answer is a 503 with
a `Retry-After` header. Once the job is running that clock stops, so a slow
forward pass never times out its own request; only a wedged worker does, after a
600s safety ceiling, as a 504. Watch `queue` and `stats` in `GET /health`.
"""

app = FastAPI(
    title="Minos API",
    description=DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
)

# Opt-in only. The API has no authentication, so a wildcard CORS policy would
# let any web page a teammate happens to visit drive this server. The bundled
# UI is served from this same origin and needs no CORS at all.
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _clean_text(v: str) -> str:
    """The `text` validator, shared by both request bodies so a passage is
    accepted or refused identically whichever endpoint it arrives at."""
    v = v.strip()
    if not v:
        raise ValueError("text must not be empty")
    if len(v) > MAX_TEXT_CHARS:
        raise ValueError(f"text must be at most {MAX_TEXT_CHARS} characters")
    return v


TEXT_DESCRIPTION = (
    "The situation to decide on. Stripped of surrounding whitespace; "
    f"must be non-empty and at most {MAX_TEXT_CHARS} characters, with no "
    "whitespace-free run longer than 256 characters. Those caps are size guards; "
    "the real limits are the model's word budget and its subword-token budget "
    "(see `input.max_words` and `input.max_tokens` in the response). Exceeding "
    "any of them is a 422."
)

TEXT_EXAMPLE = ("The vendor missed the last two delivery deadlines and quality "
                "inspections show a 12% defect rate.")

CHOICES_EXAMPLE = ["keep the current vendor", "switch to a new vendor",
                   "escalate to legal for breach of contract"]


def _model_field():
    # A factory, not a shared Field(...) instance: pydantic takes ownership of a
    # FieldInfo when it builds a model, so handing the same object to two models
    # is asking for trouble.
    return Field(
        default=None,
        description=f"Model alias ({', '.join(MODEL_ALIASES)}) or a full Hugging Face repo id. "
        f"Defaults to the server's preloaded model.",
        examples=["base"],
    )


def _clean_model(v: str | None) -> str | None:
    if v is None:
        return None
    v = v.strip()
    return v or None


class DecideRequest(BaseModel):
    text: str = Field(description=TEXT_DESCRIPTION, examples=[TEXT_EXAMPLE])
    choices: list[str] = Field(
        description=f"The candidate choices to rank, {MIN_CHOICES}-{MAX_CHOICES} of them. "
        "Each is stripped and must be non-empty; duplicates are rejected.",
        examples=[CHOICES_EXAMPLE],
    )
    model: str | None = _model_field()

    @field_validator("text")
    @classmethod
    def _check_text(cls, v: str) -> str:
        return _clean_text(v)

    @field_validator("choices")
    @classmethod
    def _check_choices(cls, v: list[str]) -> list[str]:
        # common.clean_choices, so the CLI's local path enforces exactly this.
        return clean_choices(v)

    @field_validator("model")
    @classmethod
    def _check_model(cls, v: str | None) -> str | None:
        return _clean_model(v)


class ScoredChoice(BaseModel):
    label: str = Field(description="The choice, exactly as submitted (whitespace-stripped).",
                       examples=["switch to a new vendor"])
    confidence: float = Field(
        description="How well this choice fits the passage, 0..1. Independent of the "
        "other choices: the scores in a response do not sum to 1.",
        examples=[0.83],
    )


class Timing(BaseModel):
    load_s: float = Field(description="Seconds spent loading the model; 0.0 when it was already cached.",
                          examples=[0.0])
    inference_s: float = Field(description="Seconds spent scoring the choices. When this request "
                               "was batched, it is the whole batch's forward pass.", examples=[0.031])
    queued_s: float = Field(description="Seconds the request waited in the queue before the "
                            "inference worker picked it up.", examples=[0.12])
    batch_size: int = Field(description="How many requests shared this forward pass; 1 when the "
                            "request ran on its own.", examples=[1])


class InputInfo(BaseModel):
    words: int = Field(description="Word tokens in the submitted passage, counted with gliner2's "
                       "word splitter (punctuation marks count as words).",
                       examples=[23])
    max_words: int = Field(description="The word budget of the model that scored this request.",
                           examples=[4096])
    tokens: int | None = Field(
        description="Subword tokens the passage costs the model's own tokenizer -- the "
        "figure inference cost actually tracks. Null only if the model exposes no "
        "tokenizer.", examples=[31])
    max_tokens: int = Field(description="GLINER_MAX_TOKENS: the subword-token ceiling.",
                            examples=[8192])


class DecideResponse(BaseModel):
    best: str = Field(description="The highest-scoring choice; the first entry of `ranked`.",
                      examples=["switch to a new vendor"])
    ranked: list[ScoredChoice] = Field(description="Every choice with its score, sorted by confidence descending.")
    model: str = Field(description="The Hugging Face repo id that produced the scores.",
                       examples=["fastino/gliner2.5-base-v1"])
    device: str = Field(description="The torch device the model ran on.", examples=["mps"])
    timing: Timing
    input: InputInfo


class Decision(BaseModel):
    name: str | None = Field(
        default=None,
        description="An optional label for this decision, echoed back on its result so a "
        "client can match answers to questions without relying on order. Must be unique "
        "within the request when given.",
        examples=["vendor action"],
    )
    choices: list[str] = Field(
        description=f"The candidate choices to rank, {MIN_CHOICES}-{MAX_CHOICES} of them. "
        "Each is stripped and must be non-empty; duplicates are rejected *within* this "
        "decision. Separate decisions are independent and may repeat each other's choices.",
        examples=[CHOICES_EXAMPLE],
    )

    @field_validator("choices")
    @classmethod
    def _check_choices(cls, v: list[str]) -> list[str]:
        return clean_choices(v)

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None


class BatchDecideRequest(BaseModel):
    text: str = Field(description=TEXT_DESCRIPTION, examples=[TEXT_EXAMPLE])
    decisions: list[Decision] = Field(
        min_length=1,
        max_length=MAX_DECISIONS,
        description=f"1-{MAX_DECISIONS} questions to ask about `text`, at most "
        f"{MAX_TOTAL_CHOICES} choices across all of them. They are answered in one "
        "forward pass and are *not* scored in isolation from each other -- see the "
        "endpoint description.",
    )
    model: str | None = _model_field()

    @field_validator("text")
    @classmethod
    def _check_text(cls, v: str) -> str:
        return _clean_text(v)

    @field_validator("model")
    @classmethod
    def _check_model(cls, v: str | None) -> str | None:
        return _clean_model(v)

    @model_validator(mode="after")
    def _check_batch(self) -> "BatchDecideRequest":
        # The whole batch is one forward pass, and its cost is driven by the
        # total number of labels in the schema rather than by how they are
        # grouped: 16 decisions of 20 choices is 320 labels and 9.4s on a
        # full-length passage, against 4.3s for the 20 a single request may ask.
        total = sum(len(d.choices) for d in self.decisions)
        if total > MAX_TOTAL_CHOICES:
            raise ValueError(
                f"decisions contain {total} choices in total; at most "
                f"{MAX_TOTAL_CHOICES} are accepted across one batch")
        # Task labels are the model's prompt AND the key results come back
        # under, so they have to be distinct -- including against the
        # "decision N" fallback an unnamed decision gets.
        labels = task_labels(self.decisions)
        if len(set(labels)) != len(labels):
            raise ValueError(
                "decision names must be unique, and must not collide with the "
                f"'{DEFAULT_TASK_LABEL} N' label an unnamed decision is given")
        return self


def task_labels(decisions: list[Decision]) -> list[str]:
    """The classification task label for each decision.

    A decision's `name` when it has one -- the label is the prompt the model
    sees, so a meaningful name is worth more here than a generated one. Failing
    that, "decision N" by position, except that a lone unnamed decision gets
    plain "decision": that makes a one-decision batch request literally the same
    schema as POST /api/decide, and so the same scores."""
    if len(decisions) == 1 and decisions[0].name is None:
        return [DEFAULT_TASK_LABEL]
    return [d.name if d.name is not None else f"{DEFAULT_TASK_LABEL} {i}"
            for i, d in enumerate(decisions, 1)]


class DecisionResult(BaseModel):
    index: int = Field(description="Position of this decision in the request's `decisions` "
                       "array. Results come back in request order; `index` is here so a "
                       "client that reorders or filters can still map back.", examples=[0])
    name: str | None = Field(description="The decision's `name`, or null when it had none.",
                             examples=["vendor action"])
    task: str = Field(description="The classification task label this decision was scored "
                      "under -- its `name`, or a generated \"decision N\". This is part of "
                      "the prompt the model saw, not just a key.", examples=["vendor action"])
    best: str = Field(description="The highest-scoring choice for this decision.",
                      examples=["switch to a new vendor"])
    ranked: list[ScoredChoice] = Field(description="Every choice in this decision with its "
                                       "score, sorted by confidence descending.")


class BatchDecideResponse(BaseModel):
    results: list[DecisionResult] = Field(
        description="One result per requested decision, in request order.")
    model: str = Field(description="The Hugging Face repo id that produced the scores.",
                       examples=["fastino/gliner2.5-base-v1"])
    device: str = Field(description="The torch device the model ran on.", examples=["mps"])
    timing: Timing = Field(description="Exactly as for POST /api/decide: every decision in "
                           "the batch shared this one forward pass, so there is one set of "
                           "numbers, and `batch_size` still counts *requests* coalesced by "
                           "the worker, not decisions.")
    input: InputInfo = Field(description="How the passage was counted and the limits it was "
                             "checked against. One object, not one per decision: there is "
                             "one passage, encoded once.")


class ModelInfo(BaseModel):
    alias: str = Field(description="Short name accepted by the `model` request field.", examples=["base"])
    repo: str = Field(description="The Hugging Face repo the alias points at.",
                      examples=["fastino/gliner2.5-base-v1"])
    loaded: bool = Field(description="Whether this model is currently cached in memory. "
                         "Requests for an unloaded model pay a one-time load cost.",
                         examples=[True])
    max_words: int | None = Field(
        description="The model's word budget: from the loaded model, else from its cached "
        "config.json, else null when the repo is not in the local Hugging Face cache.",
        examples=[4096],
    )


class ModelsResponse(BaseModel):
    default: str = Field(description="The alias or repo id used when a request omits `model`.",
                         examples=["base"])
    models: list[ModelInfo]


class QueueInfo(BaseModel):
    depth: int = Field(description="Jobs waiting for the worker right now (excluding the one "
                       "being processed).", examples=[3])
    max: int = Field(description="GLINER_MAX_QUEUE: depth at which new requests get a 503.",
                     examples=[64])
    busy: bool = Field(description="Whether the worker is running a forward pass or a model load.",
                       examples=[True])
    max_batch: int = Field(description="GLINER_MAX_BATCH: most jobs coalesced into one forward pass.",
                           examples=[8])
    max_decisions: int = Field(description="GLINER_MAX_DECISIONS: most decisions allowed in one "
                               "POST /api/decide/batch body.", examples=[16])
    max_total_choices: int = Field(description="GLINER_MAX_TOTAL_CHOICES: most choices allowed "
                                   "across all the decisions in one batch. The batch is a "
                                   "single forward pass and this is what bounds its cost.",
                                   examples=[120])


class QueueStats(BaseModel):
    served: int = Field(description="Requests answered successfully since startup.", examples=[1024])
    rejected_busy: int = Field(description="Requests rejected with 503 because the queue was full.",
                               examples=[7])
    timed_out: int = Field(description="Requests that gave up with 503 after waiting "
                           "GLINER_QUEUE_TIMEOUT to be picked up. Jobs already running are "
                           "never counted here: they are left to finish.", examples=[0])
    batches: int = Field(description="Forward passes run by the worker. served/batches is the "
                         "average realised batch size.", examples=[260])


class HealthResponse(BaseModel):
    status: str = Field(
        description="\"ok\" when the server is serving normally, \"degraded\" when the default "
        "model failed to preload at startup. A degraded server still answers requests that "
        "name a working model; requests for the broken default return 400.",
        examples=["ok"],
    )
    error: str | None = Field(description="Why the server is degraded, or null when it is ok.",
                              examples=[None])
    device: str = Field(description="The resolved torch device all models run on.", examples=["mps"])
    loaded_models: list[str] = Field(description="Repo ids currently cached in memory.",
                                     examples=[["fastino/gliner2.5-base-v1"]])
    max_tokens: int = Field(
        description="GLINER_MAX_TOKENS: subword tokens a passage may contain, whatever "
        "model scores it. Per-model word budgets are in GET /api/models.",
        examples=[8192],
    )
    queue: QueueInfo
    stats: QueueStats


def _max_words(repo: str) -> int | None:
    """Word budget without loading anything: loaded model first, then the local
    HF cache, then None."""
    model = _models.get(repo)
    if model is not None:
        return max_words_for(model)
    return cached_max_words(repo)


@app.get("/health", response_model=HealthResponse, summary="Liveness, queue and loaded-model check")
def health() -> HealthResponse:
    # Deliberately lock-free apart from the tiny stats lock: this must answer in
    # milliseconds while the worker is saturated. len(deque) and a bool read are
    # atomic enough; the numbers can be off by one mid-flight.
    with _stats_lock:
        stats = dict(_stats)
    return HealthResponse(
        status="degraded" if _preload_error else "ok",
        error=_preload_error,
        device=DEVICE,
        loaded_models=list(_models),
        max_tokens=MAX_TOKENS,
        queue=QueueInfo(depth=len(_pending), max=MAX_QUEUE, busy=_busy, max_batch=MAX_BATCH,
                        max_decisions=MAX_DECISIONS, max_total_choices=MAX_TOTAL_CHOICES),
        stats=QueueStats(**stats),
    )


@app.get("/api/models", response_model=ModelsResponse, summary="List known model aliases")
def list_models() -> ModelsResponse:
    return ModelsResponse(
        default=MODEL,
        models=[
            ModelInfo(alias=alias, repo=repo, loaded=repo in _models, max_words=_max_words(repo))
            for alias, repo in MODEL_ALIASES.items()
        ],
    )


def _resolve_allowed(name: str | None) -> str:
    """Resolve `model` to a repo id, refusing unknown repos unless
    GLINER_ALLOW_ANY_MODEL is set. The single worker is shared, so an arbitrary
    repo id lets any client make the server download ~1 GB -- or, with the hub
    unreachable, pin the worker for minutes of Hugging Face retries."""
    raw = name or MODEL
    repo = resolve_model_name(raw)
    if ALLOW_ANY_MODEL:
        return repo
    if repo not in set(MODEL_ALIASES.values()) | {resolve_model_name(MODEL)}:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{raw}'. Allowed: {', '.join(MODEL_ALIASES)}. "
            "Set GLINER_ALLOW_ANY_MODEL=1 to permit arbitrary Hugging Face repos.",
        )
    return repo


async def _await_job(job: Job, repo: str) -> dict:
    """Wait for an enqueued job and return its result, or raise the HTTPException
    the failure deserves. Shared by both decide endpoints, which differ only in
    what they build from the result."""
    # The queue timeout covers the *wait*, not the work: it stops the moment the
    # worker commits to this job. Timing out a request whose forward pass is
    # already running threw away GPU seconds already spent on it, and the head
    # of the queue -- the longest job, the one most likely to be mid-pass --
    # was exactly the one it hit.
    try:
        await asyncio.wait_for(asyncio.wrap_future(job.started), timeout=QUEUE_TIMEOUT)
    except asyncio.TimeoutError:
        if job.future.cancel():
            # Cancelled before the worker took it. Drop it out of the queue too:
            # a cancelled job left behind still counted against MAX_QUEUE, so
            # after a timeout storm /health reported a queue full of nobody and
            # fresh requests got 503s. (Already gone if the worker raced us.)
            with _cond:
                try:
                    _pending.remove(job)
                except ValueError:
                    pass
            _bump("timed_out")
            raise HTTPException(
                status_code=503,
                detail=f"Timed out after {QUEUE_TIMEOUT}s waiting for the inference worker.",
                headers={"Retry-After": "5"},
            ) from None
        # cancel() lost: the worker is already running this job. Wait for it.

    try:
        out = await asyncio.wait_for(asyncio.wrap_future(job.future), timeout=INFERENCE_CEILING)
    except asyncio.TimeoutError:
        log.error("Worker exceeded the %.0fs inference ceiling on %r", INFERENCE_CEILING, repo)
        raise HTTPException(
            status_code=504,
            detail=f"The inference worker took the request but did not finish within "
                   f"{INFERENCE_CEILING:g}s. It is likely wedged; check the server log.",
            headers={"Retry-After": "30"},
        ) from None
    except ModelLoadError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except InputTooLong as e:
        raise HTTPException(status_code=422, detail=_input_detail(e)) from None
    except Exception:
        log.exception("Inference failed for %r", repo)
        raise HTTPException(status_code=500, detail="Inference failed. See the server log.") from None

    return out


@app.post(
    "/api/decide",
    response_model=DecideResponse,
    summary="Rank candidate choices against a passage",
    responses={
        400: {"description": "The requested model is not allowed, or could not be loaded."},
        422: {"description": "The request failed validation, or the passage exceeds one of "
                             "the input size limits (characters, whitespace-free run, the "
                             "model's word budget, GLINER_MAX_TOKENS subword tokens)."},
        503: {"description": "The queue is full, or the request waited longer than "
                             "GLINER_QUEUE_TIMEOUT to be picked up. Both carry a "
                             "Retry-After header."},
        504: {"description": "The worker took the job but did not finish within 600s -- it "
                             "is wedged, not merely slow."},
    },
)
async def api_decide(req: DecideRequest) -> DecideResponse:
    repo = _resolve_allowed(req.model)

    job = Job(repo=repo, text=req.text, tasks=[(DEFAULT_TASK_LABEL, req.choices)])
    model = _models.get(repo)
    if model is not None:
        # Already loaded, so the limits are knowable here: reject a doomed
        # request now instead of letting it wait for the worker. On a thread,
        # never inline: the word scan is O(text) and its regex backtracks
        # quadratically on punctuation runs, which froze /health for 21s.
        try:
            job.info = await asyncio.to_thread(check_input, model, req.text, repo)
        except InputTooLong as e:
            raise HTTPException(status_code=422, detail=_input_detail(e)) from None

    try:
        _enqueue(job)
    except QueueFull:
        _bump("rejected_busy")
        raise HTTPException(
            status_code=503,
            detail=f"Server busy: {MAX_QUEUE} requests already queued. Retry shortly.",
            headers={"Retry-After": "2"},
        ) from None

    out = await _await_job(job, repo)

    _bump("served")
    ranked = out["scored"][DEFAULT_TASK_LABEL]
    return DecideResponse(
        best=ranked[0]["label"],
        ranked=[ScoredChoice(**r) for r in ranked],
        model=repo,
        device=DEVICE,
        timing=Timing(load_s=out["load_s"], inference_s=out["inference_s"],
                      queued_s=out["queued_s"], batch_size=out["batch_size"]),
        input=InputInfo(words=out["words"], max_words=out["max_words"],
                        tokens=out["tokens"], max_tokens=out["max_tokens"]),
    )


@app.post(
    "/api/decide/batch",
    response_model=BatchDecideResponse,
    summary="Ask several questions about one passage, in one forward pass",
    description=(
        "Several independent questions about the **same** passage, answered in a single "
        "forward pass: gliner2 takes a list of classification tasks in one schema, so the "
        "passage is encoded once however many decisions ride on it.\n\n"
        "**The decisions are not scored in isolation from one another.** Every decision's "
        "labels sit in the same encoder sequence as the passage and attend to each other, "
        "so a decision's confidences depend on which other decisions travelled with it, and "
        "differ from what `POST /api/decide` returns for that decision alone. The same "
        "batch always gives the same answer -- other clients' requests never affect it -- "
        "but changing, adding or reordering the decisions changes every one of them. Ask "
        "questions that belong together, and use `POST /api/decide` when a decision has to "
        "be scored on its own terms.\n\n"
        "A decision's `name` is its task label, which is part of the prompt the model sees, "
        "so it is worth naming decisions meaningfully (`\"urgency\"`, not `\"q2\"`). "
        "Unnamed decisions are labelled `decision 1`, `decision 2`, and so on."
    ),
    responses={
        400: {"description": "The requested model is not allowed, or could not be loaded."},
        422: {"description": "The request failed validation, or the passage exceeds one of "
                             "the input size limits. There is one passage, so it is checked "
                             "once, exactly as for POST /api/decide."},
        503: {"description": "The queue is full, or the request waited longer than "
                             "GLINER_QUEUE_TIMEOUT to be picked up. Both carry a "
                             "Retry-After header."},
        504: {"description": "The worker took the job but did not finish within 600s -- it "
                             "is wedged, not merely slow."},
    },
)
async def api_decide_batch(req: BatchDecideRequest) -> BatchDecideResponse:
    """One job, one forward pass, N decisions.

    Deliberately one queue slot and not N: the decisions share a single
    forward pass, so charging the queue per decision would price a batch as if
    it were a crowd. What it costs instead is bounded by MAX_DECISIONS and
    MAX_TOTAL_CHOICES, because the schema rides in the encoder sequence with
    the passage and attention is quadratic in the total length.

    The decisions are not independent in this arrangement -- see the endpoint
    description and common.decide_batch(). That is the deal the endpoint makes
    in exchange for encoding the passage once, and it is documented rather than
    hidden because it can change which choice ranks first.
    """
    repo = _resolve_allowed(req.model)
    labels = task_labels(req.decisions)

    job = Job(repo=repo, text=req.text,
              tasks=[(label, d.choices) for label, d in zip(labels, req.decisions)])
    model = _models.get(repo)
    if model is not None:
        # Already loaded, so the limits are knowable here: reject a doomed
        # request now instead of letting it wait for the worker. On a thread,
        # never inline -- see api_decide.
        try:
            job.info = await asyncio.to_thread(check_input, model, req.text, repo)
        except InputTooLong as e:
            raise HTTPException(status_code=422, detail=_input_detail(e)) from None

    try:
        _enqueue(job)
    except QueueFull:
        _bump("rejected_busy")
        raise HTTPException(
            status_code=503,
            detail=f"Server busy: {MAX_QUEUE} requests already queued. Retry shortly.",
            headers={"Retry-After": "2"},
        ) from None

    out = await _await_job(job, repo)

    _bump("served")
    scored = out["scored"]
    return BatchDecideResponse(
        results=[
            DecisionResult(
                index=i,
                name=decision.name,
                task=label,
                best=scored[label][0]["label"],
                ranked=[ScoredChoice(**r) for r in scored[label]],
            )
            for i, (decision, label) in enumerate(zip(req.decisions, labels))
        ],
        model=repo,
        device=DEVICE,
        timing=Timing(load_s=out["load_s"], inference_s=out["inference_s"],
                      queued_s=out["queued_s"], batch_size=out["batch_size"]),
        input=InputInfo(words=out["words"], max_words=out["max_words"],
                        tokens=out["tokens"], max_tokens=out["max_tokens"]),
    )


@app.get("/", include_in_schema=False)
def index():
    if INDEX_HTML.is_file():
        return FileResponse(INDEX_HTML)
    return {"message": "Minos API. UI not installed.", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    print(f"Serving on http://{HOST}:{PORT}  (docs: /docs)")
    uvicorn.run(app, host=HOST, port=PORT)
