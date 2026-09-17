# Minos

A small zero-shot "decision helper". You give it a text passage describing a
situation plus a list of candidate choices, and it scores how well each choice
fits the passage and returns them ranked best-first. It runs entirely locally on
a Mac using [GLiNER2.5](https://huggingface.co/fastino) models (default:
`fastino/gliner2.5-base-v1`, about 400 MB). No API keys, no data leaves the
machine. Ships with a web UI, an HTTP API, and a CLI.

Read the output honestly. The confidences are **independent per-choice scores,
not a probability distribution** - they do not sum to 100%, and every choice can
score high or every choice can score low. The model matches the *semantics of
your choice labels* against the passage; it does **not** reason about the
situation, weigh trade-offs, or know anything you did not put in the text. How
you phrase a choice changes its score. Different model sizes routinely disagree
on the same input. Treat a result as a signal worth a second look, never as a
verdict.

## Quickstart (this machine)

```bash
./setup.sh      # creates .venv, installs deps, pre-downloads the model
./start.sh      # serves on 0.0.0.0:8765
```

`start.sh` prints the local URL, the LAN URL, and the docs URL. Open the local
URL for the web UI, or `http://localhost:8765/docs` for interactive Swagger
docs. Ctrl-C stops it.

## Deploy on another MacBook

1. **Copy the folder**, leaving the virtualenv and logs behind:

   ```bash
   rsync -a --exclude .venv --exclude logs --exclude __pycache__ \
         --exclude '*.log' --exclude .DS_Store \
         ~/dev/ai/gliner/ othermac:~/gliner/
   ```

   If you keep it in git, `git clone <repo> ~/gliner` works just as well - the
   `.gitignore` already excludes `.venv/` and `logs/`.

2. **Run setup** on the other Mac:

   ```bash
   cd ~/gliner && ./setup.sh
   ```

   The first run downloads roughly 130 MB of PyTorch plus the ~400 MB model, so
   give it a few minutes on a fresh machine. Re-runs take seconds. Needs Python
   3.10 or newer; `setup.sh` finds one in PATH and tells you to
   `brew install python@3.12` if it cannot.

   **Intel Macs: Python 3.12 only.** gliner2 needs `torch >= 2.1`, and the last
   x86_64 macOS torch wheel is 2.2.2, which ships a cp312 build and nothing
   newer. On an Intel Mac with a current Homebrew, a 3.13/3.14 venv has no
   installable torch at all. `setup.sh` therefore prefers `python3.12` when
   several versions are present; install it with `brew install python@3.12`.
   Inference on Intel is CPU-only (no MPS) and correspondingly slower.

3. **Start it** and share the LAN URL that `./start.sh` prints (something like
   `http://192.168.1.50:8765`). The first time, macOS asks *"Do you want the
   application Python to accept incoming network connections?"* - choose
   **Allow**, or the LAN URL will not be reachable.

4. **Optional: run it in the background** so it comes back automatically:

   ```bash
   ./install-service.sh      # installs a launchd user agent, starts it now
   ./uninstall-service.sh    # stops and removes it
   ```

   Any `GLINER_*` variable set in the environment when you run
   `install-service.sh` is written into the plist - for example
   `GLINER_PORT=9000 GLINER_MAX_BATCH=4 ./install-service.sh`. Anything you do
   not set is left out, so `server.py`'s own defaults apply. It logs to
   `logs/server.log`. Re-run `install-service.sh` to change settings.

   The agent's label is `com.minos.server`. It used to be
   `com.gliner-decide.server`; `install-service.sh` retires an agent still
   carrying the old label before installing the new one, and
   `uninstall-service.sh` removes either.

   **This is a LaunchAgent in `gui/<uid>`, so it starts at *login*, not at
   boot.** A Mac that reboots and sits at the login window serves nothing until
   somebody logs in; enable automatic login for that account if it must survive
   reboots unattended (System Settings → Users & Groups → Automatic login).

To keep it off the network entirely, bind to loopback:
`GLINER_HOST=127.0.0.1 ./start.sh`.

**There is no authentication.** Anyone who can reach the port can use the model.
Run it on a trusted LAN only, never expose it to the internet.

## Web UI

`GET /` serves a single-page UI: paste the situation, add choices, pick a model,
and it shows the ranked list with confidence bars. It talks to the same
`/api/decide` endpoint documented below.

A live word counter sits above the Situation box (`≈ 53 / 4,096 words`), and the
line under it spells out all four [input limits](#input-limits). The counter is
a **client-side approximation** of gliner2's splitter - it runs in the browser
on every keystroke and never calls the server, which is why it always shows `≈`.
The word limit comes from the selected model's `max_words` in `/api/models`, so
the counter re-counts when you switch models.

Going over turns the counter red and adds a hint, but **Decide stays enabled**:
the estimate is not authoritative, so the exact counts are left to the server,
which rejects an over-limit passage with a 422 that the UI renders like any
other validation error. Past 40 000 characters the counter stops counting and
says `≈ too long (over 32,768 characters)`: the word regex backtracks
quadratically on long punctuation runs (786 ms on 45 000 dashes, measured here)
and would stutter the page on every keystroke.

The model dropdown's first entry is **Server default · base** - the alias when
the server's default maps to one, otherwise its repo id. It sends no `model` at
all, so use it if you want exactly what the server preloaded.

The result footer carries the server's own exact counts (no `≈`) and its
timings: `30 / 4,096 words · 30 / 8,192 tokens`, plus `queued 295 ms` and
`batch of 8` under load. When the server is too busy to accept the request it
shows "Server is busy (N queued). Try again in a moment." - see
[Load and concurrency](#load-and-concurrency).

## HTTP API

Interactive docs: [`/docs`](http://localhost:8765/docs). Raw schema:
[`/openapi.json`](http://localhost:8765/openapi.json).

### `GET /health`

```bash
curl -s http://localhost:8765/health
```

```json
{"status":"ok","error":null,"device":"mps",
 "loaded_models":["fastino/gliner2.5-base-v1"],"max_tokens":8192,
 "queue":{"depth":0,"max":64,"busy":false,"max_batch":8,
         "max_decisions":16,"max_total_choices":120},
 "stats":{"served":1024,"rejected_busy":7,"timed_out":0,"batches":160}}
```

`queue` and `stats` describe the inference queue - see
[Load and concurrency](#load-and-concurrency). `max_decisions` and
`max_total_choices` are the caps on a single
[`POST /api/decide/batch`](#post-apidecidebatch) body. `/health` never queues, so it
answers in about 1 ms even while the worker is saturated, which makes it a
usable liveness probe under load.

`status` is `"degraded"` when the server is up but the startup preload failed -
a bad `GLINER_MODEL`, or no network on a cold weights cache. It stays up and
keeps answering instead of exiting, and `error` carries the reason (`null` when
`status` is `"ok"`):

```json
{"status":"degraded",
 "error":"Could not preload default model 'nope/does-not-exist': RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-6aabaef5-...)",
 "device":"mps","loaded_models":[],"max_tokens":8192,
 "queue":{"depth":0,"max":64,"busy":false,"max_batch":8,
         "max_decisions":16,"max_total_choices":120},
 "stats":{"served":0,"rejected_busy":0,"timed_out":0,"batches":0}}
```

`error` is always a single line (the full traceback goes to the server log).

Staying up is deliberate: a process that exited on a preload failure would be
restarted forever by launchd, and the plist's `ThrottleInterval` of 60 s keeps
even a genuine crash loop down to one restart a minute. So health-check on
`status`, not merely on the port being open.

### `GET /api/models`

```bash
curl -s http://localhost:8765/api/models
```

```json
{
  "default": "base",
  "models": [
    {"alias": "multi", "repo": "fastino/gliner2.5-multi-v1", "loaded": false, "max_words": 4096},
    {"alias": "base",  "repo": "fastino/gliner2.5-base-v1",  "loaded": true,  "max_words": 4096},
    {"alias": "small", "repo": "fastino/gliner2.5-small-v1", "loaded": false, "max_words": 4096}
  ]
}
```

`max_words` is `null` for a model whose config is not cached on disk yet.

### `POST /api/decide`

`text` (non-empty and within every [input limit](#input-limits): the model's
word budget, 8192 subword tokens, 32 768 characters, and no whitespace-free run
over 256 characters) and `choices` (2-20 non-empty, non-duplicate strings) are
required; `model` is optional and defaults to the server's preloaded model.

```bash
curl -s -X POST http://localhost:8765/api/decide \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The vendor missed the last two delivery deadlines and quality inspections show a 12% defect rate. They are our cheapest supplier and switching would take three months.",
    "choices": ["keep the current vendor", "switch to a new vendor", "escalate to legal for breach of contract"]
  }'
```

```json
{
  "best": "switch to a new vendor",
  "ranked": [
    {"label": "switch to a new vendor", "confidence": 0.9946658611297607},
    {"label": "keep the current vendor", "confidence": 0.01456294022500515},
    {"label": "escalate to legal for breach of contract", "confidence": 0.0003139041946269572}
  ],
  "model": "fastino/gliner2.5-base-v1",
  "device": "mps",
  "timing": {"load_s": 0.0, "inference_s": 0.013059000000794185,
             "queued_s": 0.00029691600138903596, "batch_size": 1},
  "input": {"words": 30, "max_words": 4096, "tokens": 30, "max_tokens": 8192}
}
```

`input` reports how the server counted the passage - in words and in subword
tokens - and the limits each was checked against; see
[Input limits](#input-limits). `timing.queued_s` is how long the
request waited for the inference worker and `timing.batch_size` how many
requests shared its forward pass - see
[Load and concurrency](#load-and-concurrency).

`model` must name a known alias (`base`, `multi`, `small`) or the
server's own `GLINER_MODEL` repo. Arbitrary Hugging Face repo ids are refused
with a 400 unless `GLINER_ALLOW_ANY_MODEL=1` is set - see
[Load and concurrency](#load-and-concurrency) for why.

Ask for a different model with `"model": "small"` (alias) or a full repo id. A
model that is not cached yet is loaded on that request and its cost shows up in
`timing.load_s`:

```json
{
  "best": "push the deadline",
  "ranked": [
    {"label": "push the deadline", "confidence": 0.5922466516494751},
    {"label": "cut scope", "confidence": 0.09745427221059799},
    {"label": "hire contractors", "confidence": 0.05295358598232269}
  ],
  "model": "fastino/gliner2.5-small-v1",
  "device": "mps",
  "timing": {"load_s": 1.0866251250008645, "inference_s": 0.02415820899841492,
             "queued_s": 0.0000707910003256984, "batch_size": 1},
  "input": {"words": 13, "max_words": 4096, "tokens": 13, "max_tokens": 8192}
}
```

**422 - invalid request body** (empty text, fewer than 2 or more than 20
choices, duplicates, empty choice strings):

```json
{"detail":[{"type":"value_error","loc":["body","choices"],
  "msg":"Value error, choices must contain between 2 and 20 items",
  "input":["only one"],"ctx":{"error":{}}}]}
```

**422 - over one of the [input limits](#input-limits)** (same shape, `loc`
points at `text`). All four read the same way, so a client renders them
identically; only `msg` differs:

```json
{"detail":[{"loc":["body","text"],
  "msg":"text is more than 4096 words; fastino/gliner2.5-base-v1 accepts at most 4096 words (counted with gliner2's word splitter, where punctuation marks count as words)",
  "type":"value_error"}]}
```

```json
{"detail":[{"loc":["body","text"],
  "msg":"text is 10560 subword tokens for fastino/gliner2.5-base-v1; at most 8192 are accepted (GLINER_MAX_TOKENS)",
  "type":"value_error"}]}
```

```json
{"detail":[{"loc":["body","text"],
  "msg":"text contains a 300-character run without spaces; the longest allowed is 256",
  "type":"value_error"}]}
```

```json
{"detail":[{"type":"value_error","loc":["body","text"],
  "msg":"Value error, text must be at most 32768 characters","ctx":{"error":{}}}]}
```

**400 - the requested model is not on the allowlist:**

```json
{"detail":"Unknown model 'someone/other-repo'. Allowed: multi, base, small. Set GLINER_ALLOW_ANY_MODEL=1 to permit arbitrary Hugging Face repos."}
```

**400 - the requested model could not be loaded:**

```json
{"detail":"Could not load model 'nope/does-not-exist': RepositoryNotFoundError. Check the alias or Hugging Face repo id."}
```

**503 - the queue is full, or the request waited too long.** Both carry a
`Retry-After` header (2 s and 5 s respectively) and are safe to retry:

```json
{"detail":"Server busy: 64 requests already queued. Retry shortly."}
{"detail":"Timed out after 60.0s waiting for the inference worker."}
```

**504 - the worker took the job but never finished.** The 600 s safety ceiling;
it means the worker is wedged, not that you are queued behind other work. Not
worth retrying blind - check `/health` and the server log.

### `POST /api/decide/batch`

Several questions about **one** passage, answered in **one forward pass**. The
situation is sent, counted and encoded once, however many decisions ride on it.

`text` is exactly as above. `decisions` is 1-16 objects, each with its own
`choices` (2-20, same rules) and at most 120 choices across the whole batch.
`model` applies to the batch. A decision's optional `name` is echoed back *and*
used as the model's task label - see below.

```bash
curl -s -X POST http://localhost:8765/api/decide/batch \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "The vendor missed the last two delivery deadlines and quality inspections show a 12% defect rate. They are our cheapest supplier and switching would take three months.",
    "decisions": [
      {"name": "vendor action", "choices": ["keep the current vendor", "switch to a new vendor", "escalate to legal for breach of contract"]},
      {"name": "urgency",       "choices": ["urgent", "can wait a quarter"]},
      {"choices": ["tell the board now", "handle it at the team level"]}
    ]
  }'
```

```json
{
  "results": [
    {"index": 0, "name": "vendor action", "task": "vendor action",
     "best": "switch to a new vendor",
     "ranked": [
       {"label": "switch to a new vendor", "confidence": 0.9964410662651062},
       {"label": "keep the current vendor", "confidence": 0.013312903232872486},
       {"label": "escalate to legal for breach of contract", "confidence": 0.0011653545079752803}
     ]},
    {"index": 1, "name": "urgency", "task": "urgency", "best": "urgent",
     "ranked": [
       {"label": "urgent", "confidence": 0.8670076727867126},
       {"label": "can wait a quarter", "confidence": 0.31603559851646423}
     ]},
    {"index": 2, "name": null, "task": "decision 3",
     "best": "handle it at the team level",
     "ranked": [
       {"label": "handle it at the team level", "confidence": 0.262532114982605},
       {"label": "tell the board now", "confidence": 0.07403106987476349}
     ]}
  ],
  "model": "fastino/gliner2.5-base-v1",
  "device": "mps",
  "timing": {"load_s": 0.0, "inference_s": 0.04706758399697719,
             "queued_s": 0.0003902500029653311, "batch_size": 1},
  "input": {"words": 30, "max_words": 4096, "tokens": 30, "max_tokens": 8192}
}
```

`results` comes back in request order. `input` is one object, not one per
decision - there is one passage, encoded once. `timing` is the same shape as a
single request's, because that is what this is: one job, one forward pass.
`batch_size` still counts *requests* the worker coalesced, not decisions.

It costs one queue slot, not one per decision. Measured against sending the same
four decisions as four separate requests:

| Passage | One batch | Four separate requests | |
| --- | --- | --- | --- |
| 53 words | 0.019 s | 0.049 s | 2.6x |
| ~640 words | 0.138 s | 0.466 s | 3.4x |

The longer the situation, the more there is to amortise - which is the whole
point of the endpoint.

#### The decisions are not scored in isolation from each other

This is the trade, and it is not a small one. gliner2 puts every classification
task in the *same encoder sequence* as the passage, so the decisions in a batch
attend to one another. A decision's confidences therefore depend on which other
decisions travelled with it, and differ from what `POST /api/decide` returns for
that decision on its own.

Scoring one 3-choice decision on a fixed passage while adding unrelated
companion decisions to the same batch, `base` model:

| Also in the batch | `switch to a new vendor` | `keep the current vendor` | `escalate to legal` |
| --- | --- | --- | --- |
| nothing (scored alone) | 0.995393 | 0.004239 | 0.002871 |
| + urgency | 0.996224 | 0.005049 | 0.005277 |
| + urgency, audience | 0.992549 | 0.012629 | 0.018767 |
| + urgency, audience, budget | 0.990948 | 0.016327 | 0.034638 |
| the same three, reordered | 0.990419 | 0.017962 | 0.034303 |

The ranking survives in that table, but it does not always. Sweeping the five
`samples/` passages against six two- and three-choice decisions with one to four
companions, **13 of 120 combinations changed which choice came first.** The
clearest: on `samples/hiring_candidate.txt`, a decision between `act this week`
and `act next quarter` answers `act next quarter` (0.187 against 0.092) on its
own, and `act this week` (0.524 against 0.095) as soon as a single unrelated
decision joins the batch - and stays flipped with two, three and four.

Close calls are where this bites. A decision whose top choice wins by a wide
margin is unlikely to move; one that is nearly a tie can go either way depending
on what it is asked alongside.

What this does and does not mean:

* **Deterministic.** The same batch always returns the same answer. Other
  clients' traffic never affects it: requests the worker coalesces are separate
  rows in the batch and cannot see each other, only decisions *within* one
  request share a sequence. Verified under 24-way concurrent load, delta 0.
* **Order-sensitive.** Adding, removing or reordering decisions changes all of
  them. Treat a batch as one question with several parts, not as N independent
  queries that happen to be posted together.
* **Group what belongs together.** Questions about the same situation that you
  would want answered consistently are exactly the right thing to batch. A
  decision that has to be scored on its own terms belongs in `POST /api/decide`.
* **A one-decision batch is the single endpoint.** With one unnamed decision the
  schema is literally the one `POST /api/decide` builds, so the scores are
  bit-identical. That is a checked property, not a coincidence.

#### `name` is the prompt

A decision's `name` becomes its classification task label, which the model reads
as part of the prompt - so it is worth naming decisions for the model, not just
for yourself: `"urgency"` rather than `"q2"`. The label is echoed back as `task`
so you can always see what was actually asked. Decisions without a `name` are
labelled `decision 1`, `decision 2` and so on (a lone unnamed decision gets
plain `decision`, which is what makes the equivalence above hold).

The label does move the numbers, though much less than the company a decision
keeps. The same decision as the table above, alone in its batch, relabelled
(the `vendor action` row is that table's first row):

| Task label | `switch to a new vendor` | `keep the current vendor` | `escalate to legal` |
| --- | --- | --- | --- |
| `decision` | 0.994460 | 0.007458 | 0.002061 |
| `vendor action` | 0.995393 | 0.004239 | 0.002871 |
| `what should we do about the vendor` | 0.979268 | 0.011657 | 0.002149 |
| `urgency` | 0.993781 | 0.001811 | 0.005347 |
| `xyzzy` | 0.984925 | 0.011873 | 0.011554 |

Names must be unique within a request, and must not collide with the
`decision N` label an unnamed decision would get - both are a 422, because the
label is the key results come back under.

#### Limits and errors

`GLINER_MAX_DECISIONS` (16) and `GLINER_MAX_TOTAL_CHOICES` (120) bound the one
forward pass a batch becomes. The cost is driven by the total number of labels
rather than how they are grouped: on a full 4081-word passage, 20 choices - the
most a single request can ask for, and so a cost this server already accepts -
takes 4.3 s, 120 choices takes 5.7 s, and the 320 that 16 x 20 would allow takes
9.4 s. 120 keeps the worst case within a third of an ordinary request's.

Errors are the same as `POST /api/decide`; one passage means one 422 for an
over-long passage, not one per decision. The batch-specific ones:

```json
{"detail":[{"type":"too_long","loc":["body","decisions"],
  "msg":"List should have at most 16 items after validation, not 17"}]}
{"detail":[{"type":"value_error","loc":["body"],
  "msg":"Value error, decisions contain 122 choices in total; at most 120 are accepted across one batch"}]}
{"detail":[{"type":"value_error","loc":["body"],
  "msg":"Value error, decision names must be unique, and must not collide with the 'decision N' label an unnamed decision is given"}]}
```

### From Python (stdlib only)

```python
import json, urllib.request

payload = json.dumps({
    "text": "The team is burned out and the deadline is in two weeks.",
    "choices": ["cut scope", "hire contractors", "push the deadline"],
}).encode()
req = urllib.request.Request("http://localhost:8765/api/decide", data=payload,
                             headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req))["best"])
```

Several questions about the same situation, in one call and one forward pass
(the decisions influence each other - see
[above](#the-decisions-are-not-scored-in-isolation-from-each-other)):

```python
payload = json.dumps({
    "text": "The team is burned out and the deadline is in two weeks.",
    "decisions": [
        {"name": "plan", "choices": ["cut scope", "hire contractors", "push the deadline"]},
        {"name": "comms", "choices": ["tell the client now", "wait for the next check-in"]},
    ],
}).encode()
req = urllib.request.Request("http://localhost:8765/api/decide/batch", data=payload,
                             headers={"Content-Type": "application/json"})
for result in json.load(urllib.request.urlopen(req))["results"]:
    print(result["name"], "->", result["best"])
```

## Input limits

`text` is checked against four caps. All four are enforced server-side, and all
four come back as a 422 in the same pydantic `detail[]` shape, so a client
renders them identically.

| Cap | Limit | What it is for |
| --- | --- | --- |
| Words | 4096 | The model's *quality* window: `max_len` from its own config. Every GLiNER2.5 model declares 4096; a repo that declares none (only reachable via `GLINER_ALLOW_ANY_MODEL`) gets a conservative 1024. |
| Subword tokens | 8192 (`GLINER_MAX_TOKENS`) | The real cost bound, counted with the model's own tokenizer. |
| Characters | 32 768 | A cheap size guard, applied before anything is counted or tokenized. |
| Longest run without spaces | 256 characters | One pathological "word" that tokenizes into thousands of subwords. |

**Why a token cap on top of the word limit.** The word limit is the model's
quality window, *not* a cost bound, and a "word" is a wildly uneven unit: a page
of Chinese without spaces, or one giant URL, counts as a single word yet
tokenizes to thousands of subwords - which pinned the single inference worker
for 30-40 seconds. **Chinese/Japanese text without spaces is one word per
punctuation-delimited run; the token cap, not the word limit, is what protects
the server.**

A "word" is whatever gliner2's whitespace word splitter yields, which is not
what you would count by eye:

- URLs, emails and `@handles` stay whole: `https://example.com` is one word.
- A run of letters/digits/underscores with internal `-` or `_` is one word:
  `well-known` is one word.
- **Every other non-space character is a word of its own**, so punctuation
  counts. `Hello, world.` is **4 words** - `Hello`, `,`, `world`, `.`.

**The exact counts are the server's, and they are only made when you submit.**
There is no endpoint to ask in advance: `POST /api/decide` counts the passage,
rejects it with a 422 if it is over any cap, and reports what it counted in
`input` (`words` / `max_words` and `tokens` / `max_tokens`). The web UI's live
counter is a client-side approximation of the word rules (shown with a `≈`),
useful for staying roughly in bounds but not something to rely on at the
boundary - it does not block the button.

Long passages are genuinely slow: scoring is quadratic-ish in length. On this
machine (M-series, MPS) 30 words score in ~20 ms while 4000 words take **~7.6
seconds**; on CPU expect substantially worse. The word limit is not a
performance target - if you are anywhere near it, summarise first.

## Load and concurrency

**One process, one inference worker.** MPS is not safe for concurrent forward
passes, so a single daemon thread does every model load and every forward pass.
Request handlers do no inference themselves: they put a job on a bounded FIFO
queue and `await` its result, which is why a 7-second request never pins a
thread and `/health` keeps answering in about 1 ms while the worker is pinned.

**Opportunistic batching.** When the worker picks up a job, it drains the
jobs queued behind it that want the *same, already-loaded* model and runs them
as one batched forward pass (up to `GLINER_MAX_BATCH`). Scores are unaffected -
batched and unbatched confidences are bit-identical on this machine, which
`python loadtest.py --verify` checks against all five `samples/`.

**Two kinds of batching, and they are not the same.** The coalescing above puts
several *requests* in one forward pass as separate rows - padded to a common
length, unable to see each other, scores unchanged.
[`POST /api/decide/batch`](#post-apidecidebatch) puts several *decisions* about
one passage in one row, sharing a single encoder sequence - which is what makes
it fast and what makes the decisions influence each other. A batch request is
one job and one queue slot, not N; what bounds its cost is
`GLINER_MAX_DECISIONS` and `GLINER_MAX_TOTAL_CHOICES`, not the queue.

**Backpressure.** If `GLINER_MAX_QUEUE` jobs are already waiting, a new request
is refused immediately with `503` and `Retry-After: 2` rather than joining an
unbounded queue.

**`GLINER_QUEUE_TIMEOUT` counts only the time spent *waiting to be picked up*.**
A request that waits longer than that gives up with `503` and `Retry-After: 5`,
and is removed from the queue at once - so it never lingers as phantom
`queue.depth` and the worker never spends GPU time on an answer nobody will
read. Once the worker has started a request the clock stops and the request runs
to completion: a long passage is never abandoned part-way through a forward pass
that has already been paid for. A separate 600 s safety ceiling answers `504` if
the worker stops responding altogether (a wedged model load), so no caller hangs
forever.

`decide.py` retries a 503 up to three times on its own; the web UI shows
"Server is busy (N queued)".

### Configuration

| Variable | Default | When to change it |
| --- | --- | --- |
| `GLINER_MAX_QUEUE` | `64` | Lower it to fail fast under a stampede (clients get a 503 they can retry instead of a long wait); raise it only if your clients would rather wait than retry. |
| `GLINER_QUEUE_TIMEOUT` | `60` | Seconds a request may spend *waiting* for the worker; once inference starts it runs to completion. Lower it if your callers have their own short timeouts - the default is generous enough to sit behind a cold model load plus a couple of long passages. |
| `GLINER_MAX_BATCH` | `8` | Jobs coalesced into one forward pass. `1` disables batching. See the table below; 8 is where the gain flattens on this machine. |
| `GLINER_MAX_DECISIONS` | `16` | Decisions allowed in one `POST /api/decide/batch`. They share one forward pass, so this and the next row are what bound its cost. |
| `GLINER_MAX_TOTAL_CHOICES` | `120` | Choices allowed across all the decisions in one batch. The real cost bound: 120 choices on a full-length passage costs 5.7 s against the 4.3 s a single 20-choice request already costs, while `16 x 20 = 320` would cost 9.4 s. |
| `GLINER_ALLOW_ANY_MODEL` | unset | Set to `1` to accept arbitrary Hugging Face repo ids in `model`. Off by default: the worker is shared, so an unknown repo lets any client make the server download ~1 GB, and with the hub unreachable it pins the worker for minutes of HF retries. Aliases and the server's own `GLINER_MODEL` are always allowed. |

Batching is additionally capped by a padded-word budget (2048 word-slots,
`BATCH_WORD_BUDGET` in `server.py`). The collator pads every text in a batch to
the longest one and attention is quadratic in length, so mixing one long
passage into a batch of short ones is ruinous: measured here, a 1325-word
passage costs **0.43 s alone but 6.0 s** batched with seven 53-word passages.
The cap lets short requests batch freely while a long one runs on its own.

### Measuring it

```bash
python loadtest.py --concurrency 32 --requests 200 --mixed
python loadtest.py --verify        # batched == unbatched scores
```

`--mixed` rotates through all five `samples/` scenarios; without it every
request is `vendor_issue`. Other flags: `--url`, `--model`, `--timeout`. It
exits non-zero on any non-503 error, or if the same input ever gets two
different `best` answers. 503s are counted, not failed - they are the designed
backpressure.

**Batch size** (this machine, M-series/MPS, `base` model, mixed workload,
concurrency 32, 200 requests):

| `GLINER_MAX_BATCH` | req/s | p50 | p99 | mean batch | mean `inference_s` |
| --- | --- | --- | --- | --- | --- |
| 1 | 68.3 | 0.437 s | 0.608 s | 1.00 | 0.015 s |
| 4 | 92.6 | 0.321 s | 0.476 s | 4.00 | 0.043 s |
| **8** | **98.5** | **0.298 s** | **0.467 s** | 7.93 | 0.075 s |
| 16 | 96.7 | 0.288 s | 0.545 s | 15.56 | 0.144 s |

8 is the default: it captures the whole +44% throughput gain, and 16 buys
nothing measurable (run-to-run noise here is about 5%) while making p99 worse,
because a 16-job batch has to finish before any of its 16 callers hears back.

**Concurrency** (same machine, `GLINER_MAX_BATCH=8`, mixed workload, 200
requests):

| clients | req/s | p50 | p90 | p99 | mean `queued_s` | mean batch | 200s | 503s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 66.7 | 0.014 s | 0.015 s | 0.028 s | 0.000 s | 1.00 | 200 | 0 |
| 8 | 97.2 | 0.081 s | 0.085 s | 0.106 s | 0.020 s | 6.25 | 200 | 0 |
| 32 | 103.8 | 0.300 s | 0.301 s | 0.373 s | 0.207 s | 7.85 | 200 | 0 |
| 128 | 298.9 | 0.007 s | 0.513 s | 0.663 s | 0.292 s | 7.89 | 65 | 135 |

Throughput saturates around 100 req/s on short passages - that is one GPU's
worth, and no amount of client concurrency moves it. The 128-client row is the
queue doing its job: 64 wait, the rest are turned away in single-digit
milliseconds, which is why its "req/s" is mostly rejections and its p50 is
7 ms. Nothing 5xx other than those 503s, and no answer changed.

### Operational notes

- **Run one process.** `uvicorn --workers N` would load N copies of the model
  into N processes fighting over one GPU - slower than one worker, N times the
  memory. `start.sh` and the launchd service both run a single process.
- **A long request blocks everyone behind it.** A 4000-word passage occupies
  the worker for ~7-8 s on MPS, and FIFO means every queued request waits that
  out. That is what `GLINER_QUEUE_TIMEOUT` is for. Keep situations short;
  summarise before you submit.
- **A cold model load (~1-5 s) also blocks the queue**, once, and is never
  batched. Preload what you actually use via `GLINER_MODEL`.
- **Watch `/health`.** `queue.depth` persistently near `queue.max`, or a
  climbing `stats.rejected_busy`, means the box is out of headroom - there is
  no scaling knob left, so send less or get another machine.

## CLI

```bash
source .venv/bin/activate
python decide.py --text "..." -c "approve" -c "reject" -c "escalate"
python decide.py --text-file samples/vendor_issue.txt -c "keep the current vendor" -c "switch to a new vendor"
echo "some context" | python decide.py -c "approve" -c "reject"
```

Useful flags: `--model <alias|repo>`, `--device auto|mps|cpu|cuda`,
`--server-url <url>`, `--no-server`. Sample passages live in `samples/`.

`decide.py` automatically uses a running server on `127.0.0.1:8765` when there
is one, which makes calls near-instant because the model stays warm; otherwise
it loads the model locally for that one call (a few seconds).

Two things worth knowing about that split:

- **The server-reachable probe is the only fallback decision.** If the probe
  succeeds, the server owns the request: a 4xx/5xx from it (word limit
  exceeded, duplicate choices, unknown model) is printed as a readable message
  and the CLI exits 1. It does **not** quietly fall back to a local model load,
  which used to hide real errors and score your input with a *different* model
  than the one you asked the server for. A local load happens only when no
  server answered at all.
- **`--device` only applies to that local load.** A running server does its
  inference on its own `GLINER_DEVICE`; passing `--device cpu` to a call that
  goes through the server changes nothing. Use `--no-server --device cpu` to
  force it.

`--model` likewise defaults to *the server's* default: it is omitted from the
request body unless you pass it, so the CLI and the server never disagree about
which model ran. A local load with no `--model` uses `base`.

**Choices are validated the same way on both paths.** The local (`--no-server`)
path applies the server's own rules before it loads anything - non-empty after
stripping, 2 to 20 of them, no duplicates - so the CLI cannot quietly score an
input the API would have refused.

Output:

```
->  99.6%  [##############################]  switch to a new vendor
     0.9%  [------------------------------]  keep the current vendor
     0.1%  [------------------------------]  escalate to legal for breach of contract

Best choice: switch to a new vendor

Model: fastino/gliner2.5-base-v1  (device: mps, via server)
Input: 53 words of 4096 allowed, 53 subword tokens of 8192
Load time:       0.000s  (already warm)
Queued:          0.001s
Inference time:  0.048s
Wall time:       0.087s  (everything above + imports)
```

`Queued:` appears only when the server answered, and a `Batch:` line appears
when the request shared a forward pass with others - both are the queue
described in [Load and concurrency](#load-and-concurrency). Under load:

```
Queued:          0.310s
Inference time:  0.082s
Batch:          8
```

If the server is too busy to queue the request it answers 503; the CLI retries
up to three times, honouring `Retry-After`, printing one line per retry to
stderr before giving up:

```
Server busy (Server busy: 1 requests already queued. Retry shortly.); retrying in 2s (1/3)
```

A rejected request looks like this, and exits 1:

```
Error: server returned HTTP 422:
       choices must not contain duplicates
```

## Configuration

Environment variables, all read by `server.py`. `start.sh` and
`install-service.sh` only pass through what you set - the defaults below live in
one place, `server.py`, and nothing else repeats them. They keep the `GLINER_`
prefix after the rename to Minos: they configure the GLiNER2.5 backend, and
renaming them would silently ignore every environment and plist already set up:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GLINER_HOST` | `0.0.0.0` | Bind address. `0.0.0.0` = reachable on the LAN, `127.0.0.1` = this machine only. |
| `GLINER_PORT` | `8765` | TCP port. |
| `GLINER_MODEL` | `base` | Alias or full Hugging Face repo id, preloaded at startup and used when a request omits `model`. |
| `GLINER_DEVICE` | `auto` | `auto` picks MPS (Apple Silicon), then CUDA, then CPU. Force with `mps`, `cuda`, or `cpu`. |
| `GLINER_CORS_ORIGINS` | unset | Comma-separated list of browser origins allowed to call the API cross-origin, e.g. `http://localhost:3000,http://192.168.1.50:5173`. Unset or empty means no CORS middleware at all. |
| `GLINER_MAX_TOKENS` | `8192` | Subword tokens accepted in `text`, counted with the model's own tokenizer. The real cost bound - see [Input limits](#input-limits). |
| `GLINER_MAX_QUEUE` | `64` | Requests allowed to wait for the inference worker; overflow gets a 503. See [Load and concurrency](#load-and-concurrency). |
| `GLINER_QUEUE_TIMEOUT` | `60` | Seconds a request may *wait* for the worker before giving up with a 503; once inference starts it runs to completion. |
| `GLINER_MAX_BATCH` | `8` | Requests coalesced into one forward pass. `1` disables batching. |
| `GLINER_MAX_DECISIONS` | `16` | Decisions allowed in one `POST /api/decide/batch` body. |
| `GLINER_MAX_TOTAL_CHOICES` | `120` | Choices allowed across all the decisions in one batch; they share a single forward pass. |
| `GLINER_ALLOW_ANY_MODEL` | unset | `1` accepts arbitrary Hugging Face repo ids in `model`. Off by default so clients cannot make the shared worker download a random ~1 GB model. |

Example: `GLINER_PORT=9000 GLINER_MODEL=small ./start.sh`

`start.sh` passes the whole environment through, so all of these work with it.
**`install-service.sh` writes every `GLINER_*` variable that is set in the
environment you run it in into the plist**, so
`GLINER_PORT=9000 GLINER_MAX_BATCH=4 ./install-service.sh` configures the
service exactly the way the same prefix configures `./start.sh`. Variables you
do not set are simply left out, and `server.py`'s defaults apply. Re-run the
script to change a setting: it rewrites the plist and reloads the service.

**CORS is opt-in on purpose.** The API has no authentication, so a wildcard
policy would let *any* web page the user happens to have open drive this server
from inside their browser, on their LAN, with their network position. The
bundled web UI is served from this same origin and needs no CORS whatsoever;
only add origins when you are building a separate front end against the API,
and list them explicitly.

## Models

| Alias | Hugging Face repo | Size | Languages | Notes |
| --- | --- | --- | --- | --- |
| `multi` | `fastino/gliner2.5-multi-v1` | 0.3B | multilingual | GLiNER2.5 |
| `base` | `fastino/gliner2.5-base-v1` | 0.2B | multilingual | GLiNER2.5, the default |
| `small` | `fastino/gliner2.5-small-v1` | 74M | multilingual | GLiNER2.5, fastest |

Request one per call with `"model": "small"` in the JSON body, `--model small`
on the CLI, or the dropdown in the web UI; a full repo id such as
`"model": "fastino/gliner2.5-multi-v1"` works too. Only `GLINER_MODEL` is loaded
at startup - any other model is downloaded and loaded on its first request (that
request pays the cost, reported as `timing.load_s`) and then stays cached in
memory until the server restarts. Weights are cached on disk in
`~/.cache/huggingface`, shared across all copies of this project.

Each model also has its own maximum input length - 4096 words for the GLiNER2.5
models. `GET /api/models` reports it as `max_words`; see
[Input limits](#input-limits).

## Troubleshooting

**First run is slow, or fails offline.** `setup.sh` downloads PyTorch (~130 MB)
and the model (~400 MB) from the internet. Both are cached afterwards
(`~/.cache/huggingface` for weights), so later runs and other checkouts are
fast. There is no offline install path - do the first `./setup.sh` while
connected.

**`Address already in use` / `[Errno 48]`.** Something is already on the port:

```bash
lsof -i :8765 -sTCP:LISTEN      # shows the PID
kill <pid>                      # or: GLINER_PORT=8766 ./start.sh
```

If you installed the launchd service, it is probably that - `./uninstall-service.sh`.

**Intel Mac / no GPU.** `GLINER_DEVICE=auto` falls back to CPU automatically.
Everything works, inference is just noticeably slower (a second or so instead of
~100 ms). Nothing to configure at runtime - but see the Python 3.12 requirement
under "Deploy on another MacBook": that one *does* need attention before
`setup.sh` will succeed.

**Python too old, or too new.** macOS ships Python 3.9; gliner2 needs 3.10+.
`setup.sh` tries `python3.12` first, then 3.13, 3.14, 3.11, 3.10, `python3`, and
exits with instructions if it finds none: `brew install python@3.12`, then
re-run `./setup.sh`. The 3.12-first order matters only on Intel, where no newer
Python has an installable torch (see above); on Apple Silicon any of them work
and an existing `.venv` is reused as-is.

**`/health` says `"status":"degraded"`.** The server is running but its startup
preload failed; `error` in the same response says why - usually a typo in
`GLINER_MODEL`, or a cold weights cache with no network. It deliberately stays
up rather than exiting into a launchd restart loop, so check `status`, not just
that the port answers. Fix `GLINER_MODEL` (or get online) and restart:
`./uninstall-service.sh && GLINER_MODEL=base ./install-service.sh`.

**Where the service logs are.** `logs/server.log` in the project directory -
both stdout and stderr:

```bash
tail -f logs/server.log
launchctl print gui/$(id -u)/com.minos.server   # state, pid, env
```

**Switching the default model.** Set `GLINER_MODEL` before starting:
`GLINER_MODEL=small ./start.sh`, or for the background service
`GLINER_MODEL=small ./install-service.sh` (re-running it updates the installed
plist).

**Results look wrong.** Rephrase the choices - the model matches label wording
against the passage. Also try another model size; they disagree often. See the
caveats at the top.
