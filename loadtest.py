#!/usr/bin/env python
"""Load generator for server.py's inference queue. Standard library only.

    python loadtest.py --concurrency 32 --requests 200 --mixed
    python loadtest.py --verify        # batched vs single-item equivalence

Every request is a real POST /api/decide. 503s are counted, not treated as
failures (they are the queue's designed backpressure); anything else non-200
fails the run, as does a `best` that changes between two responses for the
same input.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SAMPLES = Path(__file__).parent / "samples"

# name -> choices; the passage comes from samples/<name>.txt
SCENARIOS = {
    "vendor_issue": [
        "keep the current vendor",
        "switch to a new vendor",
        "escalate to legal for breach of contract",
    ],
    "job_offer": [
        "take offer A",
        "take offer B",
        "stay at current job",
    ],
    "incident_response": [
        "take the server offline immediately",
        "monitor and wait for full triage",
        "isolate network access but keep the batch job running",
    ],
    "investment_decision": [
        "lead a new investment round",
        "pass on investing further",
        "offer a smaller bridge loan only",
    ],
    "hiring_candidate": [
        "extend an offer",
        "reject the candidate",
        "schedule an additional interview round",
    ],
}


def load_scenarios(mixed: bool) -> list[tuple[str, str, list[str]]]:
    names = list(SCENARIOS) if mixed else ["vendor_issue"]
    out = []
    for name in names:
        out.append((name, (SAMPLES / f"{name}.txt").read_text().strip(), SCENARIOS[name]))
    return out


def post(url: str, payload: dict, timeout: float):
    """Return (status, parsed_body_or_None, elapsed_seconds)."""
    req = urllib.request.Request(
        f"{url}/api/decide", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read()), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        # A JSON body need not be an object (a bare string or list is valid
        # JSON); everything below, starting with the line after this one,
        # assumes a dict.
        if not isinstance(parsed, dict):
            parsed = {"detail": body.decode(errors="replace")}
        parsed["_retry_after"] = e.headers.get("Retry-After")
        return e.code, parsed, time.perf_counter() - t0
    except Exception as e:  # connection refused, timeout, ...
        return 0, {"detail": f"{type(e).__name__}: {e}"}, time.perf_counter() - t0


def pctl(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def run_load(args) -> int:
    scenarios = load_scenarios(args.mixed)
    results = []

    def one(i):
        name, text, choices = scenarios[i % len(scenarios)]
        payload = {"text": text, "choices": choices}
        if args.model:
            payload["model"] = args.model
        status, body, elapsed = post(args.url, payload, args.timeout)
        return name, status, body, elapsed

    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for r in pool.map(one, range(args.requests)):
            results.append(r)
    wall = time.perf_counter() - wall0

    ok = [r for r in results if r[1] == 200]
    busy = [r for r in results if r[1] == 503]
    other = [r for r in results if r[1] not in (200, 503)]

    latencies = [r[3] for r in results]
    ok_lat = [r[3] for r in ok]
    queued = [r[2]["timing"]["queued_s"] for r in ok]
    infer = [r[2]["timing"]["inference_s"] for r in ok]
    batch = [r[2]["timing"]["batch_size"] for r in ok]

    first_best, inconsistent = {}, []
    for name, _status, body, _elapsed in ok:
        best = body["best"]
        if name not in first_best:
            first_best[name] = best
        elif first_best[name] != best:
            inconsistent.append((name, first_best[name], best))

    print()
    print(f"url              {args.url}")
    print(f"workload         {'mixed (5 scenarios)' if args.mixed else 'vendor_issue only'}"
          f"   model={args.model or '(server default)'}")
    print(f"concurrency      {args.concurrency}   requests={args.requests}")
    print("-" * 62)
    print(f"wall             {wall:8.2f} s")
    print(f"throughput       {len(results) / wall:8.2f} req/s   "
          f"(200s only: {len(ok) / wall:.2f})")
    print("-" * 62)
    print(f"latency p50      {pctl(latencies, 0.50):8.3f} s")
    print(f"latency p90      {pctl(latencies, 0.90):8.3f} s")
    print(f"latency p99      {pctl(latencies, 0.99):8.3f} s")
    print(f"latency max      {max(latencies):8.3f} s")
    if ok_lat and busy:
        print(f"latency p99 (200s only)  {pctl(ok_lat, 0.99):.3f} s")
    print("-" * 62)
    print(f"status 200       {len(ok):8d}")
    print(f"status 503       {len(busy):8d}   "
          f"retry-after={sorted({r[2].get('_retry_after') for r in busy}) if busy else '-'}")
    print(f"status other     {len(other):8d}")
    print("-" * 62)
    print(f"queued_s mean    {statistics.fmean(queued) if queued else 0:8.3f} s   "
          f"max={max(queued) if queued else 0:.3f} s")
    print(f"inference_s mean {statistics.fmean(infer) if infer else 0:8.3f} s")
    print(f"batch_size mean  {statistics.fmean(batch) if batch else 0:8.2f}     "
          f"max={max(batch) if batch else 0}")
    infos = [r[2].get("input") or {} for r in ok]
    words = [i["words"] for i in infos if i.get("words") is not None]
    tokens = [i["tokens"] for i in infos if i.get("tokens") is not None]
    if words:
        line = f"words mean       {statistics.fmean(words):8.1f}     max={max(words)}"
        if tokens:
            max_tokens = next((i.get("max_tokens") for i in infos if i.get("max_tokens")), None)
            line += (f"   |   tokens mean {statistics.fmean(tokens):.1f}  max={max(tokens)}"
                     + (f" of {max_tokens}" if max_tokens else ""))
        print(line)
    print(f"consistency      {'OK' if not inconsistent else 'FAILED'}   "
          f"({len(first_best)} distinct inputs)")

    failed = False
    for name, expected, got in inconsistent[:5]:
        print(f"  MISMATCH {name}: first said {expected!r}, later {got!r}")
        failed = True
    for _name, status, body, _elapsed in other[:5]:
        print(f"  ERROR HTTP {status}: {str(body.get('detail'))[:160]}")
        failed = True
    print()
    return 1 if failed else 0


def verify() -> int:
    """Score each sample alone via decide(), then all of them together via
    decide_batch(), and compare. Runs in-process: needs the venv."""
    from common import (DEFAULT_MODEL, DEFAULT_TASK_LABEL, decide, decide_batch, load_model,
                        max_words_for, resolve_device, resolve_model_name)

    repo = resolve_model_name(DEFAULT_MODEL)
    device = resolve_device("auto")
    print(f"Loading {repo} on {device} ...")
    model = load_model(repo, device)
    limit = max_words_for(model)

    scenarios = load_scenarios(mixed=True)
    # One decision per passage: this checks the axis that must NOT change scores
    # (separate texts coalesced into one forward pass). The other axis -- several
    # decisions on one passage -- changes them by design; see decide_batch().
    items = [(text, [(DEFAULT_TASK_LABEL, choices)]) for _name, text, choices in scenarios]

    single = [decide(model, text, tasks[0][1], max_len=limit) for text, tasks in items]
    batched = [scored[DEFAULT_TASK_LABEL] for scored in decide_batch(model, items, max_len=limit)]

    worst = 0.0
    failed = False
    print()
    print(f"{'scenario':<22} {'choice':<52} {'single':>9} {'batched':>9} {'delta':>10}")
    print("-" * 106)
    for (name, _t, _c), s_rank, b_rank in zip(scenarios, single, batched):
        s_map = {r["label"]: r["confidence"] for r in s_rank}
        b_map = {r["label"]: r["confidence"] for r in b_rank}
        for label in s_map:
            delta = abs(s_map[label] - b_map.get(label, float("nan")))
            worst = max(worst, delta)
            print(f"{name:<22} {label:<52} {s_map[label]:9.6f} {b_map.get(label, 0):9.6f} {delta:10.2e}")
        if s_rank[0]["label"] != b_rank[0]["label"]:
            print(f"  BEST MISMATCH {name}: {s_rank[0]['label']!r} vs {b_rank[0]['label']!r}")
            failed = True
    print("-" * 106)
    print(f"max |single - batched| confidence delta: {worst:.2e}  (tolerance 1e-4)")
    if worst > 1e-4:
        print("FAILED: batched confidences differ beyond tolerance")
        failed = True
    print("PASS" if not failed else "FAIL")
    print()
    return 1 if failed else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--model", default=None, help="Omit to use the server's default model")
    p.add_argument("--mixed", action="store_true",
                   help="Rotate through all 5 sample scenarios instead of vendor_issue only")
    p.add_argument("--timeout", type=float, default=300.0, help="Per-request client timeout")
    p.add_argument("--verify", action="store_true",
                   help="Check decide() vs decide_batch() equivalence in-process and exit")
    args = p.parse_args()
    return verify() if args.verify else run_load(args)


if __name__ == "__main__":
    sys.exit(main())
