#!/usr/bin/env python
"""Decision-making helper: given a text passage and 2+ choices, rank them by
how well the model thinks each choice fits the situation described.

Usage:
    python decide.py --text "..." -c "choice A" -c "choice B" -c "choice C"
    python decide.py --text-file notes.txt -c "approve" -c "reject" -c "escalate"
    echo "some context" | python decide.py -c "approve" -c "reject" -c "escalate"

If server.py is running (see that file), this automatically uses it so the
model stays loaded between calls instead of reloading every time. Otherwise
it falls back to loading the model locally for this one call.

The fallback is decided by a fast /health probe *before* the request. Once a
server answers that probe it owns the request: an error from it is reported and
exits non-zero, never retried locally behind your back.
"""

import time

# Captured before the remaining imports so the reported wall time includes
# them; everything except Python's own interpreter startup is accounted for.
_script_start = time.perf_counter()

import argparse
import json
import sys
import urllib.error
import urllib.request

from common import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    MODEL_ALIASES,
    SERVER_HOST,
    InputTooLong,
    bar,
    check_input,
    clean_choices,
    decide,
    load_model,
    resolve_device,
    resolve_model_name,
)

SERVER_URL = f"http://{SERVER_HOST}:{DEFAULT_PORT}"

# Long enough to cover a cold model load on the server (~5s) plus inference on
# a long passage. The *probe* is what stays fast.
SERVER_TIMEOUT = 120.0
PROBE_TIMEOUT = 0.3

# A 503 from the server means its inference queue is full or we waited too
# long -- both transient, both carrying Retry-After.
MAX_BUSY_RETRIES = 3
MAX_RETRY_SLEEP = 5.0


def die(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def server_is_up(server_url: str, timeout: float = PROBE_TIMEOUT) -> bool:
    """Cheap liveness probe. This -- and only this -- decides whether we use the
    server. Anything that goes wrong *after* a successful probe is a real error
    to report, never a reason to silently spend 5s loading the model locally
    and answering from an unvalidated second code path."""
    try:
        with urllib.request.urlopen(f"{server_url}/health", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def _format_detail(detail) -> str:
    """Render FastAPI's `detail` readably. Validation errors arrive as a list of
    objects; pydantic prefixes their messages with "Value error, "."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        msgs = []
        for item in detail:
            if isinstance(item, dict) and "msg" in item:
                msgs.append(str(item["msg"]).removeprefix("Value error, "))
            else:
                msgs.append(str(item))
        return "\n       ".join(msgs)
    return json.dumps(detail)


def run_server(server_url: str, text: str, choices: list[str], model: str | None):
    """Score via the running server. Exits non-zero on any server error."""
    payload = {"text": text, "choices": choices}
    if model is not None:
        # Omitted when unset, so the server's own GLINER_MODEL stays in charge
        # rather than the CLI forcing a second model into memory.
        payload["model"] = model
    req = urllib.request.Request(
        f"{server_url}/api/decide", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    for attempt in range(1, MAX_BUSY_RETRIES + 2):
        try:
            with urllib.request.urlopen(req, timeout=SERVER_TIMEOUT) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                detail = json.loads(body).get("detail", body.decode(errors="replace"))
            except (ValueError, UnicodeDecodeError):
                detail = body.decode(errors="replace")
            retry_after = e.headers.get("Retry-After") if e.code == 503 else None
            if retry_after is None or attempt > MAX_BUSY_RETRIES:
                die(f"server returned HTTP {e.code}:\n       {_format_detail(detail)}")
            try:
                delay = min(float(retry_after), MAX_RETRY_SLEEP)
            except ValueError:
                delay = 1.0
            print(f"Server busy ({_format_detail(detail)}); retrying in {delay:g}s "
                  f"({attempt}/{MAX_BUSY_RETRIES})", file=sys.stderr)
            time.sleep(delay)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            die(f"request to {server_url} failed: {e}")

    timing = data["timing"]
    info = data.get("input") or {}
    return (data["ranked"], data["model"], data["device"], timing["load_s"],
            timing["inference_s"], info, timing.get("queued_s"),
            timing.get("batch_size", 1))


def validate_locally(text: str, choices: list[str]) -> list[str]:
    """The checks the server would have made, for the no-server path. Returns
    the cleaned choices, which is what the model must actually be given.

    Everything except the size limits, which need the loaded model: those run in
    run_local(). Shares common.clean_choices with the server's validator, so
    this path no longer accepts (and scores) empty labels or 21 choices."""
    if not text.strip():
        die("text must not be empty")
    try:
        return clean_choices(choices)
    except ValueError as e:
        die(str(e))


def run_local(text: str, choices: list[str], model: str | None, device: str):
    model_name = resolve_model_name(model or DEFAULT_MODEL)
    text = text.strip()  # the server's validator strips too; keep the paths identical

    load_start = time.perf_counter()
    device = resolve_device(device)
    loaded = load_model(model_name, device)
    load_time = time.perf_counter() - load_start

    # Same limits, same messages as the server: this needs the loaded model for
    # its word budget and its tokenizer.
    try:
        info = check_input(loaded, text, model_name)
    except InputTooLong as e:
        die(e.message)

    infer_start = time.perf_counter()
    ranked = decide(loaded, text, choices, max_len=info["max_words"])
    infer_time = time.perf_counter() - infer_start

    return ranked, model_name, device, load_time, infer_time, info, None, 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", help="The text passage to decide on")
    parser.add_argument("--text-file", help="Read the text passage from a file")
    parser.add_argument(
        "-c", "--choice", dest="choices", action="append", required=True,
        help="A possible choice/decision. Pass at least twice (e.g. -c A -c B -c C)",
    )
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE, choices=["auto", "mps", "cpu", "cuda"],
        help="Inference device; only used when loading locally (no server). A running "
             "server uses its own GLINER_DEVICE.",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"Model alias ({', '.join(MODEL_ALIASES)}) or a full HF repo id. Unset, a running "
             f"server uses its own default model, and a local load uses '{DEFAULT_MODEL}'.",
    )
    parser.add_argument("--server-url", default=SERVER_URL, help="URL of a running server.py instance")
    parser.add_argument("--no-server", action="store_true", help="Skip the server and always load the model locally")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.text_file:
        with open(args.text_file, "r") as f:
            text = f.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("Provide --text, --text-file, or pipe text via stdin")

    if len(args.choices) < 2:
        parser.error("Provide at least two --choice/-c options")

    # The probe is the only fallback decision. If it succeeds, the server owns
    # this request including its errors; if it fails, we validate and run here.
    via_server = not args.no_server and server_is_up(args.server_url)

    if via_server:
        result = run_server(args.server_url, text, args.choices, args.model)
    else:
        choices = validate_locally(text, args.choices)
        result = run_local(text, choices, args.model, args.device)

    ranked, model_name, device, load_time, infer_time, info, queued, batch_size = result
    words, max_words = info.get("words"), info.get("max_words")
    tokens, max_tokens = info.get("tokens"), info.get("max_tokens")

    print()
    for i, item in enumerate(ranked):
        marker = "-> " if i == 0 else "   "
        print(f"{marker}{item['confidence']*100:5.1f}%  [{bar(item['confidence'])}]  {item['label']}")
    print()
    print(f"Best choice: {ranked[0]['label']}")
    print()
    print(f"Model: {model_name}  (device: {device}, via {'server' if via_server else 'local load'})")
    if words is not None:
        line = f"Input: {words} words" + (f" of {max_words} allowed" if max_words else "")
        if tokens is not None:
            line += f", {tokens} subword tokens" + (f" of {max_tokens}" if max_tokens else "")
        print(line)
    print(f"Load time:      {load_time:6.3f}s" + ("  (already warm)" if via_server and load_time == 0 else ""))
    if queued is not None:
        print(f"Queued:         {queued:6.3f}s")
    print(f"Inference time: {infer_time:6.3f}s")
    if batch_size > 1:
        print(f"Batch:          {batch_size}")
    print(f"Wall time:      {time.perf_counter() - _script_start:6.3f}s  (everything above + imports)")
    if not via_server:
        print()
        print("Tip: run ./start.sh (or: python server.py) in another terminal to keep the model loaded between calls.")


if __name__ == "__main__":
    main()
