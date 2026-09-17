"""Shared model config and inference helpers used by decide.py and server.py.

Nothing here imports torch or gliner2 at module scope: decide.py must stay a
~100ms command when a server.py is doing the inference, and importing torch
alone costs ~1.2s. Every such import lives inside the function that needs it.
"""

import os
import re
from contextlib import contextmanager
from itertools import islice

MODEL_ALIASES = {
    "multi": "fastino/gliner2.5-multi-v1",   # 0.3B, multilingual, GLiNER2.5
    "base": "fastino/gliner2.5-base-v1",     # 0.2B, multilingual, GLiNER2.5
    "small": "fastino/gliner2.5-small-v1",   # 74M,  multilingual, GLiNER2.5
    "large": "fastino/gliner2-large-v1",     # 0.5B, English-only, older GLiNER2 architecture
}

DEFAULT_MODEL = "base"
DEFAULT_DEVICE = "auto"

# Where server.py binds by default, and where decide.py looks for it. The bind
# host is a wildcard (LAN-visible); the client host is loopback.
DEFAULT_BIND_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
SERVER_HOST = "127.0.0.1"

# Word budget for models whose config.json carries no `max_len`. The only such
# model we alias is fastino/gliner2-large-v1 (older GLiNER2), whose published
# limit is 2048 *subword* tokens; 1024 words is a deliberately conservative
# stand-in, since a word usually costs more than one subword token.
FALLBACK_MAX_WORDS = 1024

# --- input size limits -------------------------------------------------------
#
# Three layers, because no single one of them bounds the cost of a forward pass:
#
#   MAX_TEXT_CHARS   a size guard on the raw payload. 4096 English words (the
#                    GLiNER2.5 budget) is ~25-30k characters, so 32k leaves the
#                    legitimate worst case room without letting a client post a
#                    100k-character blob.
#   MAX_TOKEN_CHARS  the word budget is not a cost bound on its own: the word
#                    splitter keeps `\w+` runs and URLs whole, so "x" * 30000 or
#                    a 30k-character URL is *one* word and used to sail through
#                    at 40s of inference. Cap the length of a single
#                    whitespace-free run instead.
#   MAX_TOKENS       the real bound. Cost is driven by *subword* tokens, and CJK
#                    tokenizes to roughly one subword per character, so 32k
#                    characters of Chinese is ~32k subwords (minutes of
#                    attention) while counting only a few hundred words. Only a
#                    subword count bounds the work, and it needs the model's own
#                    tokenizer -- so it is enforced wherever the model is loaded.
MAX_TEXT_CHARS = 32_768
MAX_TOKEN_CHARS = 256
MAX_TOKENS = int(os.environ.get("GLINER_MAX_TOKENS", "8192"))

# The character class of gliner2's word-splitter email alternative, the one that
# makes its regex backtrack quadratically. A run of these longer than
# MAX_TOKEN_CHARS is rejected before the splitter runs; see check_input.
_BACKTRACKING_RUN = re.compile(r"[A-Za-z0-9._%+\-]+")

MIN_CHOICES = 2
MAX_CHOICES = 20


class InputTooLong(ValueError):
    """A passage exceeds one of the input size limits.

    `kind` is one of "chars", "token_chars", "words", "tokens"; `message` is the
    client-facing sentence (server.py wraps it in the 422 list shape that
    pydantic uses, decide.py prints it)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


def clean_choices(choices: list[str]) -> list[str]:
    """Strip and validate a choice list, or raise ValueError.

    One implementation for both entry points: server.py's pydantic validator
    calls it, and decide.py's local path calls it too -- which used to validate
    less (it accepted empty labels and any number of choices)."""
    cleaned = [c.strip() for c in choices]
    if any(not c for c in cleaned):
        raise ValueError("choices must not contain empty strings")
    if not MIN_CHOICES <= len(cleaned) <= MAX_CHOICES:
        raise ValueError(f"choices must contain between {MIN_CHOICES} and {MAX_CHOICES} items")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("choices must not contain duplicates")
    return cleaned


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def resolve_device(name: str) -> str:
    if name != "auto":
        return name

    # Imported here rather than at module scope: torch costs ~1.2s to import,
    # which decide.py must not pay when a running server.py does the inference.
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@contextmanager
def _hub_offline():
    """Force huggingface_hub (and therefore transformers) to answer from the
    local cache only, for the duration of the block.

    Callers serialize model loads, so the brief global flip is safe here."""
    import huggingface_hub.constants as hub_constants
    import transformers.utils.hub as tf_hub
    from huggingface_hub.utils._http import reset_sessions

    previous_hub = hub_constants.HF_HUB_OFFLINE
    # transformers snapshots the hub's offline flag into a module global at
    # import time (transformers/utils/hub.py: _is_offline_mode = ...), and
    # while that copy is False its tokenizer loader calls model_info() over the
    # network unconditionally, ignoring local_files_only entirely
    # (tokenization_utils_base.py: _patch_mistral_regex -> is_base_mistral).
    # Patch the snapshot too, or the cached-weights path still goes online.
    previous_tf = tf_hub._is_offline_mode

    hub_constants.HF_HUB_OFFLINE = True
    tf_hub._is_offline_mode = True
    # The try: opens immediately after the flips, before the first
    # reset_sessions(): with it opening afterwards, a raising reset_sessions()
    # skipped the finally and left the whole process wedged offline for good.
    try:
        # huggingface_hub caches one requests.Session per thread and decides at
        # *session-creation* time whether to mount its OfflineAdapter, so the
        # flag alone does not affect an already-built session -- and a session
        # built inside this block would stay offline forever afterwards,
        # breaking the online fallback. Rebuild on both edges.
        reset_sessions()
        yield
    finally:
        hub_constants.HF_HUB_OFFLINE = previous_hub
        tf_hub._is_offline_mode = previous_tf
        reset_sessions()


def load_model(model_name: str, device: str):
    # Imported here rather than at module scope because it pulls in
    # torch/transformers (~1.2s), which decide.py never needs when a running
    # server.py is doing the inference.
    from gliner2 import AutoExtractor

    device = resolve_device(device)

    # Try the local cache first. When the HF Hub is unreachable (office VPN,
    # DNS hole, hub outage) an online load retries every checkpoint file five
    # times with backoff, which wedges startup for many minutes even though the
    # weights are sitting on disk.
    #
    # local_files_only alone is NOT enough: gliner2 forwards it only to its own
    # checkpoint downloads (models/loading.py: HUB_LOAD_OPTIONS -> hf_hub_download
    # in checkpoint_file), while the encoder and tokenizer are fetched separately
    # through transformers, which never sees the flag -- so the load still
    # blocked on encoder_config/config.json. Flipping huggingface_hub's offline
    # constant covers both: every hub request checks it at call time
    # (huggingface_hub/utils/_http.py) and fails instantly instead of retrying.
    try:
        with _hub_offline():
            return AutoExtractor.from_pretrained(
                model_name, map_location=device, local_files_only=True
            )
    except Exception:
        # Not cached (or a partial cache) -> fall back to the online load.
        return AutoExtractor.from_pretrained(model_name, map_location=device)


def count_words(text: str, stop_after: int | None = None) -> int:
    """Count word tokens exactly the way gliner2's processor does, checking
    each token's length in the same pass.

    gliner2 splits text into *word* tokens before subword tokenization, and the
    models' `max_len` is expressed in those units. The splitter keeps URLs,
    emails and @handles whole, keeps `\\w+(?:[-_]\\w+)*` runs whole, and makes
    every other non-space character its own token -- so "Hello, world." is 4.

    `stop_after` caps how many tokens are pulled from the splitter, so callers
    that only need "is this over the limit?" pass limit + 1 and stop there. That
    matters because the splitter's regex backtracks quadratically on long
    punctuation runs (its email alternative starts `[a-z0-9._%+-]+`, so every
    position in a run of dashes rescans the rest of the run): counting all of
    "-" * 100000 took 22 seconds and used to freeze /health with it.

    Raises InputTooLong("token_chars") when a single token the splitter yields
    is longer than MAX_TOKEN_CHARS -- checked here because it is the same pass,
    and because such a token is one cheap-looking "word" (a 10000-character URL,
    a run of CJK) that costs thousands of subword tokens.
    """
    # Lazy: anything under gliner2 drags torch in at import time.
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter

    tokens = WhitespaceTokenSplitter()(text, lower=False)
    if stop_after is not None:
        tokens = islice(tokens, stop_after)
    count = 0
    for _token, start, end in tokens:
        if end - start > MAX_TOKEN_CHARS:
            raise InputTooLong(
                "token_chars",
                f"text contains a {end - start}-character run without spaces; "
                f"the longest allowed is {MAX_TOKEN_CHARS}",
            )
        count += 1
    return count


def tokenizer_for(model):
    """The subword tokenizer of a loaded model, or None.

    gliner2 hands the tokenizer it loaded (models/boundary/model.py:
    load_extractor_tokenizer) to a SchemaTransformer, which keeps it as
    `.tokenizer` -- so it lives at model.processor.tokenizer. A plain
    model.tokenizer is checked first in case a future architecture exposes it
    directly."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(getattr(model, "processor", None), "tokenizer", None)
    return tokenizer


def check_input(model, text: str, repo: str = "this model") -> dict:
    """Every input size limit, in one place. Returns {words, max_words, tokens,
    max_tokens}; raises InputTooLong otherwise.

    Cheapest check first, and the subword count -- the only one that actually
    bounds inference cost -- last, since it needs the loaded model's tokenizer.
    `tokens` is None only when the model exposes no tokenizer at all.

    Not cheap enough for an event loop (the word scan is O(text) at best): call
    it on a worker thread or in the inference worker.
    """
    if len(text) > MAX_TEXT_CHARS:
        raise InputTooLong(
            "chars",
            f"text is {len(text)} characters; at most {MAX_TEXT_CHARS} are accepted",
        )

    # Cheap pre-check, before the splitter sees the text at all. It exists for
    # cost, not just correctness: the splitter's email alternative
    # (`[a-z0-9._%+-]+` then a required `@`) rescans such a run from every
    # position inside it, so "-" * 32768 costs 0.6s of CPU -- which the GIL
    # turns into ~75ms of added latency for every other request in flight, even
    # with the scan on a worker thread. _BACKTRACKING_RUN is exactly that
    # character class, so this both rejects the offender in microseconds and
    # bounds the backtracking of every text that gets past it. It is
    # deliberately ASCII-only: a Chinese passage has no spaces anywhere, so a
    # "longest run without whitespace" check would reject every one of them.
    longest_run = max((m.end() - m.start() for m in _BACKTRACKING_RUN.finditer(text)), default=0)
    if longest_run > MAX_TOKEN_CHARS:
        raise InputTooLong(
            "token_chars",
            f"text contains a {longest_run}-character run without spaces; "
            f"the longest allowed is {MAX_TOKEN_CHARS}",
        )

    max_words = max_words_for(model)
    # limit + 1 is all that is needed to know we are over, and stopping there
    # keeps a pathological passage from costing more than the rejection is worth.
    # count_words re-checks the length of each token the splitter actually
    # yields. That is the authoritative check -- it is what catches a 10000
    # character run of Chinese, which the ASCII pre-check above cannot see.
    words = count_words(text, stop_after=max_words + 1)
    if words > max_words:
        raise InputTooLong(
            "words",
            f"text is more than {max_words} words; {repo} accepts at most {max_words} "
            "words (counted with gliner2's word splitter, where punctuation marks "
            "count as words)",
        )

    tokens = None
    tokenizer = tokenizer_for(model)
    if tokenizer is not None:
        tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if tokens > MAX_TOKENS:
            raise InputTooLong(
                "tokens",
                f"text is {tokens} subword tokens for {repo}; at most {MAX_TOKENS} "
                "are accepted (GLINER_MAX_TOKENS)",
            )

    return {"words": words, "max_words": max_words,
            "tokens": tokens, "max_tokens": MAX_TOKENS}


def max_words_for(model) -> int:
    """The word budget of an already-loaded model."""
    max_len = getattr(getattr(model, "config", None), "max_len", None)
    return int(max_len) if max_len else FALLBACK_MAX_WORDS


def cached_max_words(repo: str) -> int | None:
    """The word budget of a model we have *not* loaded, read from the cached
    config.json. Returns None when the repo is not in the local HF cache (or
    the id is unusable). Never touches the network."""
    # Lazy so that neither huggingface_hub nor json is paid for on the CLI's
    # fast path.
    import json

    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo, "config.json", local_files_only=True)
        with open(path) as f:
            max_len = json.load(f).get("max_len")
    except Exception:
        return None
    return int(max_len) if max_len else FALLBACK_MAX_WORDS


def decide_batch(model, items: list[tuple[str, list[str]]],
                 task_label: str = "decision", max_len: int | None = None):
    """Score several (text, choices) pairs in one forward pass.

    batch_extract takes one schema *per text* and formats each result with that
    text's own classification task, so requests with different choice lists
    batch together. Returns one sorted list per item, in input order.
    """
    texts = [text for text, _ in items]
    schemas = [
        model.create_schema().classification(
            task_label, choices, multi_label=True, cls_threshold=0.0
        )
        for _, choices in items
    ]
    # max_len is a hard backstop: inference does no truncation of its own
    # (gliner2/inference/runtime.py builds an uncapped collator when max_len is
    # None), so without it an over-long passage reaches the encoder in full.
    # batch_size == len(texts) keeps the whole batch in a single forward pass.
    results = model.batch_extract(
        texts, schemas, batch_size=len(items), threshold=0.0, num_workers=0,
        format_results=True, include_confidence=True, max_len=max_len,
    )
    return [
        sorted(r.get(task_label, []), key=lambda x: x["confidence"], reverse=True)
        for r in results
    ]


def decide(model, text: str, choices: list[str], task_label: str = "decision",
           max_len: int | None = None):
    return decide_batch(model, [(text, choices)], task_label=task_label, max_len=max_len)[0]


def bar(confidence: float, width: int = 30) -> str:
    filled = round(confidence * width)
    return "#" * filled + "-" * (width - filled)
