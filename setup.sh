#!/bin/bash
# Creates .venv, installs requirements.txt, pre-downloads the default model.
# Safe to re-run: existing venv and cached model weights are reused.
set -euo pipefail

cd "$(dirname "$0")"

PROJECT_DIR="$(pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
MODEL="${GLINER_MODEL:-base}"

# 3.12 first on purpose: gliner2 needs torch >= 2.1, and the last x86_64 macOS
# torch wheel is 2.2.2, which ships cp312 only. On an Intel Mac a newer Python
# has no installable torch at all, so prefer the one that works everywhere.
find_python() {
    for candidate in python3.12 python3.13 python3.14 python3.11 python3.10 python3; do
        local exe
        exe="$(command -v "$candidate" 2>/dev/null)" || continue
        if "$exe" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            echo "$exe"
            return 0
        fi
    done
    return 1
}

echo "==> Looking for Python 3.10+"
if [ -x "$VENV_DIR/bin/python" ]; then
    echo "    Reusing existing venv: $VENV_DIR"
else
    if ! PYTHON="$(find_python)"; then
        echo "ERROR: no Python 3.10 or newer found in PATH." >&2
        echo "       gliner2 requires Python >= 3.10 (macOS ships 3.9)." >&2
        echo "       Install one with Homebrew, then re-run this script:" >&2
        echo "" >&2
        echo "           brew install python@3.12" >&2
        echo "" >&2
        echo "       Use 3.12 in particular on an Intel Mac: the last x86_64" >&2
        echo "       macOS torch wheel is 2.2.2 and it is cp312 only." >&2
        echo "" >&2
        exit 1
    fi
    echo "    Using $PYTHON ($("$PYTHON" -V 2>&1))"
    echo "==> Creating venv at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip"
python -m pip install --quiet --upgrade pip

echo "==> Installing requirements (first run downloads ~130 MB of torch; be patient)"
python -m pip install -r "$PROJECT_DIR/requirements.txt"

echo "==> Pre-downloading model '$MODEL' (~400 MB on first run, cached in ~/.cache/huggingface)"
# The name travels through the environment and is read with os.environ inside
# the snippet. Interpolating it into the Python source instead would break on
# any model name containing a quote.
GLINER_MODEL="$MODEL" python -c \
    'import os; from common import load_model, resolve_model_name; load_model(resolve_model_name(os.environ.get("GLINER_MODEL", "base")), "cpu")'

echo ""
echo "Setup complete."
echo "Next: ./start.sh"
