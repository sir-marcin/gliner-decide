#!/bin/bash
# Starts the FastAPI server in the foreground. Ctrl-C to stop.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: no .venv in $(pwd) - run ./setup.sh first" >&2
    exit 1
fi

# shellcheck source=/dev/null
source ".venv/bin/activate"

# GLINER_* is passed through untouched - server.py owns the defaults. The port
# below is only used to print URLs; it must match server.py's own default.
port="${GLINER_PORT:-8765}"

lan_ip() {
    ipconfig getifaddr en0 2>/dev/null \
        || ipconfig getifaddr en1 2>/dev/null \
        || hostname
}

echo "Local:  http://localhost:$port"
echo "LAN:    http://$(lan_ip):$port"
echo "Docs:   http://localhost:$port/docs"
echo ""

exec python server.py
