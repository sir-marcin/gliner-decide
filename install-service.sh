#!/bin/bash
# Installs a launchd user agent so the server starts at login (a LaunchAgent in
# gui/<uid> does NOT start at boot) and restarts if it dies. Re-run to update it
# with new settings.
set -euo pipefail

cd "$(dirname "$0")"

PROJECT_DIR="$(pwd)"
LABEL="com.minos.server"
LEGACY_LABEL="com.gliner-decide.server"   # pre-rename; removed below if present
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LEGACY_PLIST="$HOME/Library/LaunchAgents/$LEGACY_LABEL.plist"
PYTHON="$PROJECT_DIR/.venv/bin/python"
SERVER="$PROJECT_DIR/server.py"
LOG="$PROJECT_DIR/logs/server.log"
TARGET="gui/$(id -u)"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found - run ./setup.sh first" >&2
    exit 1
fi

# Escape the five XML predefined entities. Every value interpolated into the
# plist below goes through this: a project path containing & or < produces a
# malformed plist that launchctl silently refuses to load.
xml_escape() {
    printf '%s' "$1" | sed -e 's/&/\&amp;/g' \
                           -e 's/</\&lt;/g' \
                           -e 's/>/\&gt;/g' \
                           -e 's/"/\&quot;/g' \
                           -e "s/'/\&apos;/g"
}

# Only GLINER_* vars actually set in this environment are written into the
# plist; anything omitted falls back to server.py's own defaults. Defining them
# here too would mean two places to keep in sync.
plist_env() {
    local name value
    for name in GLINER_HOST GLINER_PORT GLINER_MODEL GLINER_DEVICE GLINER_CORS_ORIGINS \
                GLINER_MAX_QUEUE GLINER_QUEUE_TIMEOUT GLINER_MAX_BATCH GLINER_ALLOW_ANY_MODEL; do
        value="${!name-}"
        [ -n "${!name+set}" ] || continue
        printf '        <key>%s</key>\n        <string>%s</string>\n' \
            "$name" "$(xml_escape "$value")"
    done
}

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$HOME/Library/LaunchAgents"

# An agent installed under the old label would keep running and hold the port,
# so retire it before installing the renamed one.
if launchctl print "$TARGET/$LEGACY_LABEL" >/dev/null 2>&1; then
    echo "Removing the pre-rename service $LEGACY_LABEL"
    launchctl bootout "$TARGET/$LEGACY_LABEL" 2>/dev/null || true
    for _ in $(seq 1 50); do
        launchctl print "$TARGET/$LEGACY_LABEL" >/dev/null 2>&1 || break
        sleep 0.2
    done
fi
if [ -f "$LEGACY_PLIST" ]; then
    rm -f "$LEGACY_PLIST"
    echo "Removed $LEGACY_PLIST"
fi

LABEL_X="$(xml_escape "$LABEL")"
PYTHON_X="$(xml_escape "$PYTHON")"
SERVER_X="$(xml_escape "$SERVER")"
PROJECT_DIR_X="$(xml_escape "$PROJECT_DIR")"
LOG_X="$(xml_escape "$LOG")"
ENV_X="$(plist_env; printf x)"   # sentinel: $() would eat the trailing newline
ENV_X="${ENV_X%x}"

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL_X</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_X</string>
        <string>$SERVER_X</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR_X</string>
    <key>EnvironmentVariables</key>
    <dict>
$ENV_X    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>60</integer>
    <key>StandardOutPath</key>
    <string>$LOG_X</string>
    <key>StandardErrorPath</key>
    <string>$LOG_X</string>
</dict>
</plist>
PLIST_EOF

# Catch a malformed plist here rather than as a silent launchctl failure.
if ! plutil -lint "$PLIST" >/dev/null 2>&1; then
    echo "ERROR: generated plist is not valid: $PLIST" >&2
    plutil -lint "$PLIST" >&2 || true
    exit 1
fi

echo "Wrote $PLIST"

# Unload any previous version so a re-run picks up the new plist. bootout is
# asynchronous, so wait for the service to actually go away before reloading.
if launchctl print "$TARGET/$LABEL" >/dev/null 2>&1; then
    echo "Replacing the running service"
    launchctl bootout "$TARGET/$LABEL" 2>/dev/null || true
    for _ in $(seq 1 50); do
        launchctl print "$TARGET/$LABEL" >/dev/null 2>&1 || break
        sleep 0.2
    done
fi

loaded=""
for _ in 1 2 3; do
    if launchctl bootstrap "$TARGET" "$PLIST" 2>/dev/null; then
        loaded="bootstrap"
        break
    fi
    sleep 1
done

if [ -n "$loaded" ]; then
    echo "Loaded via launchctl bootstrap $TARGET"
else
    echo "bootstrap failed, falling back to launchctl load -w"
    launchctl load -w "$PLIST"
fi

echo ""
echo "Service $LABEL is installed and will start at login (not at boot)."
echo "  URL:     http://localhost:${GLINER_PORT:-8765}"
echo "  Status:  launchctl print $TARGET/$LABEL"
echo "  Logs:    tail -f \"$LOG\""
echo "  Remove:  ./uninstall-service.sh"
echo ""
echo "The model loads at startup, so give it ~10s before the first request."
