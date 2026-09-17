#!/bin/bash
# Stops and removes the launchd user agent installed by ./install-service.sh.
set -euo pipefail

# Both labels are handled: an agent installed before the rename to Minos still
# carries the old one, and this script is the only thing that will remove it.
LABELS=("com.minos.server" "com.gliner-decide.server")
TARGET="gui/$(id -u)"

for LABEL in "${LABELS[@]}"; do
    PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

    if launchctl bootout "$TARGET/$LABEL" 2>/dev/null; then
        echo "Stopped $LABEL (launchctl bootout)"
    elif [ -f "$PLIST" ] && launchctl unload -w "$PLIST" 2>/dev/null; then
        echo "Stopped $LABEL (launchctl unload)"
    else
        echo "Service $LABEL was not loaded"
    fi

    if [ -f "$PLIST" ]; then
        rm -f "$PLIST"
        echo "Removed $PLIST"
    else
        echo "No plist at $PLIST"
    fi
done

echo ""
echo "Verify it is gone (this should now fail):"
echo "  launchctl print $TARGET/${LABELS[0]}"
echo "Logs are kept at logs/server.log:"
echo "  tail -f logs/server.log"
