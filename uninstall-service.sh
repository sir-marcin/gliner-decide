#!/bin/bash
# Stops and removes the launchd user agent installed by ./install-service.sh.
set -euo pipefail

LABEL="com.gliner-decide.server"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
TARGET="gui/$(id -u)"

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

echo ""
echo "Verify it is gone (this should now fail):"
echo "  launchctl print $TARGET/$LABEL"
echo "Logs are kept at logs/server.log:"
echo "  tail -f logs/server.log"
