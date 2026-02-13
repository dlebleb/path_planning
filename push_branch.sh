#!/bin/sh
# Run this script from your terminal (not inside Cursor) to push.
# Your system Git will use your stored credentials or prompt you once.
set -e
cd "$(dirname "$0")"
git push origin rachel
echo "Pushed branch rachel successfully."
