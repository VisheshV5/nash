#!/usr/bin/env bash
# Serve the demo table: scripts/demo.sh [bundle] [port]   then open http://localhost:<port>/demo/
set -euo pipefail
cd "$(dirname "$0")/.."
export REGRET_BUNDLE="${1:-bundles/hu-100bb}"
exec .venv/bin/python -m uvicorn regret.api.app:app --host 127.0.0.1 --port "${2:-8765}"
