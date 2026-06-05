#!/usr/bin/env bash
# Run TabQA Generator locally using the gradino virtualenv.
# Prerequisites: gradino/env/ must exist with Python 3.13 and Gradino deps.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACTIVATE="$SCRIPT_DIR/gradino/env/bin/activate"

if [ ! -f "$ACTIVATE" ]; then
    echo "ERROR: $ACTIVATE not found."
    echo "Set up the gradino env first:"
    echo "  cd gradino && python3.13 -m venv env && env/bin/pip install -r requirements.txt"
    exit 1
fi

# Activate the gradino virtualenv (sets PATH, VIRTUAL_ENV, etc.)
# shellcheck source=/dev/null
source "$ACTIVATE"

echo "Python: $(python3 --version)"

# Install backend deps into the active env (fast no-op if already installed)
python3 -m pip install --quiet fastapi "uvicorn[standard]" python-multipart aiofiles "anthropic>=0.30.0"

# Load OPENAI_API_KEY from gradino/.env if not already in environment
if [ -z "$OPENAI_API_KEY" ] && [ -f "$SCRIPT_DIR/gradino/.env" ]; then
    KEY=$(grep -v '^#' "$SCRIPT_DIR/gradino/.env" | grep 'OPENAI_API_KEY' | head -1 | cut -d'=' -f2- | tr -d '"' | tr -d "'" | xargs)
    if [ -n "$KEY" ]; then
        export OPENAI_API_KEY="$KEY"
        echo "Loaded OPENAI_API_KEY from gradino/.env"
    fi
fi

export GRADINO_PATH="$SCRIPT_DIR/gradino"

echo ""
echo "Starting TabQA Generator → http://localhost:8000"
echo ""
cd "$SCRIPT_DIR/backend"
exec python3 -m uvicorn app:app --host 0.0.0.0 --port 8001 --log-level info
