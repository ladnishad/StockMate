#!/bin/bash
# Run local backend for iOS development
#
# Prerequisites:
#   1. Create deploy/local/.env with Supabase POOLER connection string
#
# Usage:
#   ./scripts/run-local.sh          # Uses Docker (requires pooler DATABASE_URL)
#   ./scripts/run-local.sh --native # Runs directly with Python (works with any DATABASE_URL)

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check for deploy/local/.env
if [ ! -f "$PROJECT_ROOT/deploy/local/.env" ]; then
    echo "ERROR: deploy/local/.env not found!"
    echo ""
    echo "Please create it with your environment variables."
    echo "IMPORTANT: Use Supabase POOLER connection string for DATABASE_URL"
    echo "Find it in: Supabase Dashboard > Project Settings > Database > Connection String > Transaction mode"
    exit 1
fi

# Get local IP address
if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "Unable to determine")
else
    LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "Unable to determine")
fi

echo "========================================"
echo "  StockMate Local Development Server"
echo "========================================"
echo ""
echo "Your local IP address: $LOCAL_IP"
echo ""
echo "API will be available at:"
echo "  - http://localhost:8000 (from this machine)"
echo "  - http://$LOCAL_IP:8000 (from iOS device/simulator)"
echo ""
echo "Update Configuration.swift if needed:"
echo "  return \"http://$LOCAL_IP:8000\""
echo ""
echo "Health check: curl http://localhost:8000/health"
echo "========================================"
echo ""

# Check if --native flag is passed
if [ "$1" == "--native" ]; then
    echo "Running in NATIVE mode (Python directly)..."
    echo ""

    # Check if virtual environment exists
    if [ ! -d "$PROJECT_ROOT/venv" ] && [ ! -d "$PROJECT_ROOT/.venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv "$PROJECT_ROOT/venv"
        echo "Installing dependencies..."
        source "$PROJECT_ROOT/venv/bin/activate"
        pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        # Activate existing venv
        if [ -d "$PROJECT_ROOT/venv" ]; then
            source "$PROJECT_ROOT/venv/bin/activate"
        else
            source "$PROJECT_ROOT/.venv/bin/activate"
        fi
    fi

    # Load deploy/local/.env
    set -a
    source "$PROJECT_ROOT/deploy/local/.env"
    set +a

    # Override for local
    export APP_ENV=local
    export LOG_LEVEL=DEBUG

    echo "Starting backend (native)..."
    echo ""
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
else
    echo "Running in DOCKER mode..."
    echo "(Use --native flag to run with Python directly)"
    echo ""

    # Start docker compose from deploy/local
    docker-compose -f "$PROJECT_ROOT/deploy/local/docker-compose.yml" up --build
fi
