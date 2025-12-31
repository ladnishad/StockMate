#!/bin/bash
# =============================================================================
# Test Server Startup Script
# Starts local PostgreSQL via Docker and runs the API with test configuration
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo "  StockMate Test Environment"
echo "========================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Load production .env to get API keys, then override with test settings
echo "[1/4] Loading environment..."
if [ -f .env ]; then
    # Extract only ALPACA and CLAUDE keys, filter out comments and empty lines
    while IFS='=' read -r key value; do
        if [[ "$key" =~ ^(ALPACA_API_KEY|ALPACA_SECRET_KEY|CLAUDE_API_KEY)$ ]]; then
            export "$key=$value"
        fi
    done < <(grep -v '^#' .env | grep -v '^$' | grep -E '^(ALPACA_|CLAUDE_)')
fi

# Export test environment variables
export APP_ENV=test
export LOG_LEVEL=DEBUG
export DATABASE_URL=postgresql://stockmate:localdev@localhost:5433/stockmate
export ALPACA_BASE_URL=https://api.alpaca.markets
export SUPABASE_URL=""
export SUPABASE_ANON_KEY=""
export SUPABASE_SERVICE_KEY=""
export SUPABASE_JWT_SECRET="test-jwt-secret-for-local-development-only"
export TEST_MODE=true
export BYPASS_AUTH=true
export RATE_LIMIT_PER_MINUTE=1000
export RATE_LIMIT_AI_PER_MINUTE=100

# Real subagents mode - set to "true" to use real API calls with chart generation
# When false (default), uses simulation mode for faster testing
export USE_REAL_SUBAGENTS=${USE_REAL_SUBAGENTS:-false}

# Start PostgreSQL container
echo "[2/4] Starting PostgreSQL container..."
docker compose up -d db

# Wait for PostgreSQL to be ready
echo "[3/4] Waiting for PostgreSQL to be healthy..."
until docker compose exec -T db pg_isready -U stockmate -d stockmate > /dev/null 2>&1; do
    echo "  Waiting for database..."
    sleep 2
done
echo "  Database is ready!"

# Activate virtual environment and start server
echo "[4/4] Starting API server..."
echo ""
echo "========================================"
echo "  Server starting on http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo "  Test endpoint: POST /plan/{symbol}/generate/v2"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop"
echo ""

source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
