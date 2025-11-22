#!/bin/bash
# Script to start Jupyter for Google Colab connection

# Get the absolute path to the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "======================================"
echo "Starting Jupyter for Google Colab"
echo "======================================"
echo "Project Root: $PROJECT_ROOT"

# Ensure container is running
if ! docker ps | grep -q "gpu4pyscf-dev"; then
    echo "Container is not running. Starting it..."
    "$SCRIPT_DIR/start-environment.sh"
fi

# Stop any existing Jupyter instances to avoid port conflicts
echo "Stopping any existing Jupyter servers..."
docker compose exec gpu4pyscf pkill -f jupyter || true
sleep 2

echo ""
echo "Starting Jupyter server..."
echo "Copy the URL with the token below and paste it into Google Colab."
echo "======================================"

docker compose exec gpu4pyscf jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0 \
  --allow-root \
  --ip=0.0.0.0 \
  --no-browser
