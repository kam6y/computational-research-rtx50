#!/bin/bash
# GPU4PySCF Environment Startup Script for WSL2
# This script checks prerequisites and starts the Docker environment

set -e

# Get the absolute path to the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "======================================"
echo "GPU4PySCF Environment Startup"
echo "======================================"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in WSL2
check_wsl2() {
    echo -n "Checking WSL2... "
    if grep -qi microsoft /proc/version; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${YELLOW}Warning: Not running in WSL2${NC}"
        return 1
    fi
}

# Check if Docker is installed
check_docker() {
    echo -n "Checking Docker... "
    if command -v docker &> /dev/null; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "Docker is not installed or not in PATH"
        return 1
    fi
}

# Check if Docker daemon is running
check_docker_daemon() {
    echo -n "Checking Docker daemon... "
    if docker info &> /dev/null; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "Docker daemon is not running. Please start Docker Desktop."
        return 1
    fi
}

# Check if docker-compose is available
check_docker_compose() {
    echo -n "Checking docker-compose... "
    if docker compose version &> /dev/null; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "docker-compose is not available"
        return 1
    fi
}

# Check NVIDIA Docker runtime
check_nvidia_docker() {
    echo -n "Checking NVIDIA Docker runtime... "
    if docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        echo "NVIDIA Docker runtime is not available."
        echo "Please install NVIDIA drivers for WSL2. See SETUP.md for instructions."
        return 1
    fi
}

# Display GPU information
show_gpu_info() {
    echo ""
    echo "GPU Information:"
    echo "----------------"
    docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi || true
    echo ""
}

# Main execution
main() {
    # Run checks
    check_wsl2 || true
    check_docker || exit 1
    check_docker_daemon || exit 1
    check_docker_compose || exit 1

    echo ""
    echo -n "Checking NVIDIA GPU support... "
    if check_nvidia_docker; then
        show_gpu_info
    else
        echo ""
        echo -e "${YELLOW}WARNING: GPU support check failed.${NC}"
        echo "The container will start but may not have GPU access."
        echo "See SETUP.md for NVIDIA driver installation instructions."
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Create cache and temp directories
    mkdir -p .pyscf_tmp .tmp .nv_cache .cupy_cache

    # Build and start the environment
    echo "======================================"
    echo "Building Docker image..."
    echo "======================================"
    docker compose build

    echo ""
    echo "======================================"
    echo "Starting GPU4PySCF environment..."
    echo "======================================"
    docker compose up -d

    echo ""
    echo -e "${GREEN}Environment started successfully!${NC}"
    echo ""
    echo "To access the container:"
    echo "  docker compose exec gpu4pyscf bash"
    echo ""
    echo "To run tests:"
    echo "  docker compose exec gpu4pyscf python3 tests/test_gpu4pyscf.py"
    echo ""
    echo "To stop the environment:"
    echo "  docker compose down"
    echo ""
}

# Run main function
main
