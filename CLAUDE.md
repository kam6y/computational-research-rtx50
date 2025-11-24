# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. your partner is Japanese. so, please answer in Japanese.

## Project Overview

This is a GPU-accelerated quantum chemistry computation environment optimized for NVIDIA RTX 5070 Ti (Blackwell generation, compute capability 10.0). The project runs GPU4PySCF v1.4.0 inside a Docker container on WSL2, enabling high-performance density functional theory (DFT) calculations.

**Base project**: Adapted from [ByteDance-Seed/JoltQC](https://github.com/ByteDance-Seed/JoltQC) Dockerfile.

## Core Technologies

- **CUDA**: 12.9.1 (specifically for Blackwell architecture)
- **GPU4PySCF**: v1.4.0 (GPU-accelerated PySCF library)
- **PySCF**: Quantum chemistry calculations
- **CuPy**: CUDA-accelerated NumPy
- **Docker**: Containerized environment with GPU passthrough
- **Automatic Memory Management**: Cache-free GPU memory cleanup utilities
- **Google Colab Integration**: Fully automated memory management for Colab notebooks

## Common Development Commands

### Environment Management

Build and start the environment:
```bash
./scripts/start-environment.sh
```

Access the running container:
```bash
docker compose exec gpu4pyscf bash
```

Stop the environment:
```bash
docker compose down
```

Rebuild the Docker image (after Dockerfile changes):
```bash
docker compose build
```

### Running Tests

Execute the full test suite inside the container:
```bash
docker compose exec gpu4pyscf python3 tests/test_gpu4pyscf.py
```

Run specific test files (all with automatic memory cleanup):
```bash
docker compose exec gpu4pyscf python3 tests/test_vitamin_d.py        # 20 consecutive runs
docker compose exec gpu4pyscf python3 tests/test_vitamin_d_opt.py    # Geometry optimization
docker compose exec gpu4pyscf python3 tests/test_memory_tracking.py  # Memory leak verification
docker compose exec gpu4pyscf python3 tests/test_cupy_cache.py       # Cache test
```

Run tests from inside the container:
```bash
python3 tests/test_gpu4pyscf.py
```

**All tests include automatic GPU memory cleanup - no memory leaks occur even with 20+ consecutive runs.**

### Running Python Scripts

From outside the container:
```bash
docker compose exec gpu4pyscf python3 your_script.py
```

From inside the container:
```bash
python3 your_script.py
```

## Architecture and Critical Details

### GPU and CUDA Configuration

**CRITICAL**: RTX 5070 Ti uses compute capability **10.0** (Blackwell generation). All CUDA compilation must target this architecture:

- `Dockerfile`: `ENV CUDA_ARCH_LIST="100"`
- CMake build: `-DCUDA_ARCHITECTURES="100"`

### First-Run JIT Compilation Behavior

**IMPORTANT**: The RTX 5070 Ti is extremely new hardware. On the **first GPU calculation**, CUDA kernels undergo Just-In-Time (JIT) compilation, which can take **15-20 minutes** and appears to hang at "Running DFT calculation on GPU...". This is normal behavior. Subsequent runs are fast (seconds) because kernels are cached.

Cache locations (persisted across container restarts):
- CuPy cache: `/workspace/.cupy_cache` (mounted from host `.cupy_cache`)
- CUDA cache: `/workspace/.nv_cache` (mounted from host `.nv_cache`)
- PySCF temp: `/workspace/.pyscf_tmp` (mounted from host `.pyscf_tmp`)

### Docker Configuration Details

The `docker-compose.yml` includes critical GPU settings:
- GPU access enabled via `deploy.resources.reservations.devices`
- Shared memory size: `2gb` (important for GPU operations)
- Environment variables for cache persistence and GPU visibility

Volumes are used to persist:
- Workspace files (current directory mounted to `/workspace`)
- Pip cache (named volume `pip-cache`)
- PySCF data (named volume `pyscf-data`)

### GPU4PySCF Build Process

The Dockerfile builds GPU4PySCF from source using CMake:
```bash
cmake -B build -S gpu4pyscf/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCHITECTURES="100" \
    -DBUILD_LIBXC=ON
```

Build output: `/opt/gpu4pyscf/build/`

Environment variables set in Dockerfile:
- `PYTHONPATH="/opt/gpu4pyscf:${PYTHONPATH}"`
- `LD_LIBRARY_PATH="/opt/gpu4pyscf/build:${LD_LIBRARY_PATH}"`

### Test Suite Structure

**[tests/test_utils.py](tests/test_utils.py)**: Automatic GPU memory cleanup utilities:
- `cleanup_gpu_memory()`: Frees CuPy memory pools and pinned memory
- `full_cleanup()`: Complete cleanup of GPU memory + object deletion + garbage collection
- `periodic_cleanup()`: Automatic cleanup for iterative calculations
- Used by all test files to prevent memory leaks

**[tests/test_gpu4pyscf.py](tests/test_gpu4pyscf.py)**: Comprehensive test suite with 7 tests (all with automatic memory cleanup):
1. GPU detection and CUDA setup (CuPy verification)
2. GPU4PySCF import verification
3. CPU DFT calculation (H2O molecule, B3LYP/def2-svp)
4. GPU DFT calculation with CPU comparison
5. Performance comparison (CPU vs GPU)
6. Larger molecule test (benzene C6H6)
7. Gradient calculation (forces on H2)

**[tests/test_vitamin_d.py](tests/test_vitamin_d.py)**: Large molecule benchmark - Vitamin D3 (Cholecalciferol, 80 atoms) using WB97XD/6-311G(d). Runs 10 GPU + 10 CPU calculations (20 total) with automatic memory cleanup after each run. No memory leaks occur.

**[tests/test_vitamin_d_opt.py](tests/test_vitamin_d_opt.py)**: Geometry optimization test with automatic cleanup after 100-step optimization.

**[tests/test_memory_tracking.py](tests/test_memory_tracking.py)**: Detailed memory leak verification - runs multiple tests to confirm no memory accumulation occurs.

**[tests/test_cupy_cache.py](tests/test_cupy_cache.py)**: Cache verification to ensure JIT-compiled kernels persist, with automatic memory cleanup.

### Typical Workflow for Quantum Chemistry Calculations

1. Define molecule geometry using `pyscf.gto.M()` with atom coordinates and basis set
2. Choose between CPU (`pyscf.dft`) or GPU (`gpu4pyscf.dft`) module
3. Create a calculation object (e.g., `RKS` for restricted Kohn-Sham)
4. Set exchange-correlation functional (e.g., `xc = 'B3LYP'`)
5. Run calculation with `.kernel()` method
6. Optionally compute gradients with `.nuc_grad_method().kernel()`

### When GPU Acceleration Helps

GPU acceleration is most beneficial for:
- Large systems (100+ atoms)
- High-precision basis sets (def2-TZVP, 6-311G(d), etc.)
- Multiple calculations (structure optimization, molecular dynamics)

Small molecules may show minimal or negative speedup due to JIT overhead.

## File Structure

```
.
├── Dockerfile                    # CUDA 12.9.1 + GPU4PySCF build
├── docker-compose.yml            # GPU passthrough + volume config
├── scripts/
│   ├── start-environment.sh      # Automated setup + prerequisite checks
│   └── start-colab.sh            # Google Colab integration
├── tests/
│   ├── test_utils.py             # Automatic GPU memory cleanup utilities
│   ├── test_gpu4pyscf.py         # Primary test suite (with auto cleanup)
│   ├── test_vitamin_d.py         # Large molecule benchmark (20 runs)
│   ├── test_vitamin_d_opt.py     # Optimization test
│   ├── test_memory_tracking.py   # Memory leak verification test
│   └── test_cupy_cache.py        # Cache persistence test
├── colab/
│   ├── colab_auto_cleanup.py     # IPython extension for auto cleanup
│   └── tutorial_basic.ipynb      # Basic Colab tutorial notebook
└── .pyscf_tmp/, .nv_cache/, .cupy_cache/  # Cache directories
```

## WSL2 and GPU Requirements

This environment requires:
- Windows 11 (or Windows 10 Build 19044+) with WSL2
- NVIDIA GPU drivers installed on **Windows side** (not inside WSL2)
- Docker Desktop with WSL2 backend enabled
- WSL Integration enabled for the Ubuntu distribution

Verify GPU access in WSL2:
```bash
nvidia-smi  # Should show RTX 5070 Ti
```

Verify Docker GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

## Common Issues

### Build fails with CUDA_ARCHITECTURES error
Ensure `ENV CUDA_ARCH_LIST="100"` is set in Dockerfile for RTX 5070 Ti.

### First GPU calculation hangs for 15-20 minutes
This is JIT compilation (see "First-Run JIT Compilation Behavior" above). Wait for completion; subsequent runs will be fast.

### Out of Memory errors
Increase `shm_size` in docker-compose.yml (e.g., to `4gb`).

### Permission errors on cache directories
Run `chmod -R 755 .cupy_cache .nv_cache .pyscf_tmp` from WSL2.

