# Dockerfile for gpu4pyscf on WSL2 with RTX 5070 Ti
# Based on ByteDance-Seed/JoltQC Dockerfile
# Optimized for NVIDIA RTX 5070 Ti (Blackwell, compute capability 10.0)

FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
# RTX 5070 Ti: Blackwell architecture (compute capability 10.0)
ENV CUDA_ARCH_LIST="100"
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev \
    wget \
    curl \
    gfortran \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel

# Install core scientific computing libraries
RUN pip3 install \
    numpy \
    scipy \
    h5py \
    pyscf \
    cupy-cuda12x \
    cutensor-cu12 \
    basis-set-exchange

# Install additional PySCF extensions
RUN pip3 install \
    pyscf-dispersion \
    geometric

# Install utilities for monitoring and testing
RUN pip3 install \
    pytest \
    matplotlib \
    pytest \
    matplotlib \
    pandas \
    jupyter \
    "notebook<7.0.0" \
    jupyter_http_over_ws

# Enable Jupyter Colab extension
RUN jupyter serverextension enable --py jupyter_http_over_ws

# Clone gpu4pyscf v1.4.0
WORKDIR /opt
RUN git clone --branch v1.4.0 --depth 1 https://github.com/pyscf/gpu4pyscf.git

# Build gpu4pyscf with cmake
WORKDIR /opt/gpu4pyscf
RUN cmake -B build -S gpu4pyscf/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ARCHITECTURES="${CUDA_ARCH_LIST}" \
    -DBUILD_LIBXC=ON

# Build with parallel compilation (adjust -j based on your system)
RUN cd build && make -j$(nproc)

# Set environment variables for gpu4pyscf
ENV PYTHONPATH="/opt/gpu4pyscf:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/opt/gpu4pyscf/build:${LD_LIBRARY_PATH}"

# Create workspace directory
WORKDIR /workspace

# Create a startup script to verify GPU
RUN echo '#!/bin/bash\n\
    echo "=== GPU4PySCF Environment ==="\n\
    echo "CUDA Version: $(nvcc --version | grep release)"\n\
    echo "GPU Devices:"\n\
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv\n\
    echo ""\n\
    echo "Python packages:"\n\
    pip3 list | grep -E "pyscf|cupy|numpy"\n\
    echo ""\n\
    echo "Environment ready. Run tests with: python3 test_gpu4pyscf.py"\n\
    exec "$@"' > /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
