# =============================================================================
# Multi-Stage Dockerfile — MLP Engine (C++17, From Scratch)
# =============================================================================
# Stage 1 (builder): Compiles the C++ engine with all build dependencies.
# Stage 2 (runtime): Produces a lean final image containing only the binary.
#
# Usage:
#   docker build -t mlp-engine:latest .
# =============================================================================

# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

LABEL stage="builder"

# Install all C++ build dependencies and Python (for dataset generation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    clang \
    make \
    libprotobuf-dev \
    protobuf-compiler \
    opencl-headers \
    ocl-icd-opencl-dev \
    ocl-icd-libopencl1 \
    python3 \
    python3-pip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source files into the builder stage
COPY src/          ./src/
COPY inferences/   ./inferences/
COPY scripts/      ./scripts/
COPY CMakeLists.txt ./
COPY src/onnx.proto ./src/

# Download ONNX Runtime shared library for the C++ inference binary
RUN mkdir -p lib && \
    curl -L -o /tmp/ort.tgz \
      https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz && \
    tar -xzf /tmp/ort.tgz -C lib/ && \
    rm /tmp/ort.tgz

# Configure and build all targets with CMake
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_CXX_COMPILER=clang++ .. && \
    make -j$(nproc)


# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

# Install only the minimal runtime libraries needed to execute the binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libprotobuf-lite32t64 \
    libstdc++6 \
    ocl-icd-libopencl1 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy --break-system-packages

WORKDIR /app

# Copy compiled binaries from builder stage
COPY --from=builder /build/build/mlp                  ./mlp
COPY --from=builder /build/lib/onnxruntime-linux-x64-1.20.0/lib/ ./lib/
COPY --from=builder /build/scripts/                   ./scripts/

# Copy entrypoint script
COPY deploy/docker/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

# Default environment variables — override via docker-compose or K8s env:
# ROLE=master|worker         Which role this container plays
# MASTER_ADDR=localhost      Hostname of the master node
# MASTER_PORT=8080           Port the master listens on
# EPOCHS=200                 Number of training epochs
# COMPUTE_TYPE=CPU           CPU or GPU (OpenCL)
ENV ROLE=master \
    MASTER_ADDR=localhost \
    MASTER_PORT=8080 \
    EPOCHS=200 \
    COMPUTE_TYPE=CPU

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
