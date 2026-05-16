#!/bin/bash
# =============================================================================
# entrypoint.sh — Docker container entrypoint for the MLP engine
# =============================================================================
# This script handles the startup logic for each node role:
#   - master:   Starts the Parameter Server and waits for workers.
#   - worker:   Connects to the master, downloads weights, trains, pushes grads.
#   - shutdown: Sends a shutdown signal to the master (used to gracefully stop).
# =============================================================================

set -e

# Generate dataset if it doesn't already exist
if [ ! -f "data/classification.csv" ]; then
    echo "[entrypoint] Generating synthetic dataset..."
    mkdir -p data
    python3 scripts/seed_csv.py
fi

echo "[entrypoint] Starting role=${ROLE} | master=${MASTER_ADDR}:${MASTER_PORT} | epochs=${EPOCHS}"

case "$ROLE" in
  master)
    echo "[entrypoint] Starting Master (Parameter Server)..."
    exec ./mlp \
        --input-dataset=data/classification.csv \
        --column-label=class \
        --column-train=f1,f2,f3 \
        --alg-type=classifier \
        --option-std \
        --hyperp-epochs="${EPOCHS}" \
        --compute-type="${COMPUTE_TYPE}" \
        --output-onnx=model.onnx \
        --type-training=distributed \
        --dist-role=master \
        --dist-master-addr="${MASTER_ADDR}" \
        --dist-master-port="${MASTER_PORT}"
    ;;

  worker)
    # Workers wait a moment for the master to be ready before connecting
    echo "[entrypoint] Waiting for master to be ready..."
    sleep "${WORKER_DELAY:-5}"
    echo "[entrypoint] Starting Worker..."
    exec ./mlp \
        --input-dataset=data/classification.csv \
        --column-label=class \
        --column-train=f1,f2,f3 \
        --alg-type=classifier \
        --option-std \
        --hyperp-epochs="${EPOCHS}" \
        --compute-type="${COMPUTE_TYPE}" \
        --type-training=distributed \
        --dist-role=worker \
        --dist-master-addr="${MASTER_ADDR}" \
        --dist-master-port="${MASTER_PORT}"
    ;;

  shutdown)
    echo "[entrypoint] Sending shutdown signal to master..."
    exec ./mlp \
        --type-training=distributed \
        --dist-role=shutdown \
        --dist-master-addr="${MASTER_ADDR}" \
        --dist-master-port="${MASTER_PORT}"
    ;;

  *)
    echo "[entrypoint] Unknown role: ${ROLE}. Valid: master | worker | shutdown"
    exit 1
    ;;
esac
