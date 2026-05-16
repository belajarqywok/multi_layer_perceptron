# =============================================================================
# Makefile — Multi-Layer Perceptron Engine (From Scratch, C++17)
# =============================================================================
# This Makefile orchestrates the full end-to-end ML workflow:
#   1. Initialize + Compile the C++ engine (CMake + clang++)
#   2. Generate synthetic training datasets
#   3. Train using one of three modes: CPU | GPU (OpenCL) | Distributed (TCP)
#   4. Export the trained model to ONNX binary format
#   5. Run cross-language inference: Python | Go | C++ | C | TypeScript
#   6. Deploy cluster: Docker | Systemd | Kubernetes | Helm
#   7. Compile the academic documentation (documentations.tex -> PDF)

.PHONY: all init build seed \
        train-class train-reg train-gpu train-dist \
        inference-py inference-go inference-cpp inference-c inference-ts \
        deploy-docker deploy-docker-build deploy-docker-scale deploy-docker-down \
        deploy-systemd deploy-systemd-workers \
        deploy-k8s deploy-k8s-scale deploy-k8s-down \
        deploy-helm deploy-helm-scale deploy-helm-down \
        docs docs-clean test-all clean

all: test-all

# ─── Build ───────────────────────────────────────────────────────────────────

# Step 1: Initialize the CMake project (generates Makefiles in ./build/)
init:
	mkdir -p build
	cd build && cmake -DCMAKE_CXX_COMPILER=clang++ ..

# Step 2: Compile all targets (mlp engine + inference_cpp + inference_c)
build:
	cd build && make -j4

# ─── Data ────────────────────────────────────────────────────────────────────

# Step 3: Generate synthetic datasets into ./data/
#   - classification.csv  (binary classification with 3 features)
#   - regression.csv      (multi-output regression with 5 features)
seed:
	python3 scripts/seed_csv.py

# ─── Training ────────────────────────────────────────────────────────────────

# Train: Binary Classifier (single-node, CPU)
#   Flags: --alg-type=classifier  --option-std (Z-score normalization)
train-class:
	cd build && ./mlp \
		--input-dataset=../data/classification.csv \
		--column-label=class \
		--column-train=f1,f2,f3 \
		--alg-type=classifier \
		--option-std \
		--hyperp-epochs=200 \
		--output-onnx=model.onnx

# Train: Multi-Output Regressor (single-node, CPU)
train-reg:
	cd build && ./mlp \
		--input-dataset=../data/regression.csv \
		--column-label=f,g \
		--column-train=a,b,c,d,e \
		--alg-type=regressor \
		--option-std \
		--hyperp-epochs=200 \
		--output-onnx=regressor.onnx

# Train: GPU-Accelerated (via OpenCL hardware acceleration)
#   --compute-type=GPU routes Matrix::dot() to the OpenCL kernel.
#   Falls back to CPU automatically if no OpenCL device is found.
train-gpu:
	cd build && ./mlp \
		--input-dataset=../data/classification.csv \
		--column-label=class \
		--column-train=f1,f2,f3 \
		--alg-type=classifier \
		--option-std \
		--hyperp-epochs=200 \
		--compute-type=GPU \
		--output-onnx=gpu_model.onnx

# Train: Distributed Parameter Server (1 Master + 3 Workers via TCP sockets)
#   Requires the test_dist.sh orchestration script.
train-dist:
	./test_dist.sh

# ─── Inference ───────────────────────────────────────────────────────────────

# Inference: Python (uses pip's onnxruntime package)
inference-py:
	python3 inferences/inference.py build/model.onnx

# Inference: Go (uses yalue/onnxruntime_go library)
inference-go:
	go run inferences/inference.go build/model.onnx

# Inference: Native C++ (uses ONNX Runtime C++ API)
inference-cpp:
	cd build && LD_LIBRARY_PATH=../lib/onnxruntime-linux-x64-1.20.0/lib ./inference_cpp model.onnx

# Inference: Native C (uses raw ONNX Runtime C API — no C++ wrappers)
inference-c:
	cd build && LD_LIBRARY_PATH=../lib/onnxruntime-linux-x64-1.20.0/lib ./inference_c model.onnx

# Inference: TypeScript (uses onnxruntime-node npm package)
#   ⚠ WSL1 LIMITATION: Node.js cannot bind stdio inside WSL1.
#   This target works on WSL2 / native Linux / macOS.
#   To run on WSL1, execute from Windows PowerShell:
#     npx ts-node inferences/inference.ts build/model.onnx
inference-ts:
	node.exe --loader ts-node/esm --no-warnings inferences/inference.ts build/model.onnx

# ─── Pipelines ───────────────────────────────────────────────────────────────

# Full end-to-end pipeline: compile → data → train → infer (all 4 native langs)
# inference-ts is excluded from test-all due to WSL1 stdio constraints.
# Run TypeScript inference separately from Windows PowerShell:
#   npx ts-node inferences/inference.ts build/model.onnx
test-all: init build seed train-class inference-py inference-go inference-cpp inference-c
	@echo "========================================="
	@echo "✅ End-to-end tests passed successfully! ✅"
	@echo "========================================="
	@echo ""
	@echo "💡 TypeScript inference (run from Windows PowerShell):"
	@echo "   npx ts-node inferences/inference.ts build/model.onnx"

# Remove all build artifacts and generated data directories
clean:
	rm -rf build/
	rm -rf data/

# ─── Deployment ──────────────────────────────────────────────────────────────

# Docker: Build the container image
deploy-docker-build:
	docker build -t mlp-engine:latest -f deploy/docker/Dockerfile .

# Docker: Start the distributed cluster (1 master + 3 workers by default)
deploy-docker:
	docker compose -f deploy/docker/docker-compose.yml up --build -d

# Docker: Scale workers dynamically (e.g. make deploy-docker-scale WORKERS=5)
deploy-docker-scale:
	docker compose -f deploy/docker/docker-compose.yml up -d --scale worker=$(WORKERS)

# Docker: Stop and remove all containers
deploy-docker-down:
	docker compose -f deploy/docker/docker-compose.yml down

# Systemd: Install and enable master + 3 workers on the local machine
deploy-systemd:
	sudo cp deploy/systemd/mlp-master.service /etc/systemd/system/
	sudo cp "deploy/systemd/mlp-worker@.service" /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable --now mlp-master
	sudo systemctl enable --now 'mlp-worker@{1,2,3}'

# Systemd: Enable N worker instances (e.g. make deploy-systemd-workers N=5)
deploy-systemd-workers:
	for i in $$(seq 1 $(N)); do \
		sudo systemctl enable --now "mlp-worker@$$i"; \
	done

# Kubernetes: Apply all manifests (namespace, configmap, master, workers)
deploy-k8s:
	kubectl apply -f deploy/k8s/namespace.yaml
	kubectl apply -f deploy/k8s/configmap.yaml
	kubectl apply -f deploy/k8s/master-deployment.yaml
	kubectl apply -f deploy/k8s/worker-deployment.yaml
	@echo "Cluster deployed. Check status: kubectl get pods -n mlp-engine"

# Kubernetes: Scale the worker Deployment to N replicas
# Usage: make deploy-k8s-scale WORKERS=5
deploy-k8s-scale:
	kubectl scale deployment mlp-worker -n mlp-engine --replicas=$(WORKERS)
	@echo "Scaled to $(WORKERS) workers."

# Kubernetes: Tear down all K8s resources
deploy-k8s-down:
	kubectl delete namespace mlp-engine

# Helm: Install the Helm chart with default values (3 workers)
deploy-helm:
	helm upgrade --install mlp-engine ./deploy/helm/mlp-engine \
		--namespace mlp-engine \
		--create-namespace
	@echo "Chart deployed. Check: kubectl get pods -n mlp-engine"

# Helm: Upgrade and change the worker count
# Usage: make deploy-helm-scale WORKERS=5
deploy-helm-scale:
	helm upgrade mlp-engine ./deploy/helm/mlp-engine \
		--namespace mlp-engine \
		--set worker.replicaCount=$(WORKERS)
	@echo "Helm release updated to $(WORKERS) workers."

# Helm: Uninstall the Helm release
deploy-helm-down:
	helm uninstall mlp-engine --namespace mlp-engine

# ─── Documentation ──────────────────────────────────────────────────────────────

# Compile documentations.tex into documentations.pdf (requires pdflatex/texlive)
# Run twice to resolve cross-references and table of contents
docs:
	pdflatex documentations.tex
	pdflatex documentations.tex
	@echo "PDF generated: documentations.pdf"

# Remove LaTeX build artifacts (keep the .pdf)
docs-clean:
	rm -f documentations.aux documentations.log documentations.out documentations.toc
