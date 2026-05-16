# Building a Universal Multi-Layer Perceptron from First Principles: A C++ Implementation with Cross-Language ONNX Portability

**Abstract**  
*This repository details the fundamental implementation of a Multi-Layer Perceptron (MLP) Artificial Neural Network constructed entirely from scratch in C++17. By abstaining from high-level backend frameworks (e.g., TensorFlow, PyTorch, Eigen) and relying strictly on the C++ Standard Library, this project demonstrates the raw mathematical mechanics of neural networks. The system features custom Matrix operations, Stochastic/Batch Gradient Descent (SGD) with Backpropagation, robust data pre-processing (Z-score standardization), and most notably, an integrated Open Neural Network Exchange (ONNX) exporter built via Protocol Buffers. This paper-like documentation elucidates the mathematical formulations driving the network and instructions for executing the pipeline.*

---

## 1. Introduction
Modern machine learning heavily abstractions the underlying calculus and linear algebra required to train a neural network. This project aims to strip away those abstractions by implementing a configurable MLP from the ground up. The primary objectives are:
1. **Mathematical Transparency**: To provide a clear, readable C++ codebase that directly translates neural network calculus into imperative logic.
2. **Framework Independence**: To build a robust system without dependencies on external math libraries like BLAS or Eigen.
3. **Hardware Flexibility**: To support both CPU computation and GPU acceleration via OpenCL (`--compute-type=GPU`), routing the $O(N^3)$ Matrix Multiplication to thousands of GPU threads in parallel.
4. **Scalability**: To support multi-node distributed training via a custom TCP-based Parameter Server architecture (`--type-training=distributed`).
5. **Interoperability**: To serialize the trained weights natively into the ONNX format (`model.onnx`), allowing the model to be run for inference in **Python**, **Go**, **C++**, **C**, and **TypeScript**.

## 2. Methodology & Mathematical Formulations

### How This Works Under The Hood (For Beginners)
If you are reading this codebase to learn how Neural Networks actually work behind the scenes, here is the secret sauce:
1. **The Matrix (`src/matrix.hpp`)**: A Neural Network is mostly just multiplying large grids of numbers together. We built a custom Matrix class that flattens 2D grids into a 1D array to maximize CPU cache speed.
2. **The Forward Pass (`src/mlp.hpp`)**: We take your data (X), multiply it by our "Weights" (W), add a "Bias" (B), and pass it through a curved function (Sigmoid/ReLU) to get a prediction.
3. **The Backward Pass (`src/mlp.hpp`)**: We see how wrong the prediction was, and use Calculus (The Chain Rule) to figure out exactly how much each Weight and Bias was responsible for that error. We then tweak them slightly to be less wrong next time.
4. **GPU Acceleration (`src/opencl_utils.hpp`)**: To make the math faster, we can send the matrices to the GPU using OpenCL, where thousands of tiny processors do the multiplication simultaneously.
5. **Distributed Training (`src/network.hpp`)**: To make it *even* faster, we use a Parameter Server architecture over TCP Sockets. A "Master" holds the weights, and multiple "Workers" download them, calculate the errors, and send the tweaks back to the Master.

The network is defined as a sequence of discrete layers, mapping an input vector $X \in \mathbb{R}^{d_{in}}$ to an output vector $Y \in \mathbb{R}^{d_{out}}$. 

### 2.1 Forward Propagation
For an arbitrary layer $L$, let $A^{[L-1]}$ denote the activation matrix of the previous layer (where $A^{[0]} = X$). The pre-activation matrix $Z^{[L]}$ is calculated using the weight matrix $W^{[L]}$ and bias vector $B^{[L]}$:

$$Z^{[L]} = A^{[L-1]} \cdot W^{[L]} + B^{[L]}$$

The activation matrix $A^{[L]}$ is subsequently derived by passing $Z^{[L]}$ through a non-linear activation function $f(x)$:

$$A^{[L]} = f(Z^{[L]})$$

#### 2.1.1 Activation Functions
To model non-linear relationships, the network employs different activation functions based on the layer's topological position and the algorithmic objective:

1. **Hidden Layers (ReLU):** 
   $$f(x) = \max(0, x)$$
   The Rectified Linear Unit (ReLU) mitigates the vanishing gradient problem and introduces sparsity.
   
2. **Output Layer (Binary Classification):**
   $$f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}$$
   The Sigmoid function maps outputs to $(0, 1)$, formulating pseudo-probabilities for binary states.

3. **Output Layer (Regression):**
   $$f(x) = x$$
   A Linear activation allows the model to predict unbounded continuous numerical values.

### 2.2 Backward Propagation & Optimization
The learning process utilizes Gradient Descent to minimize a Loss function $L(Y, \hat{Y})$. The algorithm calculates the partial derivative of the loss with respect to each weight via the **Chain Rule** ($\frac{\partial L}{\partial W}$).

#### 2.2.1 Output Layer Error ($\delta^{out}$)
Depending on the activation function, the delta (error derivative) of the output layer simplifies elegantly:
$$\delta^{[L_{out}]} = A^{[L_{out}]} - Y$$

#### 2.2.2 Hidden Layer Error ($\delta^{[L]}$)
The error is propagated backward to preceding hidden layers. Let $*$ denote the Hadamard (element-wise) product:
$$\delta^{[L]} = (\delta^{[L+1]} \cdot (W^{[L+1]})^T) * f'(Z^{[L]})$$

Where $f'(x)$ for ReLU is the Heaviside step function:
$$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

#### 2.2.3 Parameter Update (Stochastic Gradient Descent)
Gradients for the weights ($\Delta W$) and biases ($\Delta B$) are computed as:
$$\Delta W^{[L]} = (A^{[L-1]})^T \cdot \delta^{[L]}$$
$$\Delta B^{[L]} = \sum_{i=1}^{m} \delta^{[L]}_{i}$$

The parameters are updated using a learning rate $\alpha$:
$$W^{[L]} := W^{[L]} - \alpha \cdot \Delta W^{[L]}$$
$$B^{[L]} := B^{[L]} - \alpha \cdot \Delta B^{[L]}$$

## 3. Data Pre-Processing Engineering
Real-world datasets introduce numerical instability. The `CSVReader` module provides robust preprocessing mechanisms:

1. **Z-Score Standardization**: 
   To accelerate convergence and prevent exploding gradients (specifically in continuous regression tasks), data features (and optionally targets) are scaled:
   $$X_{norm} = \frac{X - \mu}{\sigma}$$
2. **Data Integrity Checks**: Automated cleansing of `NaN` vectors and duplication elimination.

## 4. System Architecture: ONNX Serialization
Rather than utilizing the heavy `onnxruntime` libraries to construct the graph, this system interfaces directly with **Protocol Buffers**. 

1. **Protocol Buffer Compilation**: The canonical `onnx.proto` is compiled via `protoc` into native C++ structures (`onnx.pb.cc` / `onnx.pb.h`).
2. **Graph Construction**: The `ONNXExporter` class iteratively maps the trained $W$ and $B$ tensors to `onnx::TensorProto` nodes.
3. **Computational DAG**: It builds a Directed Acyclic Graph (DAG) using `MatMul`, `Add`, and specific Activation operation nodes.
4. **Binary Serialization**: The model is serialized as an `IR_VERSION 8` compliant `.onnx` binary file.

---

## 5. Experimental Setup & Usage Instructions

This project includes a fully automated `Makefile` workflow designed for Ubuntu/WSL environments.

### 5.1 System Requirements
- **C++ Compiler**: `clang++` (LLVM) supporting C++17.
- **Build Tools**: `cmake`, `make`.
- **Serialization**: `protobuf-compiler`, `libprotobuf-dev`.
- **GPU Acceleration**: `opencl-headers`, `ocl-icd-opencl-dev` (optional, install via `apt`).
- **Inference — Python**: `pip install onnxruntime numpy`
- **Inference — Go**: `go` version 1.22+ (fetches `yalue/onnxruntime_go` automatically)
- **Inference — TypeScript**: `npm install` (installs `onnxruntime-node`, `ts-node`, `typescript`)

### 5.2 One-Click Verification Pipeline
To compile the C++ engine, generate synthetic experimental data, train the mathematical model, serialize the ONNX graph, and run identical inferences sequentially across Python, Go, C++, and C, execute:
```bash
make test-all
```
> **Note (TypeScript):** Due to a WSL1 `stdio` pipe limitation, TypeScript inference must be run from **Windows PowerShell**:
> ```powershell
> npx ts-node inferences/inference.ts build/model.onnx
> ```
> On WSL2 or native Linux, `make inference-ts` works directly.

### 5.3 Manual Execution Pipeline

**1. Compilation**
```bash
make init   # Initializes CMake architecture
make build  # Compiles `mlp` and `infer_cpp`
```

**2. Data Synthesis**
Generates synthetic `classification.csv` and `regression.csv` for experimental testing.
```bash
make seed
```

**3. Model Training (Single Node Classification Example)**
Execute the custom engine to train a binary classifier over 200 epochs, applying Z-score standardization, and exporting the topological graph to `model.onnx`.
```bash
cd build
./mlp --input-dataset=../data/classification.csv \
      --column-label=class \
      --column-train=f1,f2,f3 \
      --alg-type=classifier \
      --option-std \
      --hyperp-epochs=200 \
      --output-onnx=model.onnx
```

**4. Model Training (GPU Hardware Acceleration Example)**
By leveraging the OpenCL integration, you can offload the $O(N^3)$ Matrix Multiplication bottleneck to your system's GPU (or Intel HD graphics) by passing the `--compute-type=GPU` flag. If no GPU is found, the engine gracefully falls back to CPU computation.
```bash
make train-gpu
```

**5. Model Training (Distributed Parameter Server Architecture)**
We have implemented a fully custom TCP-based distributed training module. A Master node hosts the global weights, while multiple Worker nodes compute local gradients and push them asynchronously. 
```bash
make train-dist
```
*(This command runs `test_dist.sh`, spawning 1 Master and 3 Workers in the background, then automatically shuts down the Master after training to export the `dist_model.onnx` graph.)*

**6. Cross-Language Inference**
Verify the ONNX graph's portability by running predictions on a dummy tensor $[1.0, 2.0, 3.0]$ across five different language paradigms. All outputs are numerically identical.

| Make Target | Language | Runtime |
|---|---|---|
| `make inference-py` | Python | `onnxruntime` (pip) |
| `make inference-go` | Go | `yalue/onnxruntime_go` |
| `make inference-cpp` | C++ | ONNX Runtime C++ API |
| `make inference-c` | C | ONNX Runtime C API (no C++ wrappers) |
| `make inference-ts` ⚠️ | TypeScript | `onnxruntime-node` (npm) |

> ⚠️ **TypeScript on WSL1**: Run `make inference-ts` from **Windows PowerShell** instead of WSL.
> On WSL2 / native Linux, it works directly via `make inference-ts`.

## 6. Deployment

The engine supports four deployment modes, all defined under the `deploy/` directory.

### 6.1 Docker (Local Multi-Node Cluster)
A multi-stage `Dockerfile` produces a lean runtime image. `docker-compose.yml` orchestrates the full cluster locally.
```bash
# Build image and start 1 master + 3 workers
make deploy-docker

# Scale to a different number of workers on the fly
make deploy-docker-scale WORKERS=5

# Tear down
make deploy-docker-down
```

### 6.2 Systemd (Bare-Metal / VM)
Template unit files (`mlp-master.service` and `mlp-worker@.service`) are provided for direct installation on Linux hosts.
```bash
# Install and start master + 3 workers
make deploy-systemd

# Add more worker instances (e.g. 5 total)
make deploy-systemd-workers N=5

# Check status
sudo systemctl status 'mlp-worker@*'
journalctl -u mlp-master -f
```

### 6.3 Kubernetes (Plain Manifests)
Pure YAML manifests under `deploy/k8s/` deploy the cluster to any Kubernetes environment (Minikube, EKS, GKE, etc.).
```bash
# Deploy cluster (defaults: 3 workers)
make deploy-k8s

# Scale workers live
make deploy-k8s-scale WORKERS=5

# Check pods
kubectl get pods -n mlp-engine

# Tear down
make deploy-k8s-down
```

### 6.4 Helm (Configurable Chart)
The Helm chart under `deploy/helm/mlp-engine/` allows full parameterization via `values.yaml` or `--set` flags.
```bash
# Install with defaults (3 workers, CPU, 200 epochs)
make deploy-helm

# Install with custom values
helm install mlp-engine ./deploy/helm/mlp-engine \
  --set worker.replicaCount=5 \
  --set training.epochs=500 \
  --set training.computeType=GPU

# Upgrade and rescale workers
make deploy-helm-scale WORKERS=8

# Uninstall
make deploy-helm-down
```

**Worker count is the primary scaling knob across all deployment methods:**

| Deployment | Scale Command |
|---|---|
| Docker Compose | `make deploy-docker-scale WORKERS=N` |
| Systemd | `make deploy-systemd-workers N=N` |
| Kubernetes | `make deploy-k8s-scale WORKERS=N` |
| Helm | `make deploy-helm-scale WORKERS=N` |

## 7. Conclusion
By meticulously defining the linear algebra matrices, activation derivations, and backpropagation chains in native C++, this implementation serves as both a high-performance educational tool and a lightweight standalone neural engine. With support for **GPU acceleration** (OpenCL), **distributed training** (TCP Parameter Server), **cross-language ONNX inference** across 5 languages (Python, Go, C++, C, TypeScript), and **four deployment strategies** (Docker, systemd, Kubernetes, Helm), this project demonstrates the full ML lifecycle from raw mathematics to production-ready deployment — without any high-level ML framework dependency.
