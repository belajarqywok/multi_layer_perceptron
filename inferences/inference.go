package main

import (
	"fmt"
	"log"
	"os"

	"github.com/yalue/onnxruntime_go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run infer.go <model.onnx>")
		os.Exit(1)
	}
	modelPath := os.Args[1]

	// The ONNX runtime shared library needs to be set.
	// `onnxruntime_go` uses CGO to link. We should set the shared library path.
	// For Ubuntu WSL it usually requires downloading the shared library.
	// `yalue/onnxruntime_go` provides a way to set the shared library path.
	onnxruntime_go.SetSharedLibraryPath("/mnt/c/Users/pc/Documents/multi_layer_perceptron/lib/onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so")

	err := onnxruntime_go.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Error initializing ONNX runtime: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	fmt.Printf("Loading model from %s...\n", modelPath)

	// Create input tensor (1 batch, 3 features)
	inputData := []float32{1.0, 2.0, 3.0}
	inputShape := []int64{1, 3}
	inputTensor, err := onnxruntime_go.NewTensor(inputShape, inputData)
	if err != nil {
		log.Fatalf("Error creating input tensor: %v", err)
	}
	defer inputTensor.Destroy()

	// Create output tensor (1 batch, 1 label)
	outputData := make([]float32, 1)
	outputShape := []int64{1, 1}
	outputTensor, err := onnxruntime_go.NewTensor(outputShape, outputData)
	if err != nil {
		log.Fatalf("Error creating output tensor: %v", err)
	}
	defer outputTensor.Destroy()

	// Create session
	session, err := onnxruntime_go.NewAdvancedSession(modelPath,
		[]string{"X"}, []string{"Y"},
		[]onnxruntime_go.ArbitraryTensor{inputTensor},
		[]onnxruntime_go.ArbitraryTensor{outputTensor},
		nil)
	if err != nil {
		log.Fatalf("Error creating session: %v", err)
	}
	defer session.Destroy()

	fmt.Printf("Input Features: %v\n", inputData)

	err = session.Run()
	if err != nil {
		log.Fatalf("Error running session: %v", err)
	}

	out := outputTensor.GetData()
	fmt.Printf("Output: %v\n", out)
}
