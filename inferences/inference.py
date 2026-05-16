import onnxruntime as ort
import numpy as np
import sys

def infer(model_path):
    print(f"Loading model from {model_path}...")
    session = ort.InferenceSession(model_path)
    
    # Get input name and shape
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f"Expected Input Name: {input_name}")
    print(f"Expected Input Shape: {input_shape}")

    # Prepare a dummy input for testing (Classification expects 3 features)
    # Batch size 1, 3 features.
    X = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    print(f"Input Features: {X.tolist()}")

    # Run inference
    result = session.run(None, {input_name: X})
    
    print(f"Output: {result[0].tolist()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer.py <model.onnx>")
        sys.exit(1)
    infer(sys.argv[1])
