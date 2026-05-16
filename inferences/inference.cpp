#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::cout << "Loading model from " << model_path << "..." << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    // Create Session
    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // Get input info
    size_t num_input_nodes = session.GetInputCount();
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    std::string input_name = input_name_ptr.get();

    // Get output info
    size_t num_output_nodes = session.GetOutputCount();
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    std::string output_name = output_name_ptr.get();

    std::cout << "Expected Input Name: " << input_name << std::endl;

    // Prepare input data
    std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
    std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
    std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_node_dims.data(), input_node_dims.size());

    // Run inference
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Print output
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

    return 0;
}


// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }



// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }



// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }




// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }




// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }






// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }






// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }






// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }







// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }





// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }






// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }






// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }







// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }






// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }






// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }





// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: ./infer_cpp <model.onnx>" << std::endl;
//         return 1;
//     }

//     std::string model_path = argv[1];
//     std::cout << "Loading model from " << model_path << "..." << std::endl;

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MLP_Inference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Create Session
//     Ort::Session session(env, model_path.c_str(), session_options);

//     Ort::AllocatorWithDefaultOptions allocator;

//     // Get input info
//     size_t num_input_nodes = session.GetInputCount();
//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     std::string input_name = input_name_ptr.get();

//     // Get output info
//     size_t num_output_nodes = session.GetOutputCount();
//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     std::string output_name = output_name_ptr.get();

//     std::cout << "Expected Input Name: " << input_name << std::endl;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {1.0f, 2.0f, 3.0f};
//     std::vector<int64_t> input_node_dims = {1, 3}; // batch_size=1, features=3
    
//     std::cout << "Input Features: [1.0, 2.0, 3.0]" << std::endl;

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_node_dims.data(), input_node_dims.size());

//     // Run inference
//     const char* input_names[] = {input_name.c_str()};
//     const char* output_names[] = {output_name.c_str()};

//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Print output
//     float* floatarr = output_tensors.front().GetTensorMutableData<float>();
//     std::cout << "Output: [" << floatarr[0] << "]" << std::endl;

//     return 0;
// }
