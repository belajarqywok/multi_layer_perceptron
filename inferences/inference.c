#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: ./inference_c <path_to_model.onnx>\n");
        return 1;
    }

    const char* model_path = argv[1];
    printf("Loading model from %s...\n", model_path);

    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        printf("Failed to initialize ONNX Runtime C API\n");
        return 1;
    }

    OrtEnv* env;
    g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "inference_c", &env);

    OrtSessionOptions* session_options;
    g_ort->CreateSessionOptions(&session_options);

    OrtSession* session;
    g_ort->CreateSession(env, model_path, session_options, &session);

    OrtAllocator* allocator;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);

    char* input_name;
    g_ort->SessionGetInputName(session, 0, allocator, &input_name);

    printf("Expected Input Name: %s\n", input_name);
    printf("Input Features: [1.0, 2.0, 3.0]\n");

    const char* input_names[] = { input_name };
    
    // Hardcoded input
    float input_tensor_values[] = { 1.0f, 2.0f, 3.0f };
    int64_t input_node_dims[] = { 1, 3 };
    size_t input_tensor_size = 3 * sizeof(float);

    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

    OrtValue* input_tensor = NULL;
    g_ort->CreateTensorWithDataAsOrtValue(
        memory_info, input_tensor_values, input_tensor_size, 
        input_node_dims, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    char* output_name;
    g_ort->SessionGetOutputName(session, 0, allocator, &output_name);
    const char* output_names[] = { output_name };

    OrtValue* output_tensor = NULL;
    g_ort->Run(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);

    float* float_arr;
    g_ort->GetTensorMutableData(output_tensor, (void**)&float_arr);
    printf("Output: [%f]\n", float_arr[0]);

    // Cleanup
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    allocator->Free(allocator, input_name);
    allocator->Free(allocator, output_name);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    return 0;
}
