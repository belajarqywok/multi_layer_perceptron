#pragma once
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include "matrix.hpp" // For Matrix dimensions

/**
 * @class OpenCLUtils
 * @brief Singleton wrapper to manage OpenCL Context, Command Queue, and Kernels for Hardware Acceleration.
 */
class OpenCLUtils {
private:
    inline static cl_context context = nullptr;
    inline static cl_command_queue queue = nullptr;
    inline static cl_program program = nullptr;
    inline static cl_kernel matmul_kernel = nullptr;
    inline static bool initialized = false;

    // OpenCL C99 Kernel Code for Matrix Multiplication
    static constexpr const char* kernel_source = R"(
    __kernel void matmul(
        const int M, const int N, const int K,
        __global const double* A,
        __global const double* B,
        __global double* C) 
    {
        // Thread identifiers
        const int row = get_global_id(0); // 0..M-1
        const int col = get_global_id(1); // 0..N-1
        
        if (row < M && col < N) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    )";

public:
    /**
     * @brief Initializes the OpenCL context, discovers the best available device (GPU preferred),
     * and compiles the embedded C99 Matrix Multiplication Kernel.
     * @return true if OpenCL was successfully initialized, false if no compatible device/driver is found.
     */
    static bool initialize() {
        if (initialized) return true;

        cl_uint platform_count;
        clGetPlatformIDs(0, nullptr, &platform_count);
        if (platform_count == 0) {
            std::cerr << "[OpenCL] No OpenCL platforms found. Falling back to CPU." << std::endl;
            return false;
        }

        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);

        cl_device_id device_id = nullptr;
        for (auto p : platforms) {
            cl_uint device_count;
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
            if (device_count > 0) {
                clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
                break;
            }
        }

        if (!device_id) {
            std::cout << "[OpenCL] No GPU found. Checking for CPU devices..." << std::endl;
            // Fallback to CPU device if GPU is not available
            for (auto p : platforms) {
                cl_uint device_count;
                clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 0, nullptr, &device_count);
                if (device_count > 0) {
                    clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 1, &device_id, nullptr);
                    break;
                }
            }
        }

        if (!device_id) {
            std::cerr << "[OpenCL] No OpenCL capable device found. Falling back to CPU Engine." << std::endl;
            return false;
        }

        char device_name[128];
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, 128, device_name, nullptr);
        std::cout << "[OpenCL] Initialized successfully. Using device: " << device_name << std::endl;

        cl_int err;
        context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
        queue = clCreateCommandQueue(context, device_id, 0, &err);

        const char* strings[1] = { kernel_source };
        program = clCreateProgramWithSource(context, 1, strings, nullptr, &err);
        err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string build_log(log_size, ' ');
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], nullptr);
            std::cerr << "[OpenCL] Kernel build failed: " << build_log << std::endl;
            return false;
        }

        matmul_kernel = clCreateKernel(program, "matmul", &err);
        initialized = true;
        return true;
    }

    static bool is_initialized() {
        return initialized;
    }

    /**
     * @brief Performs Hardware-Accelerated Matrix Multiplication (Dot Product) on the GPU.
     * 
     * Handles the memory transfer from CPU Host to GPU Device, enqueues the parallel execution kernel,
     * and reads the result back to the CPU Host.
     * 
     * @param A First matrix
     * @param B Second matrix
     * @return Resulting Matrix C = A * B
     */
    static Matrix dot_gpu(const Matrix& A, const Matrix& B) {
        if (!initialized) {
            throw std::runtime_error("OpenCL not initialized");
        }

        int M = A.rows;
        int K = A.cols; // == B.rows
        int N = B.cols;

        cl_int err;
        cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * A.data.size(), (void*)A.data.data(), &err);
        cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * B.data.size(), (void*)B.data.data(), &err);
        cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * M * N, nullptr, &err);

        clSetKernelArg(matmul_kernel, 0, sizeof(int), &M);
        clSetKernelArg(matmul_kernel, 1, sizeof(int), &N);
        clSetKernelArg(matmul_kernel, 2, sizeof(int), &K);
        clSetKernelArg(matmul_kernel, 3, sizeof(cl_mem), &bufA);
        clSetKernelArg(matmul_kernel, 4, sizeof(cl_mem), &bufB);
        clSetKernelArg(matmul_kernel, 5, sizeof(cl_mem), &bufC);

        size_t global_work_size[2] = { (size_t)M, (size_t)N };
        
        err = clEnqueueNDRangeKernel(queue, matmul_kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue OpenCL matmul kernel");
        }

        Matrix C(M, N);
        clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeof(double) * M * N, C.data.data(), 0, nullptr, nullptr);

        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);

        return C;
    }

    /**
     * @brief Safely releases all OpenCL memory handles, command queues, and contexts.
     * Must be called before program termination to prevent GPU memory leaks.
     */
    static void cleanup() {
        if (initialized) {
            clReleaseKernel(matmul_kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
            initialized = false;
        }
    }
};
