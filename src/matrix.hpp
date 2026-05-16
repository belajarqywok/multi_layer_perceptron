#pragma once
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <functional>

/**
 * @class Matrix
 * @brief A lightweight, standalone matrix mathematics library designed for Neural Networks.
 * 
 * **Why use a 1D array instead of 2D (std::vector<std::vector<double>>)?**
 * In modern CPUs, fetching data from RAM is very slow compared to processing it. 
 * By flattening a 2D grid (Rows x Columns) into a single 1D array, we ensure all the numbers 
 * sit right next to each other in the physical memory (contiguous memory). 
 * This maximizes "Cache Locality," meaning the CPU can load chunks of the matrix into its 
 * super-fast L1/L2 cache, making matrix multiplication drastically faster.
 */
class Matrix {
public:
    size_t rows;
    size_t cols;
    std::vector<double> data; // The flattened 1D array representing the 2D grid
    
    // Compute routing flag
    static std::string global_compute_type;

    /**
     * @brief Constructs a Matrix filled with 0.0
     * @param r Number of rows
     * @param c Number of columns
     */
    Matrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0.0) {}

    /**
     * @brief Access element at (row, col) with bounds calculation.
     */
    double& at(size_t r, size_t c) {
        return data[r * cols + c];
    }

    /**
     * @brief Read-only access element at (row, col).
     */
    const double& at(size_t r, size_t c) const {
        return data[r * cols + c];
    }

    /**
     * @brief Randomizes matrix weights/biases using a Uniform distribution.
     * Often used during the initialization phase before training.
     */
    void randomize(double min = -0.5, double max = 0.5, int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(min, max);
        for (auto& val : data) {
            val = dist(gen);
        }
    }

    /**
     * @brief Performs standard Matrix Multiplication (Dot Product).
     * 
     * **Concept:** If Matrix A is (M x K) and Matrix B is (K x N), the result is (M x N).
     * To find the value at row `i` and column `j` in the new matrix, we multiply the `i`-th row of A
     * with the `j`-th column of B element-by-element and sum them up.
     * 
     * Calculates C = A * B. The Time Complexity is O(n^3).
     * Automatically routes to the GPU via OpenCL if global_compute_type is set to "GPU".
     */
    static Matrix dot(const Matrix& a, const Matrix& b);

    /**
     * @brief Transposes the matrix (flips rows and columns).
     */
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at(j, i) = at(i, j);
            }
        }
        return result;
    }

    /**
     * @brief Broadcasting addition of a row vector (bias) to every row of the matrix.
     */
    void addBias(const Matrix& bias) {
        if (bias.rows != 1 || bias.cols != cols) {
            throw std::invalid_argument("Bias dimension mismatch.");
        }
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                at(i, j) += bias.at(0, j);
            }
        }
    }

    /**
     * @brief Applies an activation function (like ReLU or Sigmoid) to every element.
     * 
     * **Why Activation Functions?**
     * Without them, a neural network is just a giant linear equation (y = mx + c) no matter 
     * how many layers it has. Applying a non-linear function like Sigmoid allows the network 
     * to learn complex, non-straight patterns (like curves and circles).
     */
    Matrix apply(std::function<double(double)> func) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = func(data[i]);
        }
        return result;
    }

    /**
     * @brief Performs the Hadamard Product (element-wise multiplication).
     */
    Matrix elementwiseMul(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Elementwise multiplication dimension mismatch.");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }
    
    /**
     * @brief Subtracts another matrix element-wise (used for delta/error calculations).
     */
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Subtraction dimension mismatch.");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }

    /**
     * @brief Scales the entire matrix by a scalar value (used during Gradient application).
     */
    void multiplyScalar(double scalar) {
        for (auto& val : data) {
            val *= scalar;
        }
    }
};
