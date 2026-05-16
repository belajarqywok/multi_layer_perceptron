#pragma once
#include "matrix.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

/**
 * @class MLP
 * @brief Multi-Layer Perceptron (Neural Network) core engine.
 *
 * Implements Forward Propagation, Backpropagation (Stochastic/Batch Gradient Descent),
 * and parameter flattening for Distributed Training serialization.
 */
class MLP {
public:
    std::vector<Matrix> weights; /// Array of weight matrices for each layer
    std::vector<Matrix> biases;  /// Array of bias row-vectors for each layer
    std::string alg_type;        /// "classifier" (Sigmoid output) or "regressor" (Linear output)
    double learning_rate = 0.01; /// Step size for gradient descent

    MLP(int input_size, int hidden_size, int num_hidden_layers, int output_size, const std::string& type, int seed) 
        : alg_type(type) {
        
        std::vector<int> layer_sizes;
        layer_sizes.push_back(input_size);
        for (int i = 0; i < num_hidden_layers; ++i) {
            layer_sizes.push_back(hidden_size);
        }
        layer_sizes.push_back(output_size);

        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            Matrix w(layer_sizes[i], layer_sizes[i+1]);
            // He initialization approximation
            double limit = std::sqrt(2.0 / layer_sizes[i]);
            w.randomize(-limit, limit, seed + i);
            weights.push_back(w);

            Matrix b(1, layer_sizes[i+1]);
            b.randomize(-0.1, 0.1, seed + i + 100);
            biases.push_back(b);
        }
    }

    // --- Activation Functions ---
    // Note: ReLU introduces non-linearity and solves the vanishing gradient problem.
    static double relu(double x) { return x > 0 ? x : 0; }
    static double relu_deriv(double x) { return x > 0 ? 1 : 0; } // Derivative (Heaviside step)

    // Sigmoid is used for Binary Classification output (squashes value to 0..1)
    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    static double sigmoid_deriv(double x) { double s = sigmoid(x); return s * (1 - s); } // s * (1 - s)

    // Linear activation used for Regression output
    static double linear(double x) { return x; }
    static double linear_deriv(double x) { return 1; }

    /**
     * @brief Performs a forward pass through the network.
     * 
     * **Concept (Forward Propagation):**
     * The input data `X` is passed through the network layer by layer. 
     * For each layer, we multiply the incoming data by the weights (Matrix::dot), add the bias, 
     * and apply an activation function to get the output `A` (Activations). 
     * This output `A` becomes the input for the next layer.
     * 
     * @param X Input matrix (batch size x input features).
     * @return A vector of matrices storing the post-activation output of each layer.
     *         The last element is the final prediction output.
     */
    std::vector<Matrix> forward(const Matrix& X) {
        std::vector<Matrix> activations;
        activations.push_back(X); // A^[0] is the input data

        Matrix current_a = X;

        for (size_t i = 0; i < weights.size(); ++i) {
            Matrix z = Matrix::dot(current_a, weights[i]);
            z.addBias(biases[i]);

            if (i == weights.size() - 1) {
                // Output layer
                if (alg_type == "classifier") {
                    current_a = z.apply(sigmoid);
                } else {
                    current_a = z.apply(linear);
                }
            } else {
                // Hidden layer
                current_a = z.apply(relu);
            }
            activations.push_back(current_a);
        }
        return activations;
    }

    /**
     * @brief Flattens all weights and biases into a 1D vector.
     * Used exclusively for serializing parameters over the network in Distributed Training.
     */
    std::vector<double> flatten_parameters() const {
        std::vector<double> flat;
        for (const auto& w : weights) {
            flat.insert(flat.end(), w.data.begin(), w.data.end());
        }
        for (const auto& b : biases) {
            flat.insert(flat.end(), b.data.begin(), b.data.end());
        }
        return flat;
    }

    void unflatten_parameters(const std::vector<double>& flat) {
        size_t idx = 0;
        for (auto& w : weights) {
            std::copy(flat.begin() + idx, flat.begin() + idx + w.data.size(), w.data.begin());
            idx += w.data.size();
        }
        for (auto& b : biases) {
            std::copy(flat.begin() + idx, flat.begin() + idx + b.data.size(), b.data.begin());
            idx += b.data.size();
        }
    }

    /**
     * @brief Calculates gradients for all weights and biases using Backpropagation.
     * 
     * **Concept (Backpropagation & The Chain Rule):**
     * 1. We start by calculating how "wrong" our final prediction was compared to the actual Truth (Y). This is our Error/Delta.
     * 2. We then walk backwards (from the Output layer to the Input layer).
     * 3. At each layer, we calculate how much the Weights and Biases contributed to that Error (the Gradient).
     * 4. We pass the remaining Error backward to the previous layer so they can calculate their gradients too.
     * 
     * @param X Input features matrix
     * @param Y Ground truth labels matrix
     * @param out_loss Reference passed to store the computed Mean Squared Error
     * @return Flattened 1D vector containing all computed gradients (Delta W, Delta B).
     */
    std::vector<double> compute_gradients(const Matrix& X, const Matrix& Y, double& out_loss) {
        std::vector<Matrix> activations = forward(X);
        std::vector<Matrix> delta_w(weights.size(), Matrix(0, 0));
        std::vector<Matrix> delta_b(biases.size(), Matrix(0, 0));

        // Step 1: Calculate output error (delta).
        // Since we pair Sigmoid with Binary Cross-Entropy, or Linear with MSE, 
        // the mathematical derivative beautifully simplifies to: A - Y
        Matrix current_delta = activations.back() - Y; 

        // Step 2: Backpropagate through hidden layers (Chain Rule)
        for (int i = weights.size() - 1; i >= 0; --i) {
            Matrix a_prev = activations[i];
            
            // Gradient for Weights: dW = A_prev^T * Delta
            delta_w[i] = Matrix::dot(a_prev.transpose(), current_delta);
            
            // Gradient for Biases: dB = Sum over batch of Delta
            Matrix db(1, current_delta.cols);
            for (size_t c = 0; c < current_delta.cols; ++c) {
                double sum = 0;
                for (size_t r = 0; r < current_delta.rows; ++r) { sum += current_delta.at(r, c); }
                db.at(0, c) = sum;
            }
            delta_b[i] = db;

            // Step 3: Propagate Delta backwards to the preceding layer
            if (i > 0) {
                // Reconstruct pre-activation Z for the previous layer
                Matrix z_prev = Matrix::dot(activations[i-1], weights[i-1]); 
                z_prev.addBias(biases[i-1]);
                
                // Calculate derivative of the activation function (f'(Z))
                Matrix d_activation = z_prev.apply(relu_deriv);
                
                // Calculate new Delta: (W^T * Delta) element-wise-multiplied by f'(Z)
                Matrix w_T = weights[i].transpose();
                Matrix propagated_error = Matrix::dot(current_delta, w_T);
                current_delta = propagated_error.elementwiseMul(d_activation);
            }
        }

        std::vector<double> flat_grads;
        for (size_t i = 0; i < weights.size(); ++i) {
            for(auto val : delta_w[i].data) flat_grads.push_back(val / X.rows);
        }
        for (size_t i = 0; i < biases.size(); ++i) {
            for(auto val : delta_b[i].data) flat_grads.push_back(val / X.rows);
        }

        out_loss = 0.0;
        Matrix out = activations.back();
        for (size_t r = 0; r < out.rows; ++r) {
            for (size_t c = 0; c < out.cols; ++c) {
                double diff = out.at(r, c) - Y.at(r, c);
                out_loss += diff * diff;
            }
        }
        out_loss /= X.rows;

        return flat_grads;
    }

    void apply_gradients(const std::vector<double>& flat_grads) {
        size_t idx = 0;
        for (auto& w : weights) {
            for(size_t i=0; i<w.data.size(); ++i) w.data[i] -= learning_rate * flat_grads[idx++];
        }
        for (auto& b : biases) {
            for(size_t i=0; i<b.data.size(); ++i) b.data[i] -= learning_rate * flat_grads[idx++];
        }
    }

    /**
     * @brief Trains the Neural Network using Batch Gradient Descent.
     * 
     * **Concept:**
     * Training is basically a loop: 
     * 1. Guess the answer (Forward Pass inside compute_gradients)
     * 2. See how wrong the guess was (Calculate Loss)
     * 3. Figure out which way to tweak the weights to be less wrong next time (Calculate Gradients)
     * 4. Actually tweak the weights (Apply Gradients by subtracting them scaled by the Learning Rate)
     */
    void train(const Matrix& X, const Matrix& Y, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double loss;
            std::vector<double> grads = compute_gradients(X, Y, loss);
            apply_gradients(grads);

            if ((epoch + 1) % 10 == 0 || epoch == 0) {
                std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << loss << std::endl;
            }
        }
    }
};
