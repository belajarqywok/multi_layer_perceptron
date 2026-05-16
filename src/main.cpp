#include <iostream>
#include "cli_parser.hpp"
#include "csv_reader.hpp"
#include "mlp.hpp"
#include "onnx_exporter.hpp"
#include "network.hpp"
#include "opencl_utils.hpp"
#include <thread>
#include <chrono>

std::string Matrix::global_compute_type = "CPU";

Matrix Matrix::dot(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dot product dimension mismatch.");
    }
    
    if (Matrix::global_compute_type == "GPU" && OpenCLUtils::is_initialized()) {
        return OpenCLUtils::dot_gpu(a, b);
    }
    
    // CPU Fallback implementation
    Matrix result(a.rows, b.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < b.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < a.cols; ++k) {
                sum += a.at(i, k) * b.at(k, j);
            }
            result.at(i, j) = sum;
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "Starting MLP Initialization..." << std::endl;
        
        CLIArguments args = CLIParser::parse(argc, argv);

        Matrix::global_compute_type = args.compute_type;
        if (args.compute_type == "GPU") {
            if (!OpenCLUtils::initialize()) {
                std::cerr << "[Warning] OpenCL initialization failed. Falling back to CPU compute mode." << std::endl;
                Matrix::global_compute_type = "CPU";
            }
        }

        
        if (args.input_dataset.empty()) {
            std::cerr << "Error: --input-dataset is required." << std::endl;
            return 1;
        }

        std::cout << "Configuration:" << std::endl;
        std::cout << "- Dataset: " << args.input_dataset << std::endl;
        std::cout << "- Algorithm Type: " << args.alg_type << std::endl;
        std::cout << "- Hyperparams: Layers=" << args.hyperp_layers 
                  << ", Epochs=" << args.hyperp_epochs 
                  << ", Seed=" << args.hyperp_random_state << std::endl;

        std::cout << "\nLoading dataset..." << std::endl;
        Dataset dataset = CSVReader::load(args);

        std::cout << "Dataset loaded successfully!" << std::endl;
        if (dataset.features.empty()) {
            std::cerr << "Dataset is empty!" << std::endl;
            return 1;
        }

        int num_samples = dataset.features.size();
        int input_size = dataset.features[0].size();
        int output_size = dataset.labels[0].size();

        std::cout << "Number of samples: " << num_samples << std::endl;
        std::cout << "Number of features: " << input_size << std::endl;
        std::cout << "Number of labels: " << output_size << std::endl;

        // Convert dataset to Matrix
        Matrix X(num_samples, input_size);
        Matrix Y(num_samples, output_size);

        for (int r = 0; r < num_samples; ++r) {
            for (int c = 0; c < input_size; ++c) {
                X.at(r, c) = dataset.features[r][c];
            }
            for (int c = 0; c < output_size; ++c) {
                Y.at(r, c) = dataset.labels[r][c];
            }
        }

        // Initialize MLP
        // Defaulting hidden size to 16 since it wasn't parameterized
        int hidden_size = 16;
        MLP mlp(input_size, hidden_size, args.hyperp_layers, output_size, args.alg_type, args.hyperp_random_state);
        
        std::cout << "\nTraining MLP..." << std::endl;
        
        if (args.type_training == "distributed") {
            if (args.dist_role == "master") {
                std::vector<double> init_weights = mlp.flatten_parameters();
                ParameterServer server(init_weights, mlp.learning_rate);
                server.start(args.dist_master_port);
                
                std::vector<double> final_weights = server.get_final_weights();
                mlp.unflatten_parameters(final_weights);
            } else if (args.dist_role == "worker") {
                ParameterClient client;
                int retries = 10;
                while(retries-- > 0 && !client.connect_to_master(args.dist_master_addr, args.dist_master_port)) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                if (retries <= 0) {
                    std::cerr << "Worker failed to connect to master." << std::endl;
                    return 1;
                }
                
                for (int epoch = 0; epoch < args.hyperp_epochs; ++epoch) {
                    std::vector<double> weights;
                    if (!client.fetch_weights(weights)) break;
                    mlp.unflatten_parameters(weights);

                    double loss;
                    std::vector<double> grads = mlp.compute_gradients(X, Y, loss);
                    if (!client.push_gradients(grads)) break;

                    if ((epoch + 1) % 10 == 0 || epoch == 0) {
                        std::cout << "[Worker] Epoch " << epoch + 1 << "/" << args.hyperp_epochs << " - Loss: " << loss << std::endl;
                    }
                }
                client.disconnect();
                args.output_onnx = ""; // Only master exports
            } else if (args.dist_role == "shutdown") {
                ParameterClient client;
                if(client.connect_to_master(args.dist_master_addr, args.dist_master_port)) {
                    client.send_shutdown();
                    client.disconnect();
                }
                return 0; // Exit successfully
            }
        } else {
            mlp.train(X, Y, args.hyperp_epochs);
        }

        std::cout << "\nTraining completed!" << std::endl;

        if (!args.output_onnx.empty()) {
            std::cout << "\nExporting model to ONNX..." << std::endl;
            ONNXExporter::export_model(mlp, args.output_onnx, input_size, output_size);
        }
        OpenCLUtils::cleanup();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        OpenCLUtils::cleanup();
        return 1;
    }
}
