#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>

struct CLIArguments {
    std::string input_dataset;
    std::vector<std::string> column_labels;
    std::vector<std::string> column_train;
    std::string output_onnx;
    int hyperp_layers = 1;
    int hyperp_epochs = 100;
    int hyperp_random_state = 42;
    std::string alg_type = "classifier";
    bool option_cleaning = false;
    bool option_std = false;
    bool option_remove_duplication = false;

    // Distributed Training Flags
    std::string type_training = "single"; // "single" or "distributed"
    std::string dist_role = "worker"; // "master" or "worker"
    std::string dist_master_addr = "127.0.0.1";
    int dist_master_port = 8080;

    // Hardware Acceleration Flags
    std::string compute_type = "CPU"; // "CPU" or "GPU"
};

class CLIParser {
public:
    static CLIArguments parse(int argc, char* argv[]) {
        CLIArguments args;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg.find("--input-dataset=") == 0) {
                args.input_dataset = arg.substr(16);
            } else if (arg.find("--column-label=") == 0) {
                args.column_labels = split(arg.substr(15), ',');
            } else if (arg.find("--column-train=") == 0) {
                args.column_train = split(arg.substr(15), ',');
            } else if (arg.find("--output-onnx=") == 0) {
                args.output_onnx = arg.substr(14);
            } else if (arg.find("--hyperp-layers=") == 0) {
                args.hyperp_layers = std::stoi(arg.substr(16));
            } else if (arg.find("--hyperp-epochs=") == 0) {
                args.hyperp_epochs = std::stoi(arg.substr(16));
            } else if (arg.find("--hyperp-random-state=") == 0) {
                args.hyperp_random_state = std::stoi(arg.substr(22));
            } else if (arg.find("--alg-type=") == 0) {
                args.alg_type = arg.substr(11);
            } else if (arg == "--option-cleaning") {
                args.option_cleaning = true;
            } else if (arg == "--option-std") {
                args.option_std = true;
            } else if (arg == "--option-remove-duplication") {
                args.option_remove_duplication = true;
            } else if (arg.find("--type-training=") == 0) {
                args.type_training = arg.substr(16);
            } else if (arg.find("--dist-role=") == 0) {
                args.dist_role = arg.substr(12);
            } else if (arg.find("--dist-master-addr=") == 0) {
                args.dist_master_addr = arg.substr(19);
            } else if (arg.find("--dist-master-port=") == 0) {
                args.dist_master_port = std::stoi(arg.substr(19));
            } else if (arg.find("--compute-type=") == 0) {
                args.compute_type = arg.substr(15);
            }
        }
        return args;
    }

private:
    static std::vector<std::string> split(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }
};
