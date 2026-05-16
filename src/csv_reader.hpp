#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include "cli_parser.hpp"

/**
 * @struct Dataset
 * @brief Holds parsed CSV data, splitting it into training features (X) and ground truth labels (Y).
 */
struct Dataset {
    std::vector<std::vector<double>> features;  /// The input parameters (X)
    std::vector<std::vector<double>> labels;    /// The target output (Y)
    std::vector<std::string> feature_names;     /// Column names mapped to features
    std::vector<std::string> label_names;       /// Column names mapped to labels
};

/**
 * @class CSVReader
 * @brief Parses CSV files, extracts specific columns, and performs data preprocessing.
 */
class CSVReader {
public:
    /**
     * @brief Loads and parses a CSV dataset based on CLI arguments.
     * Extracts relevant columns, drops missing data, handles deduplication, and applies Standardization.
     */
    static Dataset load(const CLIArguments& args) {
        Dataset dataset;
        std::ifstream file(args.input_dataset);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open dataset file: " + args.input_dataset);
        }

        std::string line;
        if (!std::getline(file, line)) {
            throw std::runtime_error("Dataset is empty.");
        }

        // Parse header
        std::vector<std::string> header = split(line, ',');
        
        // Find indices for features and labels
        std::vector<int> feature_indices;
        for (const auto& col : args.column_train) {
            auto it = std::find(header.begin(), header.end(), col);
            if (it != header.end()) {
                feature_indices.push_back(std::distance(header.begin(), it));
                dataset.feature_names.push_back(col);
            } else {
                throw std::runtime_error("Feature column not found: " + col);
            }
        }

        std::vector<int> label_indices;
        for (const auto& col : args.column_labels) {
            auto it = std::find(header.begin(), header.end(), col);
            if (it != header.end()) {
                label_indices.push_back(std::distance(header.begin(), it));
                dataset.label_names.push_back(col);
            } else {
                throw std::runtime_error("Label column not found: " + col);
            }
        }

        std::set<std::string> unique_rows;

        while (std::getline(file, line)) {
            if (line.empty()) continue;

            if (args.option_remove_duplication) {
                if (unique_rows.find(line) != unique_rows.end()) {
                    continue; // Skip duplicate
                }
                unique_rows.insert(line);
            }

            std::vector<std::string> tokens = split(line, ',');
            
            // Check for missing values (simple check for empty tokens)
            bool has_missing = false;
            for (const auto& token : tokens) {
                if (token.empty()) {
                    has_missing = true;
                    break;
                }
            }

            if (args.option_cleaning && has_missing) {
                continue; // Skip row with missing values
            }

            if (tokens.size() != header.size()) {
                continue; // Skip malformed rows
            }

            try {
                std::vector<double> row_features;
                for (int idx : feature_indices) {
                    row_features.push_back(std::stod(tokens[idx]));
                }
                
                std::vector<double> row_labels;
                for (int idx : label_indices) {
                    row_labels.push_back(std::stod(tokens[idx]));
                }

                dataset.features.push_back(row_features);
                dataset.labels.push_back(row_labels);
            } catch (const std::exception& e) {
                if (!args.option_cleaning) {
                    // If not cleaning, maybe we should throw or just ignore?
                    // Usually we ignore or throw. Let's skip the bad row for robustness.
                }
            }
        }

        // Basic Standardization
        if (args.option_std) {
            if (!dataset.features.empty()) standardize(dataset.features);
            if (!dataset.labels.empty() && args.alg_type == "regressor") standardize(dataset.labels);
        }

        return dataset;
    }

private:
    static std::vector<std::string> split(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            // Trim \r
            if (!token.empty() && token.back() == '\r') {
                token.pop_back();
            }
            tokens.push_back(token);
        }
        // Handle trailing empty columns e.g. "a,b,"
        if (!s.empty() && s.back() == delimiter) {
            tokens.push_back("");
        }
        return tokens;
    }

    /**
     * @brief Applies Z-Score Standardization: z = (x - mean) / std_deviation
     * Normalizing inputs around 0 mean and 1 variance helps gradients flow better during Backpropagation
     * and dramatically speeds up SGD convergence while preventing exploding gradients.
     */
    static void standardize(std::vector<std::vector<double>>& features) {
        if (features.empty() || features[0].empty()) return;
        
        size_t num_features = features[0].size();
        size_t num_samples = features.size();

        for (size_t j = 0; j < num_features; ++j) {
            double mean = 0.0;
            for (size_t i = 0; i < num_samples; ++i) {
                mean += features[i][j];
            }
            mean /= num_samples;

            double var = 0.0;
            for (size_t i = 0; i < num_samples; ++i) {
                var += (features[i][j] - mean) * (features[i][j] - mean);
            }
            double std_dev = std::sqrt(var / num_samples);
            if (std_dev < 1e-8) std_dev = 1e-8; // avoid division by zero

            for (size_t i = 0; i < num_samples; ++i) {
                features[i][j] = (features[i][j] - mean) / std_dev;
            }
        }
    }
};
