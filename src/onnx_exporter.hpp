#pragma once
#include "mlp.hpp"
#include <string>
#include <fstream>
#include "onnx.pb.h"

/**
 * @class ONNXExporter
 * @brief Handles serialization of the trained MLP weights into an ONNX binary format.
 *
 * It uses pure Google Protocol Buffers (protobuf) to construct the ONNX Directed Acyclic Graph (DAG)
 * node by node, thereby eliminating the need for bulky vendor-specific ML dependencies during C++ training.
 */
class ONNXExporter {
public:
    /**
     * @brief Exports the mathematical representation of the trained MLP to a .onnx file.
     * @param mlp The trained MLP engine containing the latest Matrix weights.
     * @param filename Target file output (e.g. "model.onnx")
     * @param input_size Number of features
     * @param output_size Number of labels
     */
    static void export_model(const MLP& mlp, const std::string& filename, int input_size, int output_size) {
        onnx::ModelProto model;
        model.set_ir_version(8);
        model.set_producer_name("MLP-From-Scratch-C++");
        
        auto* opset = model.add_opset_import();
        opset->set_domain("");
        opset->set_version(14); // use standard opset 14

        onnx::GraphProto* graph = model.mutable_graph();
        graph->set_name("MLP_Graph");

        // --- Define Input Tensor Shape ---
        // Describes the dynamic batch_size x features architecture constraint
        onnx::ValueInfoProto* input_info = graph->add_input();
        input_info->set_name("X");
        auto* input_type = input_info->mutable_type()->mutable_tensor_type();
        input_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* input_shape = input_type->mutable_shape();
        input_shape->add_dim()->set_dim_param("batch_size");
        input_shape->add_dim()->set_dim_value(input_size);

        // Define Output
        onnx::ValueInfoProto* output_info = graph->add_output();
        output_info->set_name("Y");
        auto* output_type = output_info->mutable_type()->mutable_tensor_type();
        output_type->set_elem_type(onnx::TensorProto::FLOAT);
        auto* output_shape = output_type->mutable_shape();
        output_shape->add_dim()->set_dim_param("batch_size");
        output_shape->add_dim()->set_dim_value(output_size);

        std::string current_input = "X";

        for (size_t i = 0; i < mlp.weights.size(); ++i) {
            std::string w_name = "W" + std::to_string(i);
            std::string b_name = "B" + std::to_string(i);

            // Add Weight Tensor
            onnx::TensorProto* w_tensor = graph->add_initializer();
            w_tensor->set_name(w_name);
            w_tensor->set_data_type(onnx::TensorProto::FLOAT);
            w_tensor->add_dims(mlp.weights[i].rows);
            w_tensor->add_dims(mlp.weights[i].cols);
            for (size_t r = 0; r < mlp.weights[i].rows; ++r) {
                for (size_t c = 0; c < mlp.weights[i].cols; ++c) {
                    w_tensor->add_float_data(mlp.weights[i].at(r, c));
                }
            }

            // Add Bias Tensor
            onnx::TensorProto* b_tensor = graph->add_initializer();
            b_tensor->set_name(b_name);
            b_tensor->set_data_type(onnx::TensorProto::FLOAT);
            b_tensor->add_dims(mlp.biases[i].cols); // 1D bias
            for (size_t c = 0; c < mlp.biases[i].cols; ++c) {
                b_tensor->add_float_data(mlp.biases[i].at(0, c));
            }

            // --- Define Computational DAG Nodes ---

            // Node: MatMul (Z = X * W)
            std::string matmul_out = "matmul_" + std::to_string(i);
            onnx::NodeProto* matmul_node = graph->add_node();
            matmul_node->add_input(current_input);
            matmul_node->add_input(w_name);
            matmul_node->add_output(matmul_out);
            matmul_node->set_op_type("MatMul");

            // Node: Add (Z = Z + Bias)
            std::string add_out = "add_" + std::to_string(i);
            onnx::NodeProto* add_node = graph->add_node();
            add_node->add_input(matmul_out);
            add_node->add_input(b_name);
            add_node->add_output(add_out);
            add_node->set_op_type("Add");

            // Node: Activation (A = f(Z))
            std::string act_out = "act_" + std::to_string(i);
            if (i == mlp.weights.size() - 1) {
                // Output layer
                if (mlp.alg_type == "classifier") {
                    onnx::NodeProto* act_node = graph->add_node();
                    act_node->add_input(add_out);
                    act_node->add_output("Y"); // Final output
                    act_node->set_op_type("Sigmoid");
                } else {
                    // Linear: No activation node, just rename Add's output to "Y"
                    add_node->set_output(0, "Y");
                }
            } else {
                // Hidden layer: Relu
                onnx::NodeProto* act_node = graph->add_node();
                act_node->add_input(add_out);
                act_node->add_output(act_out);
                act_node->set_op_type("Relu");
                current_input = act_out;
            }
        }

        std::ofstream output(filename, std::ios::binary);
        if (!model.SerializeToOstream(&output)) {
            std::cerr << "Failed to write ONNX model to " << filename << std::endl;
        } else {
            std::cout << "Successfully exported model to " << filename << std::endl;
        }
    }
};
