#pragma once
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <thread>
#include <mutex>

/**
 * @class NetworkUtils
 * @brief Provides static utility functions for sending and receiving raw binary vectors of doubles over POSIX TCP Sockets.
 *
 * NOTE: These functions transmit raw memory bytes. They do not perform endianness conversion.
 * It assumes all nodes in the cluster (Master and Workers) share the same endianness and 
 * floating-point representation (e.g., IEEE 754 64-bit on x86_64/arm64).
 */
class NetworkUtils {
public:
    /**
     * @brief Sends a vector of doubles over a TCP socket.
     * Protocol: First sends a size_t representing the number of elements, then sends the raw bytes.
     */
    static bool send_doubles(int sock, const std::vector<double>& data) {
        size_t size = data.size();
        if (send(sock, &size, sizeof(size_t), 0) != sizeof(size_t)) return false;
        if (size == 0) return true;
        size_t bytes = size * sizeof(double);
        size_t sent = 0;
        const char* buf = reinterpret_cast<const char*>(data.data());
        while (sent < bytes) {
            ssize_t res = send(sock, buf + sent, bytes - sent, 0);
            if (res <= 0) return false;
            sent += res;
        }
        return true;
    }

    /**
     * @brief Receives a vector of doubles from a TCP socket.
     * Expects a size_t representing the element count, followed by the raw bytes.
     */
    static bool recv_doubles(int sock, std::vector<double>& data) {
        size_t size = 0;
        ssize_t s_res = recv(sock, &size, sizeof(size_t), MSG_WAITALL);
        if (s_res != sizeof(size_t)) {
            std::cerr << "[Network] Error reading size. res=" << s_res << std::endl;
            return false;
        }
        if (size > 1000000) { // arbitrary safeguard 1M doubles
            std::cerr << "[Network] Insane size received: " << size << std::endl;
            return false;
        }
        
        try {
            data.resize(size);
        } catch (const std::exception& e) {
            std::cerr << "[Network] resize failed for size " << size << ": " << e.what() << std::endl;
            return false;
        }

        if (size == 0) return true;
        size_t bytes = size * sizeof(double);
        size_t received = 0;
        char* buf = reinterpret_cast<char*>(data.data());
        while (received < bytes) {
            ssize_t res = recv(sock, buf + received, bytes - received, MSG_WAITALL);
            if (res <= 0) {
                std::cerr << "[Network] Error receiving data payload." << std::endl;
                return false;
            }
            received += res;
        }
        return true;
    }
};

/**
 * @class ParameterClient
 * @brief Worker node connection handler for Distributed Training.
 * 
 * **Concept (The Worker):**
 * The Worker is the "Employee". It connects to the Master, says "Give me the latest weights", 
 * uses those weights to perform a Forward Pass and Backpropagation on its own subset of the data, 
 * and then says "Here are my gradients!". It repeats this process until the Master tells it to stop.
 */
class ParameterClient {
    int sock;
public:
    /**
     * @brief Establishes a TCP connection to the Master node.
     */
    bool connect_to_master(const std::string& ip, int port) {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in serv_addr;
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr);
        if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            return false;
        }
        return true;
    }

    /**
     * @brief Fetches the latest global weights from the Master node.
     * Sends CMD = 1.
     */
    bool fetch_weights(std::vector<double>& flat_weights) {
        int cmd = 1; // 1 = fetch
        if (send(sock, &cmd, sizeof(int), 0) <= 0) return false;
        return NetworkUtils::recv_doubles(sock, flat_weights);
    }

    bool push_gradients(const std::vector<double>& flat_gradients) {
        int cmd = 2; // 2 = push
        if (send(sock, &cmd, sizeof(int), 0) <= 0) return false;
        return NetworkUtils::send_doubles(sock, flat_gradients);
    }

    bool send_shutdown() {
        int cmd = 3;
        if (send(sock, &cmd, sizeof(int), 0) <= 0) return false;
        return true;
    }

    void disconnect() {
        close(sock);
    }
};

/**
 * @class ParameterServer
 * @brief Master node orchestration for Distributed Training.
 * 
 * **Concept (Distributed Training via Parameter Server):**
 * Instead of training on one computer, we split the job. 
 * This `ParameterServer` is the "Boss" (Master). It holds the absolute, global Truth of the weights.
 * It listens on a network port for "Workers" to connect. 
 * Workers download the current weights, calculate the gradients locally, and push those gradients 
 * back to the Master. The Master then updates the global weights.
 */
class ParameterServer {
    int server_fd;
    std::vector<double> global_weights;
    std::mutex weights_mutex;
    double learning_rate;
    bool keep_running = true;

    /**
     * @brief Handles communication with a single connected Worker (runs in a detached thread).
     * Parses commands: 1 (Fetch Weights), 2 (Push Gradients), 3 (Shutdown).
     */
    void handle_client(int client_socket) {
        while (keep_running) {
            int cmd;
            if (recv(client_socket, &cmd, sizeof(int), MSG_WAITALL) <= 0) break;
            
            if (cmd == 1) { // fetch weights
                std::lock_guard<std::mutex> lock(weights_mutex);
                if (!NetworkUtils::send_doubles(client_socket, global_weights)) break;
            } else if (cmd == 2) { // push gradients
                std::vector<double> grads;
                if (!NetworkUtils::recv_doubles(client_socket, grads)) break;
                
                std::lock_guard<std::mutex> lock(weights_mutex);
                if (grads.size() == global_weights.size()) {
                    for (size_t i = 0; i < global_weights.size(); ++i) {
                        global_weights[i] -= learning_rate * grads[i];
                    }
                }
            } else if (cmd == 3) { // shutdown
                keep_running = false;
                break;
            }
        }
        close(client_socket);
    }

public:
    ParameterServer(const std::vector<double>& init_weights, double lr) : global_weights(init_weights), learning_rate(lr) {}

    /**
     * @brief Starts the TCP server loop, binding to the specified port and listening for Workers.
     * Uses select() with a timeout to periodically check if keep_running is false.
     */
    void start(int port) {
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt));
        
        struct sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);
        
        bind(server_fd, (struct sockaddr*)&address, sizeof(address));
        listen(server_fd, 10);
        
        std::cout << "[Master] Parameter Server Listening on port " << port << std::endl;
        
        while (keep_running) {
            // Use select to allow timeout so we can check keep_running
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(server_fd, &readfds);
            struct timeval tv;
            tv.tv_sec = 1;
            tv.tv_usec = 0;
            
            int activity = select(server_fd + 1, &readfds, NULL, NULL, &tv);
            if (activity > 0 && FD_ISSET(server_fd, &readfds)) {
                int new_socket = accept(server_fd, nullptr, nullptr);
                if (new_socket >= 0) {
                    std::thread(&ParameterServer::handle_client, this, new_socket).detach();
                }
            }
        }
        close(server_fd);
        std::cout << "[Master] Shutting down parameter server." << std::endl;
    }
    
    std::vector<double> get_final_weights() {
        std::lock_guard<std::mutex> lock(weights_mutex);
        return global_weights;
    }
};
