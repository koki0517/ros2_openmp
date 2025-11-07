#include "rclcpp/rclcpp.hpp"
#include <vector>
#include <iostream>
#include <omp.h>

class OpenMPGpuNode : public rclcpp::Node
{
public:
    OpenMPGpuNode() : Node("openmp_gpu_node")
    {
        RCLCPP_INFO(this->get_logger(), "OpenMP GPU Node has been started.");

        // Check for GPU availability
        int num_devices = omp_get_num_devices();
        RCLCPP_INFO(this->get_logger(), "Number of available GPUs: %d", num_devices);

        if (num_devices == 0) {
            RCLCPP_ERROR(this->get_logger(), "No GPU devices found for OpenMP offloading.");
            return;
        }

        // Run the GPU computation
        this->perform_gpu_computation();
    }

private:
    void perform_gpu_computation()
    {
        const int n = 100000;
        std::vector<float> a(n), b(n), c(n);

        // Initialize vectors
        for (int i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i * 2);
        }

        RCLCPP_INFO(this->get_logger(), "Performing vector addition on the GPU...");

        // Offload computation to the GPU
        #pragma omp target teams distribute parallel for map(to:a.data()[0:n], b.data()[0:n]) map(from:c.data()[0:n])
        for (int i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }

        RCLCPP_INFO(this->get_logger(), "Computation finished.");

        // Verification
        bool success = true;
        for (int i = 0; i < 5; ++i) { // Check first 5 elements
             RCLCPP_INFO(this->get_logger(), "c[%d] = %f", i, c[i]);
            if (c[i] != a[i] + b[i]) {
                success = false;
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Last element c[%d] = %f", n-1, c[n-1]);


        if (success) {
            RCLCPP_INFO(this->get_logger(), "GPU computation was successful!");
        } else {
            RCLCPP_ERROR(this->get_logger(), "GPU computation failed!");
        }
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OpenMPGpuNode>());
    rclcpp::shutdown();
    return 0;
}
