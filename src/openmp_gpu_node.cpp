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

        // --- 修正箇所 1 ---
        // データポインタをローカル変数 (lvalue) に格納する
        float* a_ptr = a.data();
        float* b_ptr = b.data();
        float* c_ptr = c.data();

        RCLCPP_INFO(this->get_logger(), "Performing vector addition on the GPU...");

        // Offload computation to the GPU
        // --- 修正箇所 2 ---
        // map節でローカル変数のポインタを使用する
        #pragma omp target teams distribute parallel for map(to:a_ptr[0:n], b_ptr[0:n]) map(from:c_ptr[0:n])
        for (int i = 0; i < n; ++i) {
            // --- 修正箇所 3 ---
            // デバイス上のループ内でもポインタを使用する
            c_ptr[i] = a_ptr[i] + b_ptr[i];
        }

        RCLCPP_INFO(this->get_logger(), "Computation finished.");

        // Verification (ここはホスト側で実行されるため、元の 'c[i]' のままで問題ない)
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
