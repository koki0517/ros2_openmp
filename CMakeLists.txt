cmake_minimum_required(VERSION 3.19)
set(CMAKE_C_COMPILER nvc)
set(CMAKE_CXX_COMPILER nvc++)
project(openmp_test)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
# find_package(OpenMP) は不要

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(openmp_gpu_node src/openmp_gpu_node.cpp)

# --- 修正箇所: フラグを2つの別々の引数として定義 ---
set(NVIDIA_OMP_FLAGS "-mp=gpu" "-gpu=sm_86")
target_compile_options(openmp_gpu_node PRIVATE ${NVIDIA_OMP_FLAGS})
target_link_options(openmp_gpu_node PRIVATE ${NVIDIA_OMP_FLAGS})

ament_target_dependencies(openmp_gpu_node
  "rclcpp"
)

install(TARGETS
  openmp_gpu_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
