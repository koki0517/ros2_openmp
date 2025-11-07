cmake_minimum_required(VERSION 3.19)
set(CMAKE_C_COMPILER clang-16)
set(CMAKE_CXX_COMPILER clang++-16)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
project(openmp_test)

# Add compile options for Clang
add_compile_options(-Wall -Wextra -Wpedantic)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenMP REQUIRED)
find_package(rclcpp REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

add_executable(openmp_gpu_node src/openmp_gpu_node.cpp)

# Add OpenMP flags for Clang
if(OpenMP_FOUND)
    target_compile_options(openmp_gpu_node PRIVATE -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --offload-arch=sm_86)
    target_link_libraries(openmp_gpu_node OpenMP::OpenMP_CXX)
endif()

ament_target_dependencies(openmp_gpu_node
  "rclcpp"
)

install(TARGETS
  openmp_gpu_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
