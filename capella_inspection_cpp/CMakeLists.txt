cmake_minimum_required(VERSION 3.8)
project(capella_inspection_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 要求 C++17（std::optional 等特性需要）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ======================== ROS 2 依赖 ========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(capella_ros_msg REQUIRED)

# ======================== 非 ROS 依赖 ========================
find_package(OpenCV REQUIRED)

# ONNX Runtime —— 手动指定路径（根据你的实际安装位置修改）
# 如果你是解压到 /usr/local 的，下面的默认路径通常能直接用
set(ONNXRUNTIME_ROOT "/usr/local" CACHE PATH "ONNX Runtime install prefix")
find_library(ONNXRUNTIME_LIB onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib
)
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include/onnxruntime
        ${ONNXRUNTIME_ROOT}/include
)

if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE)
  message(FATAL_ERROR
    "ONNX Runtime not found!\n"
    "  Searched lib in: ${ONNXRUNTIME_ROOT}/lib\n"
    "  Searched headers in: ${ONNXRUNTIME_ROOT}/include\n"
    "  Set -DONNXRUNTIME_ROOT=<path> to specify the install location.")
endif()

message(STATUS "ONNX Runtime lib:     ${ONNXRUNTIME_LIB}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE}")

# ======================== 可执行文件 ========================
add_executable(capella_inspection_node
  src/capella_inspection_node.cpp
)

# 头文件搜索路径
target_include_directories(capella_inspection_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE}
  ${OpenCV_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(capella_inspection_node
  ${ONNXRUNTIME_LIB}
  ${OpenCV_LIBS}
)

# ROS 2 依赖的头文件 + 链接（ament 风格）
ament_target_dependencies(capella_inspection_node
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  builtin_interfaces
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  image_transport
  capella_ros_msg
)

# ======================== 安装 ========================
install(TARGETS capella_inspection_node
  DESTINATION lib/${PROJECT_NAME}
)

# 如果有 launch 文件或配置文件，也一并安装
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL  # 目录不存在时不报错
)

# ======================== 测试 ========================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()