cmake_minimum_required(VERSION 3.8)
project(capella_avoiding_pedestrians)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

# 优先用本地 third_party，找不到就用系统路径
set(ONNXRUNTIME_LOCAL ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime)

if(EXISTS ${ONNXRUNTIME_LOCAL}/lib/libonnxruntime.so)
  set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_LOCAL}/include)
  set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LOCAL}/lib)
  message(STATUS "Using bundled ONNX Runtime")
else()
  set(ONNXRUNTIME_INCLUDE_DIR /usr/local/include)
  set(ONNXRUNTIME_LIB_DIR /usr/local/lib)
  message(STATUS "Using system ONNX Runtime")
endif()

add_executable(behavior_detection_node
  src/avoiding_pedestrians.cpp
)

target_include_directories(behavior_detection_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_INCLUDE_DIR}
)

ament_target_dependencies(behavior_detection_node
  ament_index_cpp
  builtin_interfaces
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)

target_link_libraries(behavior_detection_node
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)

set_target_properties(behavior_detection_node PROPERTIES
  BUILD_RPATH ${ONNXRUNTIME_LIB_DIR}
  INSTALL_RPATH "$ORIGIN"
)

install(TARGETS behavior_detection_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(FILES src/avoid_person.onnx
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
  ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.16.3
  ${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_shared.so
  ${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_cuda.so
  ${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_tensorrt.so
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
