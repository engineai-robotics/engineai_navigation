cmake_minimum_required(VERSION 3.8)
project(mid360_preprocessor)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)

add_library(mid360_resampler
  src/resampler.cpp
)
target_include_directories(mid360_resampler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(mid360_resampler Eigen3::Eigen)

add_executable(mid360_preprocessor_node
  src/mid360_preprocessor_node.cpp
)
target_include_directories(mid360_preprocessor_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(mid360_preprocessor_node
  mid360_resampler
  Eigen3::Eigen
)
ament_target_dependencies(mid360_preprocessor_node
  geometry_msgs
  livox_ros_driver2
  rclcpp
  sensor_msgs
  tf2_eigen
  tf2_ros
)

install(
  TARGETS mid360_resampler mid360_preprocessor_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(mid360_resampler)
ament_export_dependencies(Eigen3)
ament_package()
