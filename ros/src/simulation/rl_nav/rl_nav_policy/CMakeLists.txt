cmake_minimum_required(VERSION 3.8)
project(rl_nav_policy)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    scripts/rl_nav_policy_node
    scripts/lio_robo_tf
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY config launch policy
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
