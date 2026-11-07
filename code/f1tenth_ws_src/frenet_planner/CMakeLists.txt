cmake_minimum_required(VERSION 3.5)
project(frenet_planner)

foreach(PACKAGE
  ament_cmake
  ament_cmake_python
  rclpy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  ackermann_msgs
  visualization_msgs
  std_srvs)

  find_package(${PACKAGE} REQUIRED)

endforeach()

# Planner core (no ROS imports)
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/frenet_planner_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch maps rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(frenet_planner_core test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()

ament_package()
