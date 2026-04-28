cmake_minimum_required(VERSION 3.5)
project(capella_ros_msg)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

#添加消息文件和依�?
rosidl_generate_interfaces(${PROJECT_NAME}
  #---msg---
  "msg/Battery.msg"
  "msg/Imu.msg"
  "msg/PID.msg"
  "msg/Velocities.msg"
  "msg/AtomizeSwitchCtrStu.msg"
  "msg/IrData.msg"
  "msg/ContactDriveStu.msg"
  "msg/SingleDetector.msg"
  "msg/DetectResult.msg"
  "msg/CrowdStatistics.msg"
  "msg/AttitudePar.msg"
  "msg/DeviceStatus.msg"
  "msg/TrackPersonBox.msg"
  "msg/TrackPersonBoxes.msg"
  "msg/UwbData.msg"
  "msg/RobotPoseWithNamespace.msg"
  "msg/PlanWithNamespace.msg"
  "msg/RobotInfo.msg"
  "msg/RobotInfos.msg"
  "msg/GarbageDetect.msg"
  "msg/CleaningConfig.msg"
  "msg/PassedPosesIndex.msg"
  "msg/MultiRobotsInfo.msg"
  "msg/GarbageDetectSingle.msg"
  "msg/GarbageDetectArray.msg"
  "msg/HighlightZones.msg"
  "msg/CarDetectArray.msg"
  "msg/CarDetectSingle.msg"
  "msg/SingleRecognition.msg"
  "msg/Recognitions.msg"
  "msg/WheelSpeed.msg"
  "msg/PushAndMotorCtrCmd.msg"
  "msg/LaneCenterPaths.msg"
  "msg/CarPassable.msg"
  #---srv---
  "srv/ChargePileWifi.srv"
  "srv/TurnoffPcPower.srv"
  "srv/IsCarPassable.srv"
  #---action---
  "action/RelocalizationSpin.action"
  "action/FollowPerson.action"
  "action/SearchGarbage.action"
  "action/Empty.action"
  "action/DriveIntoElevator.action"
  "action/DriveOutofElevator.action"
  "action/MoveToGoal.action"
  "action/IntelligentCleanGarbage.action"
  "action/FindCarAvoidancePoint.action"
  "action/Welt.action"
  "action/PathCheck.action"
  
  
  DEPENDENCIES sensor_msgs geometry_msgs nav_msgs
 )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
