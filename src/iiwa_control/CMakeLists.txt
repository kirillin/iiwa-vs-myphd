cmake_minimum_required(VERSION 3.0.2)
project(iiwa_control)

# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(drake REQUIRED)
find_package(lcm REQUIRED)


find_package(VISP REQUIRED)
if(VISP_FOUND)
  include(${VISP_USE_FILE})
endif(VISP_FOUND)

find_package(Boost COMPONENTS system thread REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

list(APPEND REALSENSE2_INC_SEARCH_PATH /usr/include/librealsense2/include) 
list(APPEND REALSENSE2_LIB_SEARCH_PATH /usr/lib/x86_64-linux-gnu/lib) 
list(APPEND REALSENSE2_INC_SEARCH_PATH /usr/include/librealsense2/include) 
list(APPEND REALSENSE2_LIB_SEARCH_PATH /usr/lib/x86_64-linux-gnu/lib) 

# find_library(realsense2 2.51.1 REQUIRED)
find_library(REALSENSE2_FOUND realsense2 2.51.1 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
    SET(REALSENSE2_FOUND "realsense2")
    message(WARN "Failed to find_library(realsense2)")
endif()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES iiwa_control
#  CATKIN_DEPENDS roscpp std_msgs
#  DEPENDS system_lib
)

include_directories(
  include/iiwa_control
  ${catkin_INCLUDE_DIRS}
  ${VISP_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}  
)


add_executable(cv_node src/cv.cpp src/fine_measuarments_emulator.cpp src/multicamera.cpp)
target_link_libraries(cv_node
  ${catkin_LIBRARIES}
  Eigen3::Eigen
  ${REALSENSE2_FOUND} 
  ${OpenCV_LIBS}
)


add_executable(${PROJECT_NAME}_node src/loop.cpp)
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${fri_client_library} ${LCM_NAMESPACE}lcm ${gflags_LIBRARIES} drake::drake-lcmtypes-cpp Eigen3::Eigen
)