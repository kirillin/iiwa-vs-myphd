cmake_minimum_required(VERSION 3.8)
project(iiwa-vs-myphd)

add_definitions(-std=c++11 -lpthread)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_EXE_LINKER_FLAGS:STRING=-Wl,--copy-dt-needed-entries)

# find packages
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(drake REQUIRED)
find_package(lcm REQUIRED)

# find_package(Boost REQUIRED 
# 	COMPONENTS filesystem system signals regex date_time program_options thread
# )

find_package(VISP REQUIRED)
if(VISP_FOUND)
  include(${VISP_USE_FILE})
endif(VISP_FOUND)

find_package(Boost COMPONENTS system thread REQUIRED)

# find_package(kdl_parser REQUIRED)
# find_package(orocos_kdl REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

# find_package(urdf QUIET)
# if(urdf_FOUND)
#   add_definitions(-DHAS_URDF)
#   include_directories(${urdf_INCLUDE_DIRS})
# endif()

find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
    SET(REALSENSE2_FOUND "realsense2")
    message(WARN "Failed to find_library(realsense2)")
endif()

include_directories(
    include
  	${VISP_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# add_executable (ic src/impedance_controller_300Hz.cpp)
# target_link_libraries (ic ${fri_client_library} ${LCM_NAMESPACE}lcm ${gflags_LIBRARIES} drake::drake-lcmtypes-cpp Eigen3::Eigen ${REALSENSE2_FOUND} ${orocos_kdl_LIBRARIES} kdl_parser urdf ${OpenCV_LIBS})

# add_executable (ic src/impedance_controller_300Hz.cpp)
# target_link_libraries (ic ${fri_client_library} ${LCM_NAMESPACE}lcm ${gflags_LIBRARIES} drake::drake-lcmtypes-cpp Eigen3::Eigen ${REALSENSE2_FOUND} ${OpenCV_LIBS})

add_executable(multicamera src/multicamera.cpp)
target_link_libraries(multicamera ${REALSENSE2_FOUND} ${OpenCV_LIBS})
