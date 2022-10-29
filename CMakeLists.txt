cmake_minimum_required(VERSION 3.8)
project(iiwa-vs-myphd)

add_definitions(-std=c++11)

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")

# find packages
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(drake REQUIRED)
find_package(lcm REQUIRED)


include_directories(include)

add_executable (ic src/impedance_controller.cpp)
target_link_libraries (ic ${fri_client_library} ${LCM_NAMESPACE}lcm ${gflags_LIBRARIES} drake::drake-lcmtypes-cpp Eigen3::Eigen)


# include(${LCM_USE_FILE})
# lcm_wrap_types(
#         C_EXPORT lcm_types
#         C_SOURCES c_sources
#         C_HEADERS c_headers
#         CPP_HEADERS cpp_headers
#         example_t.lcm
# )

# lcm_add_library(lcm_types-cpp CPP ${cpp_headers})
# target_include_directories(lcm_types-cpp INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)


# add_executable(list src/listener.cpp)
# target_link_libraries(list ${fri_client_library} ${LCM_NAMESPACE}lcm ${gflags_LIBRARIES} drake::drake-lcmtypes-cpp lcm_types-cpp)

# add_executable(send_message src/send_message.cpp)
# target_link_libraries(send_message ${fri_client_library} ${LCM_NAMESPACE}lcm ${gflags_LIBRARIES} drake::drake-lcmtypes-cpp lcm_types-cpp)

# add_executable(read_log src/read_log.cpp)
# target_link_libraries(read_log ${fri_client_library} ${LCM_NAMESPACE}lcm ${gflags_LIBRARIES} drake::drake-lcmtypes-cpp lcm_types-cpp)