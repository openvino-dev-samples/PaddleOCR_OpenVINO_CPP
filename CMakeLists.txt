cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)
#find_package(OpenVINO REQUIRED)

set(openvino_LIBRARIES "/opt/intel/openvino_2022.2/runtime/lib/intel64/libopenvino.so")

include_directories(
    ./
    /opt/intel/openvino_2022.2/runtime/include
    /opt/intel/openvino_2022.2/runtime/include/ie
    /opt/intel/openvino_2022.2/runtime/include/ngraph
    /opt/intel/openvino_2022.2/runtime/include/openvino
    ${OpenCV_INCLUDE_DIR}
)
link_directories("/opt/intel/openvino_2022.2/runtime/lib")
aux_source_directory(src SRC)
add_executable(reader main.cpp ${SRC})
target_link_libraries(reader PRIVATE ${openvino_LIBRARIES} ${OpenCV_LIBS} gflags)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")