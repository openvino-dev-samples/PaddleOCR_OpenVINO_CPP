cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
#find_package(OpenVINO REQUIRED)

set(openvino_LIBRARIES "/home/ethan/intel/openvino_2022.1.0.643/runtime/lib/intel64/libopenvino.so")

include_directories(
    ./
    /home/ethan/intel/openvino_2022.1.0.643/runtime/include
    /home/ethan/intel/openvino_2022.1.0.643/runtime/include/ie
    /home/ethan/intel/openvino_2022.1.0.643/runtime/include/ngraph
    /home/ethan/intel/openvino_2022.1.0.643/runtime/include/openvino
    ${OpenCV_INCLUDE_DIR}
)
link_directories("/home/ethan/intel/openvino_2022.1.0.643/runtime/lib")
aux_source_directory(src SRC)
add_executable(meter_reader main.cpp ${SRC})
target_link_libraries(meter_reader PRIVATE ${openvino_LIBRARIES} ${OpenCV_LIBS})
