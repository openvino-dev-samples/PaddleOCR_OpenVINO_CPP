# PaddleOCR_OpenVINO_CPP
This sample shows how to use the OpenVINO C++ 2.0 API to deploy Paddle PP-OCRv3 model, modified from the example in (PaddleOCR)[https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/deploy/cpp_infer].

<img width="468" alt="PaddleOCR" src="https://user-images.githubusercontent.com/91237924/205211572-ee387d7c-6341-4541-85e7-6bfcbcdcbd79.png">

## System requirements

| Optimized for    | Description
|----------------- | ----------------------------------------
| OS               | Ubuntu* 20.04
| Hardware         | Intel® - CPU platform
| Software         | Intel® - OpenVINO 2022.2

## How to build the sample

### Install OpenVINO toolkits 2022.2 from achieved package
Download and install OpenVINO C++ runtime:
https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_from_archive_linux.html

### Download the repository
```shell
git clone git@github.com:OpenVINO-dev-contest/PaddleOCR_OpenVINO_CPP.git
```

### Configure the CMakeLists.txt
set-up the OpenVINO library path according your installation. 
```shell
...
set(openvino_LIBRARIES "/opt/intel/openvino_2022.2/runtime/lib/intel64/libopenvino.so")

include_directories(
    ./
    /opt/intel/openvino_2022.2/runtime/include
    /opt/intel/openvino_2022.2/runtime/include/ie
    /opt/intel/openvino_2022.2/runtime/include/ngraph
    /opt/intel/openvino_2022.2runtime/include/openvino
    ${OpenCV_INCLUDE_DIR}
)
link_directories("/opt/intel/openvino_2022.2/runtime/lib")
...
```

### Build the source code
```
mkdir build
cd build
cmake ..
make
```

### Download test model
Download the models:
PP-OCRv3 Series Model List（Update on September 8th）

| Model introduction                                           | Model name                   | Recommended scene | Detection model                                              | Direction classifier                                         | Recognition model                                            |
| ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Chinese and English ultra-lightweight PP-OCRv3 model（16.2M）     | ch_PP-OCRv3_xx          | Mobile & Server | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) |

### Run the program
```
./build/ocr_reader
```

