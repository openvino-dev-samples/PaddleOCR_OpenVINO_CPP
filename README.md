# PaddleOCR_OpenVINO_CPP
This sample shows how to use the OpenVINO C++ 2.0 API to deploy Paddle PP-OCRv3 and PP-structure models, modified from the example in [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/deploy/cpp_infer).

## PP-OCR
PP-OCR is a two-stage OCR system, in which the text detection algorithm is DB, and the text recognition algorithm is SVTR. Besides, a text direction classifier is added between the detection and recognition modules to deal with text in different directions.

<img width="1236" alt="ppocrv3_framework" src="https://user-images.githubusercontent.com/91237924/212936308-05a184af-5826-40e5-9cc5-719b15efaf60.png">

## Layout Information Extraction
In the layout analysis task, the image first goes through the layout analysis model to divide the image into different areas such as text, table, and figure, and then analyze these areas separately. For example, the table area is sent to the form recognition module for structured recognition, and the text area is sent to the OCR engine for text recognition. Finally, the layout recovery module restores it to a word or pdf file with the same layout as the original image.

<img src="https://user-images.githubusercontent.com/14270174/195265734-6f4b5a7f-59b1-4fcc-af6d-89afc9bd51e1.jpg" width="100%"/>


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
$ git clone git@github.com:OpenVINO-dev-contest/PaddleOCR_OpenVINO_CPP.git
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
    /opt/intel/openvino_2022.2/runtime/include/openvino
    ${OpenCV_INCLUDE_DIR}
)
link_directories("/opt/intel/openvino_2022.2/runtime/lib")
...
```

### Build the source code
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### Download test model
Download the models:

**1) PP-OCRv3 Series Model List**

| Model introduction                                           | Model name                   | Recommended scene | Detection model                                              | Direction classifier                                         | Recognition model                                            |
| ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Chinese and English ultra-lightweight PP-OCRv3 model（16.2M）     | ch_PP-OCRv3_xx          | Mobile & Server | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) |

**2) PP-structure models**

   You can find the latest Layout Analysis model and Table Recognition model at [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/models_list_en.md).

### Run the program
**1) OCR**
```
$ ./build/reader \
    -type ocr \
    -input ~/input_image.jpg \
    -label_dir ../data/ppocr_keys_v1.txt \
    -det_model_dir ~/detection.pdmodel \
    -cls_model_dir ~/classifier.pdmodel \
    -rec_model_dir ~/recognizer.pdmodel
```

**2) Layout Information Extraction**
```
$ ./build/reader \
    -type structure \
    -input ~/input_image.jpg \
    -label_dir ../data/ppocr_keys_v1.txt \
    -layout_dict_dir ./data/layout_publaynet_dict.txt \
    -table_dict_dir ./data/table_structure_dict.txt \
    -det_model_dir ~/detection.pdmodel \
    -rec_model_dir ~/recognizer.pdmodel \
    -lay_model_dir ~/layout.pdmodel \
    -tab_model_dir ~/table.pdmodel
```

p.s You can try [Paddle2onnx](https://github.com/PaddlePaddle/Paddle2ONNX) to convert a model unsupportted by OpenVINO yet.
### Output example

**1) OCR**

![00056221](https://user-images.githubusercontent.com/91237924/205421176-77296ee7-f200-4914-a719-dc1e827d0dd1.jpg)
![result](https://user-images.githubusercontent.com/91237924/205421169-08045ce3-5e7d-42f5-bd1a-911f76aac59b.jpg)

```
0       det boxes: [[0,0],[160,0],[160,51],[0,51]] rec text: 7788.com rec score: 0.9815 cls label: 0 cls score: 0.93939
1       det boxes: [[74,100],[231,98],[231,126],[74,128]] rec text: Z57A001950 rec score: 0.9929 cls label: 0 cls score: 1
2       det boxes: [[406,101],[508,101],[508,133],[406,133]] rec text: 杭州东售 rec score: 0.99703 cls label: 0 cls score: 1
3       det boxes: [[66,138],[325,137],[325,162],[66,163]] rec text: 2013年07月07日13：39开 rec score: 0.924703 cls label: 0 cls score: 1
4       det boxes: [[391,139],[506,139],[506,161],[391,161]] rec text: 06车12B号 rec score: 0.913608 cls label: 0 cls score: 1
5       det boxes: [[440,158],[508,156],[509,185],[441,187]] rec text: 二等座 rec score: 0.985737 cls label: 0 cls score: 0.998492
6       det boxes: [[89,179],[198,179],[198,217],[89,217]] rec text: 杭州东 rec score: 0.996495 cls label: 0 cls score: 1
7       det boxes: [[236,171],[354,173],[354,205],[236,203]] rec text: G7512次 rec score: 0.945879 cls label: 0 cls score: 1
8       det boxes: [[382,180],[521,182],[521,217],[382,215]] rec text: 上海虹桥 rec score: 0.982963 cls label: 0 cls score: 1
9       det boxes: [[78,214],[223,216],[223,241],[78,239]] rec text: HangZhouDong rec score: 0.989637 cls label: 0 cls score: 0.99992
10      det boxes: [[360,216],[529,216],[529,240],[360,240]] rec text: Shang HaiHongQiao rec score: 0.927789 cls label: 0 cls score: 1
11      det boxes: [[75,245],[181,245],[181,266],[75,266]] rec text: ￥73.00元 rec score: 0.937644 cls label: 0 cls score: 1
12      det boxes: [[75,273],[220,273],[220,298],[75,298]] rec text: 限乘当日当次车 rec score: 0.97018 cls label: 0 cls score: 1
13      det boxes: [[72,299],[148,299],[148,327],[72,327]] rec text: 余友红 rec score: 0.902762 cls label: 0 cls score: 1
14      det boxes: [[296,314],[406,304],[409,336],[299,346]] rec text: 检票口16 rec score: 0.981898 cls label: 0 cls score: 0.999999
15      det boxes: [[69,327],[286,321],[287,352],[70,358]] rec text: 3623301993****0941 rec score: 0.958505 cls label: 0 cls score: 1
16      det boxes: [[427,345],[449,343],[450,353],[428,355]] rec text: DA rec score: 0.379216 cls label: 0 cls score: 0.946252
17      det boxes: [[61,363],[327,363],[327,387],[61,387]] rec text: 9004-1300-5707-08A0-0195-0 rec score: 0.911793 cls label: 0 cls score: 0.999793
18      det boxes: [[419,357],[512,357],[512,382],[419,382]] rec text: 和谐号 rec score: 0.995543 cls label: 0 cls score: 0.999999
19      det boxes: [[14,492],[242,491],[242,506],[14,507]] rec text: Canon PowerShot A3400 IS F2.8 1/20s IS0400 rec score: 0.914529 cls label: 0 cls score: 0.998959
```

**2) Layout Information Extraction**
![ppstructure](https://user-images.githubusercontent.com/91237924/213063872-81380100-227e-469a-a7bd-5becb9c14220.GIF)
