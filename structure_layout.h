#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/utility.h>
#include <include/preprocess_op.h>
#include <include/postprocess_op.h>
#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

namespace PaddleOCR {

class Layout
{
public:
    explicit Layout(std::string model_path, std::string layout_dict_path);
    void Run(cv::Mat &src_img, std::vector<StructurePredictResult> &structure_result);

private:

    ov::InferRequest infer_request;
    std::string model_path;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    
    cv::Mat src_img;
    cv::Mat resize_img;
    double e = 1.0 / 255.0;
    const int layout_img_h_ = 800;
    const int layout_img_w_ = 608;
    double layout_nms_threshold = 0.5;
    double layout_score_threshold = 0.5;
    std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    std::vector<float> scale_ = {0.229f, 0.224f, 0.225f};

    // resize    
    Resize resize_op_;
    // post-process
    PicodetPostProcessor post_processor_;
};
}