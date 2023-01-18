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

#include <include/preprocess_op.h>
#include <include/postprocess_op.h>
#include <openvino/openvino.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

namespace PaddleOCR {

class Classifier
{
public:
    explicit Classifier(std::string model_path);
    void Run(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results);
    
    double cls_thresh = 0.5;

private:
    ov::InferRequest infer_request;
    std::string model_path;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;

    double e = 1.0 / 255.0;
    std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale_ = {0.5f, 0.5f, 0.5f};
    int cls_batch_num_ = 1;
    std::vector<int> cls_image_shape = {3, 48, 192};

    // resize
    ClsResizeImg resize_op_;
};
}