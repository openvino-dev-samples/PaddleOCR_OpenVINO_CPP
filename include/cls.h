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

class Cls
{
public:
    Cls();
    ~Cls();
    double cls_thresh = 0.9;
    bool init(std::string model_path);
    
    bool run(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results);

private:
    // ov::CompiledModel detect_model;
    ov::InferRequest infer_request;
    string model_path;
    shared_ptr<ov::Model> model;
    ov::CompiledModel cls_model;
    int cls_batch_num_ = 1;
    std::vector<int> cls_image_shape = {3, 48, 192};
    ClsResizeImg resize_op_;

};
}