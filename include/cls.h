#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "paddle_inference_api.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/utility.h>
#include <include/postprocess_op.h>
#include <include/preprocess_op.h>

namespace PaddleOCR {

class Cls
{
public:
    Cls();
    ~Cls();
    double cls_thresh = 0.9;
    bool init(std:: model_path);
    
    bool run(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results);

private:
    // ov::CompiledModel detect_model;
    ov::InferRequest infer_request;
    string model_path;
    shared_ptr<ov::Model> model;

    int cls_batch_num_ = 1;
    ClsResizeImg resize_op_;

};
}