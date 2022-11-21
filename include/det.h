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

class Det
{
public:
    Det();
    ~Det();

    bool init(std::string model_path);

    bool run(cv::Mat &src_img, std::vector<OCRPredictResult> &ocr_results);

private:
    // ov::CompiledModel detect_model;
    ov::InferRequest infer_request;
    string model_path;
    shared_ptr<ov::Model> model;


    string limit_type_ = "max";
    int limit_side_len_ = 960;

    double det_db_thresh_ = 0.3;
    double det_db_box_thresh_ = 0.5;
    double det_db_unclip_ratio_ = 2.0;
    std::string det_db_score_mode_ = "slow";
    bool use_dilation_ = false;

    ResizeImgType0 resize_op_;

    // post-process
    DBPostProcessor post_processor_;
};
}