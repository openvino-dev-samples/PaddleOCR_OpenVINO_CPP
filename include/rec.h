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

using namespace std;
using namespace cv;

namespace PaddleOCR {

class Rec
{
public:
    Rec();
    ~Rec();

    bool init(string model_path, const string &label_path);

    bool run(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results);

private:
    // ov::CompiledModel detect_model;
    ov::InferRequest infer_request;
    string model_path;
    shared_ptr<ov::Model> model;

    std::vector<std::string> label_list_;
    int rec_batch_num_ = 6;
    int rec_img_h_ = 32;
    int rec_img_w_ = 320;
    std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};


    CrnnResizeImg resize_op_;
};
}