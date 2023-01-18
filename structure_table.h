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

namespace PaddleOCR {

class Table
{
public:
    explicit Table(std::string model_path, const std::string table_char_dict_path);
    void Run(std::vector<cv::Mat> img_list,
            std::vector<std::vector<std::string>> &structure_html_tags,
            std::vector<float> &structure_scores,
            std::vector<std::vector<std::vector<int>>> &structure_boxes);

private:

    ov::InferRequest infer_request;
    std::string model_path;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;

    cv::Mat src_img;
    cv::Mat resize_img;
    const std::string table_char_dict_path;

    int table_batch_num_ = 1;
    int table_max_len_ = 488;
    std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    bool is_scale_ = true;

    // pre-process
    TableResizeImg resize_op_;
    Normalize normalize_op_;
    PermuteBatch permute_op_;
    TablePadImg pad_op_;

    // post-process
    TablePostProcessor post_processor_;
};
}