#include "include/ocr_det.h"

namespace PaddleOCR {
    
Detector::Detector(std::string model_path)
{
    ov::Core core;
    this->model_path = model_path;
    this->model = core.read_model(this->model_path);
    this->model->reshape({1, 3, ov::Dimension(32, this->limit_side_len_), ov::Dimension(1, this->limit_side_len_)});
    this->compiled_model = core.compile_model(this->model, "CPU");
    this->infer_request = this->compiled_model.create_infer_request();
}

void Detector::Run(cv::Mat &src_img, std::vector<OCRPredictResult> &ocr_results)
{
    this->src_img = src_img;
    this->resize_op_.Run(this->src_img, this->resize_img, this->limit_type_,
                            this->limit_side_len_, this->ratio_h, this->ratio_w);

    this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                            this->is_scale_);

    std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
    ov::Shape intput_shape = {1, 3, resize_img.rows, resize_img.cols};
    this->permute_op_.Run(&resize_img, input.data());

    std::vector<std::vector<std::vector<int>>> boxes;
    auto input_port = this->compiled_model.input();

    // -------- set input --------
    ov::Tensor input_tensor(input_port.get_element_type(), intput_shape, input.data());
    this->infer_request.set_input_tensor(input_tensor);
    // -------- start inference --------

    this->infer_request.infer();

    auto output = this->infer_request.get_output_tensor(0);
    const float *out_data = output.data<const float>();

    ov::Shape output_shape = output.get_shape();
    const size_t n2 = output_shape[2];
    const size_t n3 = output_shape[3];
    const int n = n2 * n3;

    std::vector<float> pred(n, 0.0);
    std::vector<unsigned char> cbuf(n, ' ');

    for (int i = 0; i < n; i++) {
        pred[i] = float(out_data[i]);
        cbuf[i] = (unsigned char)((out_data[i]) * 255);
    }

    cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
    cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

    const double threshold = this->det_db_thresh_ * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    if (this->use_dilation_) {
        cv::Mat dila_ele =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, bit_map, dila_ele);
    }

    boxes = post_processor_.BoxesFromBitmap(
        pred_map, bit_map, this->det_db_box_thresh_, this->det_db_unclip_ratio_,
        this->det_db_score_mode_);

    boxes = post_processor_.FilterTagDetRes(boxes, this->ratio_h, this->ratio_w, this->src_img);
    for (int i = 0; i < boxes.size(); i++) {
        OCRPredictResult res;
        res.box = boxes[i];
        ocr_results.push_back(res);
    }
    // sort boex from top to bottom, from left to right
    Utility::sorted_boxes(ocr_results);
}
}