#include "include/structure_layout.h"

namespace PaddleOCR {

Layout::Layout(std::string model_path, std::string layout_dict_path) {
    ov::Core core;
    this->model_path = model_path;
    this->model = core.read_model(this->model_path);
    this->model->reshape({1, 3, this->layout_img_h_, this->layout_img_w_});

    // preprocessing API
    ov::preprocess::PrePostProcessor prep(this->model);
    // declare section of desired application's input format
    prep.input().tensor().set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    // specify actual model layout
    prep.input().model().set_layout("NCHW");
    prep.input().preprocess().mean(this->mean_).scale(this->scale_);
    // dump preprocessor
    std::cout << "Preprocessor: " << prep << std::endl;
    this->model = prep.build();
    this->compiled_model = core.compile_model(this->model, "CPU");
    this->infer_request = this->compiled_model.create_infer_request();

    this->post_processor_.init(layout_dict_path, this->layout_score_threshold,
                                this->layout_nms_threshold);
}

void Layout::Run(cv::Mat &src_img, std::vector<StructurePredictResult> &structure_result) {
    this->src_img = src_img;
    this->resize_op_.Run(this->src_img, this->resize_img, this->layout_img_h_, this->layout_img_w_);
    std::vector<std::vector<std::vector<int>>> boxes;
    auto input_port = this->compiled_model.input();

    // -------- set input --------
    this->resize_img.convertTo(this->resize_img, CV_32FC3, e);
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), (float *)this->resize_img.data);
    this->infer_request.set_input_tensor(input_tensor);
    // -------- start inference --------
    this->infer_request.infer();

    std::vector<std::vector<float>> out_tensor_list;
    std::vector<ov::Shape> output_shape_list;
    for (int j = 0; j < (this->model->outputs()).size(); j++) {
        auto output = this->infer_request.get_output_tensor(j);
        auto output_shape = output.get_shape();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                        std::multiplies<int>());
        output_shape_list.push_back(output_shape);

        const float *out_data = output.data<const float>();
        std::vector<float> out_tensor(out_data, out_data + out_num);
        out_tensor_list.push_back(out_tensor);
    }

    std::vector<int> bbox_num;
    int reg_max = 0;
    for (int i = 0; i < out_tensor_list.size(); i++) {
        if (i == this->post_processor_.fpn_stride_.size()) {
            reg_max = output_shape_list[i][2] / 4;
            break;
        }
    }
    std::vector<int> ori_shape = {this->src_img.rows, this->src_img.cols};
    std::vector<int> resize_shape = {this->resize_img.rows, this->resize_img.cols};
    this->post_processor_.Run(structure_result, out_tensor_list, ori_shape, resize_shape,
                                reg_max);
    bbox_num.push_back(structure_result.size());
}
}