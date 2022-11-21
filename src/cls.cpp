#include "include/cls.h"

namespace PaddleOCR {

Cls::Cls() {}

Cls::~Cls() {}

bool Cls::init(std::string model_path)
{
    this->model_path = model_path;
    ov::Core core;
    this->model = core.read_model(this->model_path);
    // -------- Step 3. Preprocessing API--------
    ov::preprocess::PrePostProcessor prep(this->model);
    // Declare section of desired application's input format
    prep.input().tensor()
        .set_layout("NCHW")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    // Specify actual model layout
    prep.input().model()
        .set_layout("NCHW");
    prep.input().preprocess()
        .mean({0.5f, 0.5f, 0.5f}).
        .scale({0.5f, 0.5f, 0.5f});
    // Dump preprocessor
    std::cout << "Preprocessor: " << prep << std::endl;
    this->model = prep.build();
    return true;
}

bool Cls::run(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results)
{
    std::vector<int> cls_labels(img_list.size(), 0);
    std::vector<float> cls_scores(img_list.size(), 0);
    std::vector<double> cls_times;

    int img_num = img_list.size();
    std::vector<int> cls_image_shape = {3, 48, 192};
    this->model->reshape({{this->cls_batch_num_, cls_image_shape[0], cls_image_shape[1],-1}});
    ov::CompiledModel cls_model = core.compile_model(this->model, "CPU");
    this->infer_request = cls_model.create_infer_request();
    auto input_port = cls_model.input();

    for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += this->cls_batch_num_) {
        auto preprocess_start = std::chrono::steady_clock::now();
        int end_img_no = std::min(img_num, beg_img_no + this->cls_batch_num_);
        int batch_num = end_img_no - beg_img_no;
        // preprocess    
        std::vector<ov::Tensor> batch_tensors;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            cv::Mat srcimg;
            img_list[ino].copyTo(srcimg);
            cv::Mat resize_img;
            this->resize_op_.Run(srcimg, resize_img, cls_image_shape);

            if (resize_img.cols < cls_image_shape[2]) {
                cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                                cls_image_shape[2] - resize_img.cols,
                                cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            }
            resize_img.convertTo(resize_img, CV_32FC3);
            ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), (float*)resize_img.data);
            batch_tensors.push_back(input_tensor);
        }
    
        this->infer_request.set_input_tensor(batch_tensors);
        // -------- Step 7. Start inference --------
        this->infer_request.infer();


        auto output = this->infer_request.get_output_tensor(0);
        const float *out_data = output.data<const float>();
        for (size_t batch_idx = 0; batch_idx < output.get_size() / 2; batch_idx++){
            int label = int(
                Utility::argmax(&out_data[batch_idx * 2],
                            &out_data[(batch_idx + 1) * 2]));
            float score = float(*std::max_element(
                &out_data[batch_idx * 2],
                &out_data[(batch_idx + 1) * 2]));
            cls_labels[beg_img_no + batch_idx] = label;
            cls_scores[beg_img_no + batch_idx] = score;


        }
    }

    for (int i = 0; i < cls_labels.size(); i++) {
        ocr_results[i].cls_label = cls_labels[i];
        ocr_results[i].cls_score = cls_scores[i];
    }
    return true;
}
}