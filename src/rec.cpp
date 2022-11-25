#include "include/rec.h"

namespace PaddleOCR {

Rec::Rec() {}

Rec::~Rec() {}

bool Rec::init(string model_path, const string &label_path)
{
    this->model_path = model_path;
    ov::Core core;
    this->model = core.read_model(this->model_path);
    this->model->reshape({{-1, this->rec_image_shape_[0], this->rec_image_shape_[1],-1}});
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
        .mean({0.5f, 0.5f, 0.5f})
        .scale({0.5f, 0.5f, 0.5f});
    // Dump preprocessor
    std::cout << "Preprocessor: " << prep << std::endl;
    this->model = prep.build();
    
    this->rec_model = core.compile_model(this->model, "CPU");
    this->infer_request = this->rec_model.create_infer_request();
    this->label_list_ = Utility::ReadDict(label_path);
    this->label_list_.insert(this->label_list_.begin(),
                             "#"); // blank char for ctc
    this->label_list_.push_back(" ");

    return true;
}

bool Rec::run(std::vector<cv::Mat> img_list, std::vector<OCRPredictResult> &ocr_results)
{
    std::vector<std::string> rec_texts(img_list.size(), "");
    std::vector<float> rec_text_scores(img_list.size(), 0);

    int img_num = img_list.size();
    std::vector<float> width_list;
    for (int i = 0; i < img_num; i++) {
        width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
    }
    std::vector<int> indices = Utility::argsort(width_list);
    auto input_port = this->rec_model.input();


    for (int beg_img_no = 0; beg_img_no < img_num;
        beg_img_no += this->rec_batch_num_) {
        int end_img_no = std::min(img_num, beg_img_no + this->rec_batch_num_);
        int batch_num = end_img_no - beg_img_no;
        int imgH = this->rec_image_shape_[1];
        int imgW = this->rec_image_shape_[2];
        float max_wh_ratio = imgW * 1.0 / imgH;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            int h = img_list[indices[ino]].rows;
            int w = img_list[indices[ino]].cols;
            float wh_ratio = w * 1.0 / h;
            max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
        }

        std::vector<cv::Mat> img_batch;
        std::vector<ov::Tensor> batch_tensors;


        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            cv::Mat srcimg;
            img_list[indices[ino]].copyTo(srcimg);
            cv::Mat resize_img;
            this->resize_op_.Run(srcimg, resize_img, max_wh_ratio, this->rec_image_shape_);
            
            resize_img.convertTo(resize_img, CV_32FC3);
            auto input_tensor = this->infer_request.get_input_tensor();
            input_tensor.data<float>() = (float*)resize_img.data;
            // ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), (float*)resize_img.data);
            batch_tensors.push_back(input_tensor);
        }


        this->infer_request.set_input_tensors(batch_tensors);
        // -------- Step 7. Start inference --------
        this->infer_request.infer();

        auto output = this->infer_request.get_output_tensor(0);
        const float *out_data = output.data<const float>();

        ov::Shape predict_shape = this->model->output().get_shape();


        // predict_batch is the result of Last FC with softmax
        for (int m = 0; m < predict_shape[0]; m++) {
            std::string str_res;
            int argmax_idx;
            int last_index = 0;
            float score = 0.f;
            int count = 0;
            float max_value = 0.0f;

            for (int n = 0; n < predict_shape[1]; n++) {
                // get idx
                argmax_idx = int(Utility::argmax(
                    &out_data[(m * predict_shape[1] + n) * predict_shape[2]],
                    &out_data[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
                // get score
                max_value = float(*std::max_element(
                    &out_data[(m * predict_shape[1] + n) * predict_shape[2]],
                    &out_data[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

                if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                score += max_value;
                count += 1;
                str_res += this->label_list_[argmax_idx];
                }
                last_index = argmax_idx;
            }
            score /= count;
            if (std::isnan(score)) {
                continue;
            }
            rec_texts[indices[beg_img_no + m]] = str_res;
            rec_text_scores[indices[beg_img_no + m]] = score;
        }
    }
    // sort boex from top to bottom, from left to right
    for (int i = 0; i < rec_texts.size(); i++) {
        ocr_results[i].text = rec_texts[i];
        ocr_results[i].score = rec_text_scores[i];
    }
    return true;
}

}