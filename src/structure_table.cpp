#include "include/structure_table.h"

namespace PaddleOCR {

Table::Table(std::string model_path, const std::string table_char_dict_path) {
    ov::Core core;
    this->model_path = model_path;
    this->model = core.read_model(this->model_path);
    // reshape the model for dynamic batch size and sentence width
    this->model->reshape({{ov::Dimension(1, this->table_batch_num_), 3, this->table_max_len_, this->table_max_len_}});
    this->compiled_model = core.compile_model(this->model, "CPU");
    this->infer_request = this->compiled_model.create_infer_request();
    this->post_processor_.init(table_char_dict_path, false);
}

void Table::Run(std::vector<cv::Mat> img_list,
                std::vector<std::vector<std::string>> &structure_html_tags,
                std::vector<float> &structure_scores,
                std::vector<std::vector<std::vector<int>>> &structure_boxes) {
    int img_num = img_list.size();
    for (int beg_img_no = 0; beg_img_no < img_num;
            beg_img_no += this->table_batch_num_) {
        // preprocess
        auto preprocess_start = std::chrono::steady_clock::now();
        int end_img_no = std::min(img_num, beg_img_no + this->table_batch_num_);
        int batch_num = end_img_no - beg_img_no;
        std::vector<cv::Mat> norm_img_batch;
        std::vector<int> width_list;
        std::vector<int> height_list;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            cv::Mat srcimg;
            img_list[ino].copyTo(srcimg);
            cv::Mat resize_img;
            cv::Mat pad_img;
            this->resize_op_.Run(srcimg, resize_img, this->table_max_len_);
            this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                                    this->is_scale_);
            this->pad_op_.Run(resize_img, pad_img, this->table_max_len_);
            norm_img_batch.push_back(pad_img);
            width_list.push_back(srcimg.cols);
            height_list.push_back(srcimg.rows);
        }

        std::vector<float> input(
            batch_num * 3 * this->table_max_len_ * this->table_max_len_, 0.0f);
        ov::Shape intput_shape = {batch_num, 3, this->table_max_len_, this->table_max_len_};
        this->permute_op_.Run(norm_img_batch, input.data());
        // inference.
        auto input_port = this->compiled_model.input();
        ov::Tensor input_tensor(input_port.get_element_type(), intput_shape, input.data());
        this->infer_request.set_input_tensor(input_tensor);
        // start inference
        this->infer_request.infer();

        auto output0 = this->infer_request.get_output_tensor(0);
        const float *out_data0 = output0.data<const float>();
        auto predict_shape0 = output0.get_shape();
        auto output1 = this->infer_request.get_output_tensor(1);
        const float *out_data1 = output1.data<const float>();
        auto predict_shape1 = output1.get_shape();

        int out_num0 = std::accumulate(predict_shape0.begin(), predict_shape0.end(),
                                        1, std::multiplies<int>());
        int out_num1 = std::accumulate(predict_shape1.begin(), predict_shape1.end(),
                                        1, std::multiplies<int>());

        std::vector<float> loc_preds(out_data0, out_data0 + out_num0);
        std::vector<float> structure_probs(out_data1, out_data1 + out_num1);

        // postprocess
        std::vector<std::vector<std::string>> structure_html_tag_batch;
        std::vector<float> structure_score_batch;
        std::vector<std::vector<std::vector<int>>> structure_boxes_batch;
        this->post_processor_.Run(loc_preds, structure_probs, structure_score_batch,
                                    predict_shape0, predict_shape1,
                                    structure_html_tag_batch, structure_boxes_batch,
                                    width_list, height_list);
        for (int m = 0; m < predict_shape0[0]; m++) {

            structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(),
                                                "<table>");
            structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(),
                                                "<body>");
            structure_html_tag_batch[m].insert(structure_html_tag_batch[m].begin(),
                                                "<html>");
            structure_html_tag_batch[m].push_back("</table>");
            structure_html_tag_batch[m].push_back("</body>");
            structure_html_tag_batch[m].push_back("</html>");
            structure_html_tags.push_back(structure_html_tag_batch[m]);
            structure_scores.push_back(structure_score_batch[m]);
            structure_boxes.push_back(structure_boxes_batch[m]);
        }
    }
}
}