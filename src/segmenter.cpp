#include "include/segmenter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <openvino/openvino.hpp>

Segmenter::Segmenter() {}

Segmenter::~Segmenter() {}

bool Segmenter::init(string model_path)
{
    this->model_path = model_path;
    ov::Core core;
    shared_ptr<ov::Model> model = core.read_model(this->model_path);
    map<string, ov::PartialShape> name_to_shape;
    model->reshape({{-1, 3, 512, 512}});
    ov::CompiledModel segment_model = core.compile_model(model, "CPU");
    this->infer_request = segment_model.create_infer_request();
    return true;
}

bool Segmenter::run(vector<Mat> &inframes, vector<Mat> &masks)
{
    static map<int32_t, Vec3b> color_table = {
        {0, Vec3b(0, 0, 0)},
        {1, Vec3b(20, 59, 255)},
        {2, Vec3b(120, 59, 200)},
    };
    float mean[3] = {0.5, 0.5, 0.5};
    float std[3] = {0.5, 0.5, 0.5};

    int batch_size = inframes.size();
    ov::Tensor input_tensor0 = this->infer_request.get_input_tensor(0);
    input_tensor0.set_shape({batch_size, 3, 512, 512});
    auto data0 = input_tensor0.data<float>();
    // nhwc -> nchw
    for (int batch = 0; batch < batch_size; batch++)
    {
        resize(inframes[batch], inframes[batch], Size(512, 512));
        for (int h = 0; h < 512; h++)
        {
            for (int w = 0; w < 512; w++)
            {
                for (int c = 0; c < 3; c++)
                {
                    int out_index = batch * 3 * 512 * 512 + c * 512 * 512 + h * 512 + w;
                    data0[out_index] = float(((float(inframes[batch].at<Vec3b>(h, w)[c]) / 255.0f) - mean[c]) / std[c]);
                }
            }
        }
    }

    //start inference
    this->infer_request.infer();

    //extract the output data
    auto output = this->infer_request.get_output_tensor(0);
    const float *result = output.data<const float>();
    // nchw -> nhwc
    for (int batch = 0; batch < batch_size; batch++)
    {
        Mat mask = Mat::zeros(512, 512, CV_8UC1);
        for (int h = 0; h < 512; h++)
        {
            for (int w = 0; w < 512; w++)
            {
                int argmax_id;
                float max_conf = numeric_limits<float>::min();
                for (int c = 0; c < 3; c++)
                {
                    int out_index = batch * 3 * 512 * 512 + c * 512 * 512 + h * 512 + w;
                    float out_value = result[out_index];
                    if (out_value > max_conf)
                    {
                        argmax_id = c;
                        max_conf = out_value;
                    }
                }
                mask.at<uchar>(h, w) = argmax_id;
            }
        }
        masks.push_back(mask);
    }

    return true;
}