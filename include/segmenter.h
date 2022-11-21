#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <openvino/openvino.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
using namespace std;
using namespace cv;

class Segmenter
{
public:
    Segmenter();
    ~Segmenter();

    bool init(string xml_path);

    bool run(vector<Mat> &inframes, vector<Mat> &masks);

private:
    // ov::CompiledModel detect_model;
    ov::InferRequest infer_request;
    string model_path;
};