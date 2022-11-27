#include "include/det.h"
#include "include/cls.h"
#include "include/rec.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>

using namespace PaddleOCR;

int main(int argc, char *argv[])
{
    const char *image_path{argv[1]};
    const std::string &label_path{argv[2]};
    const std::string det_model_path{argv[3]};
    const std::string cls_model_path{argv[4]};
    const std::string rec_model_path{argv[5]};
    std::vector<OCRPredictResult> ocr_result;
  
    cv::Mat src_img = imread(image_path);
    Det det;
    det.init(src_img, det_model_path);
    Cls cls;
    cls.init(cls_model_path);
    Rec rec;
    rec.init(rec_model_path, label_path);

    
  
    det.run(ocr_result);

    // crop image
    std::vector<cv::Mat> img_list;
    for (int j = 0; j < ocr_result.size(); j++) {
        cv::Mat crop_img;
        crop_img = Utility::GetRotateCropImage(src_img, ocr_result[j].box);
        img_list.push_back(crop_img);
    }

    cls.run(img_list, ocr_result);
    for (int i = 0; i < img_list.size(); i++) {
      if (ocr_result[i].cls_label % 2 == 1 &&
          ocr_result[i].cls_score > cls.cls_thresh) {
        cv::rotate(img_list[i], img_list[i], 1);
      }
    }
    rec.run(img_list, ocr_result);
    Utility::print_result(ocr_result);
    Utility::VisualizeBboxes(src_img, ocr_result,
                                 "./result.jpg");
}