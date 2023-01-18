// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <include/args.h>
#include <include/paddleocr.h>

namespace PaddleOCR {
  
PPOCR::PPOCR() {
  this->detector_ = new Detector(FLAGS_det_model_dir);
  if (!FLAGS_cls_model_dir.empty()) {
    this->classifier_ = new Classifier(FLAGS_cls_model_dir);
  }
  this->recognizer_ = new Recognizer(FLAGS_rec_model_dir, FLAGS_label_dir);
};

std::vector<OCRPredictResult> PPOCR::ocr(cv::Mat img) {
  std::vector<OCRPredictResult> ocr_result;

  // detect the sentence in input image
  this->detector_->Run(img, ocr_result);
  // crop image
  std::vector<cv::Mat> img_list;
  for (int j = 0; j < ocr_result.size(); j++) {
    cv::Mat crop_img;
    crop_img = Utility::GetRotateCropImage(img, ocr_result[j].box);
    img_list.push_back(crop_img);
  }

  if (this->classifier_ != nullptr) {
    // find the reversed sentence and flip it
    this->classifier_->Run(img_list, ocr_result);
    for (int i = 0; i < img_list.size(); i++) {
      if (ocr_result[i].cls_label % 2 == 1 &&
          ocr_result[i].cls_score > classifier_->cls_thresh) {
        cv::rotate(img_list[i], img_list[i], 1);
      }
    }
  }

  // recognize the words in sentence and print them
  this->recognizer_->Run(img_list, ocr_result);

  return ocr_result;
}

PPOCR::~PPOCR() {
  if (this->detector_ != nullptr) {
    delete this->detector_;
  }
  if (this->classifier_ != nullptr) {
    delete this->classifier_;
  }
  if (this->recognizer_ != nullptr) {
    delete this->recognizer_;
  }
}
} // namespace PaddleOCR
