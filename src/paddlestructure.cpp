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
#include <include/paddlestructure.h>

namespace PaddleOCR {

PaddleStructure::PaddleStructure() {
  this->layout_model_ = new Layout(FLAGS_lay_model_dir, FLAGS_layout_dict_dir);
  this->table_model_ = new Table(FLAGS_tab_model_dir, FLAGS_table_dict_dir);
};

std::vector<StructurePredictResult> PaddleStructure::structure(cv::Mat src_img) {
  std::vector<StructurePredictResult> structure_results;

  this->layout_model_->Run(src_img, structure_results);
  cv::Mat roi_img;
  for (int i = 0; i < structure_results.size(); i++) {
    // crop image
    roi_img = Utility::crop_image(src_img, structure_results[i].box);
    if (structure_results[i].type == "table"){
      std::vector<std::vector<std::string>> structure_html_tags;
      std::vector<float> structure_scores(1, 0);
      std::vector<std::vector<std::vector<int>>> structure_boxes;
      std::vector<cv::Mat> img_list;

      img_list.push_back(roi_img);
      this->table_model_->Run(img_list, structure_html_tags, structure_scores, structure_boxes);
      std::vector<OCRPredictResult> ocr_result;
      std::string html;
      int expand_pixel = 3;

      for (int j = 0; j < img_list.size(); j++) {
        this->detector_->Run(img_list[j], ocr_result);
        // crop image
        std::vector<cv::Mat> rec_img_list;
        std::vector<int> ocr_box;
        for (int k = 0; k < ocr_result.size(); k++) {
          ocr_box = Utility::xyxyxyxy2xyxy(ocr_result[k].box);
          ocr_box[0] = std::max(0, ocr_box[0] - expand_pixel);
          ocr_box[1] = std::max(0, ocr_box[1] - expand_pixel),
          ocr_box[2] = std::min(img_list[j].cols, ocr_box[2] + expand_pixel);
          ocr_box[3] = std::min(img_list[j].rows, ocr_box[3] + expand_pixel);

          cv::Mat crop_img = Utility::crop_image(img_list[j], ocr_box);
          rec_img_list.push_back(crop_img);
        }
        // rec
        this->recognizer_->Run(rec_img_list, ocr_result);
        // rebuild table
        html = this->rebuild_table(structure_html_tags[j], structure_boxes[j],
                                    ocr_result);
        structure_results[i].html = html;
        structure_results[i].cell_box = structure_boxes[j];
        structure_results[i].html_score = structure_scores[j];
      }
    }
    else {
      structure_results[i].text_res = ocr(roi_img);
    }
  }
  return structure_results;
};

std::string
PaddleStructure::rebuild_table(std::vector<std::string> structure_html_tags,
                               std::vector<std::vector<int>> structure_boxes,
                               std::vector<OCRPredictResult> &ocr_result) {
  // match text in same cell
  std::vector<std::vector<std::string>> matched(structure_boxes.size(),
                                                std::vector<std::string>());

  std::vector<int> ocr_box;
  std::vector<int> structure_box;
  for (int i = 0; i < ocr_result.size(); i++) {
    ocr_box = Utility::xyxyxyxy2xyxy(ocr_result[i].box);
    ocr_box[0] -= 1;
    ocr_box[1] -= 1;
    ocr_box[2] += 1;
    ocr_box[3] += 1;
    std::vector<std::vector<float>> dis_list(structure_boxes.size(),
                                             std::vector<float>(3, 100000.0));
    for (int j = 0; j < structure_boxes.size(); j++) {
      if (structure_boxes[i].size() == 8) {
        structure_box = Utility::xyxyxyxy2xyxy(structure_boxes[j]);
      } else {
        structure_box = structure_boxes[j];
      }
      dis_list[j][0] = this->dis(ocr_box, structure_box);
      dis_list[j][1] = 1 - Utility::iou(ocr_box, structure_box);
      dis_list[j][2] = j;
    }
    // find min dis idx
    std::sort(dis_list.begin(), dis_list.end(),
              PaddleStructure::comparison_dis);
    matched[dis_list[0][2]].push_back(ocr_result[i].text);
  }

  // get pred html
  std::string html_str = "";
  int td_tag_idx = 0;
  for (int i = 0; i < structure_html_tags.size(); i++) {
    if (structure_html_tags[i].find("</td>") != std::string::npos) {
      if (structure_html_tags[i].find("<td></td>") != std::string::npos) {
        html_str += "<td>";
      }
      if (matched[td_tag_idx].size() > 0) {
        bool b_with = false;
        if (matched[td_tag_idx][0].find("<b>") != std::string::npos &&
            matched[td_tag_idx].size() > 1) {
          b_with = true;
          html_str += "<b>";
        }
        for (int j = 0; j < matched[td_tag_idx].size(); j++) {
          std::string content = matched[td_tag_idx][j];
          if (matched[td_tag_idx].size() > 1) {
            // remove blank, <b> and </b>
            if (content.length() > 0 && content.at(0) == ' ') {
              content = content.substr(0);
            }
            if (content.length() > 2 && content.substr(0, 3) == "<b>") {
              content = content.substr(3);
            }
            if (content.length() > 4 &&
                content.substr(content.length() - 4) == "</b>") {
              content = content.substr(0, content.length() - 4);
            }
            if (content.empty()) {
              continue;
            }
            // add blank
            if (j != matched[td_tag_idx].size() - 1 &&
                content.at(content.length() - 1) != ' ') {
              content += ' ';
            }
          }
          html_str += content;
        }
        if (b_with) {
          html_str += "</b>";
        }
      }
      if (structure_html_tags[i].find("<td></td>") != std::string::npos) {
        html_str += "</td>";
      } else {
        html_str += structure_html_tags[i];
      }
      td_tag_idx += 1;
    } else {
      html_str += structure_html_tags[i];
    }
  }
  return html_str;
}

float PaddleStructure::dis(std::vector<int> &box1, std::vector<int> &box2) {
  int x1_1 = box1[0];
  int y1_1 = box1[1];
  int x2_1 = box1[2];
  int y2_1 = box1[3];

  int x1_2 = box2[0];
  int y1_2 = box2[1];
  int x2_2 = box2[2];
  int y2_2 = box2[3];

  float dis =
      abs(x1_2 - x1_1) + abs(y1_2 - y1_1) + abs(x2_2 - x2_1) + abs(y2_2 - y2_1);
  float dis_2 = abs(x1_2 - x1_1) + abs(y1_2 - y1_1);
  float dis_3 = abs(x2_2 - x2_1) + abs(y2_2 - y2_1);
  return dis + std::min(dis_2, dis_3);
}

PaddleStructure::~PaddleStructure() {
  if (this->layout_model_ != nullptr) {
    delete this->layout_model_;
  }
  if (this->table_model_ != nullptr) {
    delete this->table_model_;
  }
};
} // namespace PaddleOCR