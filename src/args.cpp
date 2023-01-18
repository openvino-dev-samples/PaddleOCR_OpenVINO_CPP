#include <gflags/gflags.h>

DEFINE_string(input, "", "Required. Path to image file");
DEFINE_string(type, "", "Required. Task type ('ocr' or 'structure')");
DEFINE_string(output_dir, "./", "Path to output results.");
DEFINE_string(det_model_dir, "", "Path to detection model file");
DEFINE_string(cls_model_dir, "", "Path to classification model file");
DEFINE_string(rec_model_dir, "", "Path to recognition model file");
DEFINE_string(lay_model_dir, "", "Path to layout model file");
DEFINE_string(tab_model_dir, "", "Path to table model file");
DEFINE_string(label_dir, "", "Required. Path to label file");
DEFINE_string(layout_dict_dir,
              "/home/ethan/PaddleOCR_OpenVINO_CPP/data/layout_publaynet_dict.txt",
              "Path of dictionary.");
DEFINE_string(table_dict_dir,
              "/home/ethan/PaddleOCR_OpenVINO_CPP/data/table_structure_dict.txt",
              "Path of dictionary.");