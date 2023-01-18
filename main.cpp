#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <include/args.h>
#include <include/paddleocr.h>
#include <include/paddlestructure.h>
#include <gflags/gflags.h>

using namespace PaddleOCR;

void check_params()
{
  if (FLAGS_type == "ocr")
  {
    if (FLAGS_det_model_dir.empty() || FLAGS_rec_model_dir.empty())
    {
      std::cout << "Need a path to detection and recogition model"
                   "[Usage] --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --rec_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << std::endl;
      exit(1);
    }
  }
  else if (FLAGS_type == "structure")
  {
    if (FLAGS_det_model_dir.empty() || FLAGS_rec_model_dir.empty() || FLAGS_lay_model_dir.empty() || FLAGS_tab_model_dir.empty())
    {
      std::cout << "Need a path to detection, recogition, layout and table model"
                   "[Usage] --det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --rec_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --lay_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ --tab_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                << std::endl;
      exit(1);
    }
  }
}

int main(int argc, char *argv[])
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  check_params();

  // read image
  cv::Mat src_img = imread(FLAGS_input);

  if (FLAGS_type == "ocr")
  {
    PPOCR ppocr;
    std::vector<OCRPredictResult> ocr_result = ppocr.ocr(src_img);
    Utility::print_result(ocr_result);
    Utility::VisualizeBboxes(src_img, ocr_result,
                             "./ocr_result.jpg");
  }
  else if (FLAGS_type == "structure")
  {

    PaddleStructure paddlestructure;
    std::vector<StructurePredictResult> structure_results = paddlestructure.structure(src_img);
    for (int j = 0; j < structure_results.size(); j++)
    {
      std::cout << j << "\ttype: " << structure_results[j].type
                << ", region: [";
      std::cout << structure_results[j].box[0] << ","
                << structure_results[j].box[1] << ","
                << structure_results[j].box[2] << ","
                << structure_results[j].box[3] << "], score: ";
      std::cout << structure_results[j].confidence << ", res: ";

      if (structure_results[j].type == "table")
      {
        std::cout << structure_results[j].html << std::endl;
        if (structure_results[j].cell_box.size() > 0)
        {
          Utility::VisualizeBboxes(src_img, structure_results[j],
                                   "./structure_result" + std::to_string(j) + ".jpg");
        }
      }
      else
      {
        std::cout << "count of ocr result is : "
                  << structure_results[j].text_res.size() << std::endl;
        if (structure_results[j].text_res.size() > 0)
        {
          std::cout << "********** print ocr result "
                    << "**********" << std::endl;
          Utility::print_result(structure_results[j].text_res);
          std::cout << "********** end print ocr result "
                    << "**********" << std::endl;
        }
      }
    }
  }
  else
  {
    std::cout << "only value in ['ocr','structure'] is supported" << std::endl;
  }
}