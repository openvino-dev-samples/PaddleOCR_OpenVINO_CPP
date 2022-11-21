#pragma once

#include <vector>
#include <string>
#include <map>

struct MeterConfig
{
  float scale_interval_value_;
  float range_;
  std::string unit_;

  MeterConfig() {}

  MeterConfig(const float &scale_interval_value,
              const float &range,
              const std::string &unit) : scale_interval_value_(scale_interval_value),
                                         range_(range), unit_(unit) {}
};

struct MeterResult
{
  // the number of scales
  int num_scales_;
  // the pointer location relative to the scales
  float pointed_scale_;

  MeterResult() {}

  MeterResult(const int &num_scales, const float &pointed_scale) : num_scales_(num_scales), pointed_scale_(pointed_scale) {}
};

// The size of inputting images of the segmenter.
extern std::vector<int> METER_SHAPE; // height x width
// Center of a circular meter
extern std::vector<int> CIRCLE_CENTER; // height x width
// Radius of a circular meter
extern int CIRCLE_RADIUS;
extern float PI;

// During the postprocess phase, annulus formed by the radius from
// 130 to 250 of a circular meter will be converted to a rectangle.
// So the height of the rectangle is 120.
extern int RECTANGLE_HEIGHT;
// The width of the rectangle is 1570, that is to say the perimeter
// of a circular meter.
extern int RECTANGLE_WIDTH;

// The configuration information of a meter,
// composed of scale value, range, unit
extern int TYPE_THRESHOLD;
extern std::vector<MeterConfig> METER_CONFIG;
extern std::map<std::string, uint8_t> SEG_CNAME2CLSID;