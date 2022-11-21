
#include "include/meter_config.h"

std::vector<int> METER_SHAPE = {512, 512};  // height x width
std::vector<int> CIRCLE_CENTER = {256, 256};
int CIRCLE_RADIUS = 250;
float PI = 3.1415926536;
int RECTANGLE_HEIGHT = 120;
int RECTANGLE_WIDTH = 1570;

int TYPE_THRESHOLD = 40;
std::vector<MeterConfig> METER_CONFIG = {
  MeterConfig(25.0f/50.0f, 25.0f, "(MPa)"),
  MeterConfig(1.6f/32.0f,  1.6f,   "(MPa)")
};

std::map<std::string, uint8_t> SEG_CNAME2CLSID = {
  {"background", 0}, {"pointer", 1}, {"scale", 2}
};