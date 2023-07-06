#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace perception {

struct Object {
  cv::Rect_<float> rect;
  int label;
  float prob;
};

inline float GetIntersectionArea(const Object &a, const Object &b) {
  cv::Rect_<float> inter = a.rect & b.rect;
  return inter.area();
}

} // namespace perception