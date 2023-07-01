#include "include/ncnn/layer.h"
#include "include/ncnn/net.h"
#include "modules/perception/common/common.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#define MAX_STRIDE 64

namespace perception {

class YOLOP {
public:
  YOLOP(std::string param_file, std::string bin_file,
        const int target_size = 640, const float prob_threshold = 0.30f,
        const float nms_threshold = 0.45f)
      : param_file_(param_file), bin_file_(bin_file), target_size_(target_size),
        prob_threshold_(prob_threshold), nms_threshold_(nms_threshold) {
    yolopv2.load_param(param_file_.c_str());
    yolopv2.load_model(bin_file_.c_str());
  };

  ~YOLOP(){};

  std::vector<Object> GetObjects();
  cv::Mat GetLaneMask();
  cv::Mat GetAccessibleAreaMask();
  cv::Mat GetVisualization(const cv::Mat &bgr);

  void Detect(cv::Mat &bgr);

private:
  inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
  };

  void Slice(const ncnn::Mat &in, ncnn::Mat &out, int start, int end, int axis);
  void Interp(const ncnn::Mat &in, const float &scale, const int &out_w,
              const int &out_h, ncnn::Mat &out);
  void QsortDescentInplace(std::vector<Object> &faceobjects, int left,
                           int right);
  void QsortDescentInplace(std::vector<Object> &faceobjects);
  void NmsSortedBboxes(const std::vector<Object> &faceobjects,
                       std::vector<int> &picked, float nms_threshold);
  void GenerateProposals(const ncnn::Mat &anchors, int stride,
                         const ncnn::Mat &in_pad, const ncnn::Mat &feat_blob,
                         float prob_threshold, std::vector<Object> &objects);

private:
  ncnn::Net yolopv2;
  std::string param_file_;
  std::string bin_file_;

  ncnn::Mat da_seg_mask_, ll_seg_mask_;
  std::vector<Object> objects_;

  int target_size_ = 640;
  float prob_threshold_ = 0.30f;
  float nms_threshold_ = 0.45f;
};

} // namespace perception