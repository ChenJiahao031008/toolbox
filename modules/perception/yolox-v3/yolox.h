#pragma once

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "modules/perception/common/common.h"
#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>
#include "common/timer.h"

#define CHECK(status)                                                          \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      std::cerr << "Cuda failure: " << ret << std::endl;                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)
#define DEVICE 0 // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

using namespace nvinfer1;

namespace perception {

struct GridAndStride {
  int grid0;
  int grid1;
  int stride;
};

class YOLOX {
public:
  YOLOX(const std::string &engine_file_path);

  ~YOLOX();

  void Detect(cv::Mat &image);

  std::vector<Object> GetObjects() { return objects; };

  cv::Mat GetVisualization(const cv::Mat &bgr);

private:
  cv::Mat StaticResize(cv::Mat &img);

  void GenerateGridsAndStride(std::vector<int> &strides,
                              std::vector<GridAndStride> &grid_strides);

  void QsortDescentInplace(std::vector<Object> &faceobjects, int left,
                           int right);

  void QsortDescentInplace(std::vector<Object> &objects);

  void NmsSortedBboxes(const std::vector<Object> &faceobjects,
                       std::vector<int> &picked, float nms_threshold);

  void GenerateYoloxProposals(std::vector<GridAndStride> grid_strides,
                              float *feat_blob, float prob_threshold,
                              std::vector<Object> &objects);

  float *BlobFromImage(cv::Mat &img);

  void DecodeOutputs(float *prob, std::vector<Object> &objects, float scale,
                     const int img_w, const int img_h);

  void DoInference(IExecutionContext &context, float *input, float *output,
                   const int output_size, cv::Size input_shape);

private:
  const int INPUT_W = 640;
  const int INPUT_H = 640;
  const int NUM_CLASSES = 80;

  const char *INPUT_BLOB_NAME = "input_0";
  const char *OUTPUT_BLOB_NAME = "output_0";

  Logger gLogger;
  IRuntime *runtime = nullptr;
  ICudaEngine *engine = nullptr;
  IExecutionContext *context = nullptr;

  float *prob;
  int output_size = 1;
  std::vector<Object> objects;
};

static const char *class_names[] = {"person",        "bicycle",      "car",
                             "motorcycle",    "airplane",     "bus",
                             "train",         "truck",        "boat",
                             "traffic light", "fire hydrant", "stop sign",
                             "parking meter", "bench",        "bird",
                             "cat",           "dog",          "horse",
                             "sheep",         "cow",          "elephant",
                             "bear",          "zebra",        "giraffe",
                             "backpack",      "umbrella",     "handbag",
                             "tie",           "suitcase",     "frisbee",
                             "skis",          "snowboard",    "sports ball",
                             "kite",          "baseball bat", "baseball glove",
                             "skateboard",    "surfboard",    "tennis racket",
                             "bottle",        "wine glass",   "cup",
                             "fork",          "knife",        "spoon",
                             "bowl",          "banana",       "apple",
                             "sandwich",      "orange",       "broccoli",
                             "carrot",        "hot dog",      "pizza",
                             "donut",         "cake",         "chair",
                             "couch",         "potted plant", "bed",
                             "dining table",  "toilet",       "tv",
                             "laptop",        "mouse",        "remote",
                             "keyboard",      "cell phone",   "microwave",
                             "oven",          "toaster",      "sink",
                             "refrigerator",  "book",         "clock",
                             "vase",          "scissors",     "teddy bear",
                             "hair drier",    "toothbrush"};

static const float color_list[80][3] = {
    {0.000, 0.447, 0.741}, {0.850, 0.325, 0.098}, {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556}, {0.466, 0.674, 0.188}, {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184}, {0.300, 0.300, 0.300}, {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000}, {1.000, 0.500, 0.000}, {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000}, {0.000, 0.000, 1.000}, {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000}, {0.333, 0.667, 0.000}, {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000}, {0.667, 0.667, 0.000}, {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000}, {1.000, 0.667, 0.000}, {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500}, {0.000, 0.667, 0.500}, {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500}, {0.333, 0.333, 0.500}, {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500}, {0.667, 0.000, 0.500}, {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500}, {0.667, 1.000, 0.500}, {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500}, {1.000, 0.667, 0.500}, {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000}, {0.000, 0.667, 1.000}, {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000}, {0.333, 0.333, 1.000}, {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000}, {0.667, 0.000, 1.000}, {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000}, {0.667, 1.000, 1.000}, {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000}, {1.000, 0.667, 1.000}, {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000}, {0.667, 0.000, 0.000}, {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000}, {0.000, 0.167, 0.000}, {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000}, {0.000, 0.667, 0.000}, {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000}, {0.000, 0.000, 0.167}, {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500}, {0.000, 0.000, 0.667}, {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000}, {0.000, 0.000, 0.000}, {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286}, {0.429, 0.429, 0.429}, {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714}, {0.857, 0.857, 0.857}, {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741}, {0.50, 0.5, 0}};

} // namespace perception