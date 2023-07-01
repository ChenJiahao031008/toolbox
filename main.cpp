#include <iomanip>
#include <iostream>

#include "common/file_manager.h"
#include "common/logger.hpp"
#include "common/timer.h"
#include "config/config.h"
#include "modules/perception/yolop-v2/yolop.h"
#include "modules/perception/yolox-v3/yolox.h"
#include "modules/perception/bytetrack/bytetrack.h"
#include <interpreter.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#ifndef LOG_CORE_DUMP_CAPTURE
#define BACKWARD_HAS_DW 1
#endif

int main(int argc, char **argv) {
  common::Logger logger(argc, argv);
  std::string current_dir = common::FileManager::GetCurrentDir();
  GINFO << "Cuurent Dir: " << current_dir;

  ezcfg::Interpreter itp(current_dir + "/../config/config.txt", true);
  Config conf;
  itp.parse(conf);
  GINFO << conf.example_name;

  // 测试backward-cpp模块的core dump捕获功能
  // char* c = "hello world";
  // c[1] = 'H';

  // 测试yolopv2
  std::string model_path =
      current_dir + "/../modules/perception/yolop-v2/models/";
  std::string output_path = current_dir + "/../output/";

  perception::YOLOP network(model_path + "yolopv2.param",
                            model_path + "yolopv2.bin");
  cv::Mat image = cv::imread(current_dir + "/../data/example.jpg", 1);
  auto &timer = common::Timer::GetInstance();
  timer.Evaluate([&, &network]() { network.Detect(image); }, "    YOLOP v2");

  cv::Mat visualization = network.GetVisualization(image);
  cv::Mat lane_mask = network.GetLaneMask();
  cv::Mat accessible_area_mask = network.GetAccessibleAreaMask();
  cv::imwrite(output_path + "/yolop_visualization.png", visualization);
  cv::imwrite(output_path + "/yolop_lane_mask.png", lane_mask);
  cv::imwrite(output_path + "/yolop_accessible_area_mask.png",
              accessible_area_mask);

  // 测试 yoloxv3
  std::string engine_path =
      current_dir + "/../modules/perception/yolox-v3/models/model_trt.engine";
  perception::YOLOX network2(engine_path);
  timer.Evaluate([&, &network2]() { network2.Detect(image); }, "    YOLOX v3");
  cv::Mat visualization2 = network2.GetVisualization(image);
  cv::imwrite(output_path + "/yolox_visualization2.png", visualization2);

  // 测试bytetrack
  std::string input_video_path = current_dir + "/../data/example.mp4";
  cv::VideoCapture cap(input_video_path);
  if (!cap.isOpened()) {
    GERROR << "Video is NOT Open.";
  }
  cv::VideoWriter writer(
      output_path + "/demo.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30,
      cv::Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

  perception::ByteTrack bytetrack(engine_path);
  while (true) {
    if (!cap.read(image)) break;
    timer.Evaluate([&, &bytetrack]() { bytetrack.Detect(image); }, "   ByteTrack");
    cv::Mat visualization3 = bytetrack.GetVisualization(image);
    writer.write(visualization3);
  }

  // 统计时间
  timer.PrintAll();

  return 0;
}
