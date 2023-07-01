#pragma once

#include "include/BYTETracker.h"
#include "modules/perception/yolox-v3/yolox.h"

namespace perception {

class ByteTrack{
public:
    ByteTrack(const std::string &engine_file_path){
        yolox_ptr_ = std::make_shared<YOLOX>(engine_file_path);
        tracker_ptr = std::make_shared<BYTETracker>();
    };

    ~ByteTrack(){};

    void Detect(cv::Mat &image);

    std::vector<Object> GetObjects() { return objects_; };

    cv::Mat GetVisualization(const cv::Mat &bgr);

private:
    int fps = 30;
    std::shared_ptr<BYTETracker> tracker_ptr = nullptr;
    std::shared_ptr<YOLOX> yolox_ptr_= nullptr;
    std::vector<STrack> output_stracks_{};
    std::vector<Object> objects_{};
};

}