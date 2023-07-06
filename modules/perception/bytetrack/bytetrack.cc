#include "bytetrack.h"

namespace perception {

void ByteTrack::Detect(cv::Mat &image){
    objects_.clear();

    yolox_ptr_->Detect(image);
    vector<Object> yolox_objects = yolox_ptr_->GetObjects();
    output_stracks_ = tracker_ptr->update(yolox_objects);

    for (auto &strack : output_stracks_)
    {
        if (strack.state == TrackState::Tracked)
        {
            Object object;
            object.rect = cv::Rect_<float>(strack.tlwh[0], strack.tlwh[1], strack.tlwh[2], strack.tlwh[3]);
            object.label = strack.label;
            object.prob = strack.score;
            objects_.push_back(object);
        }
    }

}

cv::Mat ByteTrack::GetVisualization(const cv::Mat &bgr){
    cv::Mat image = bgr.clone();

  for (size_t i = 0; i < objects_.size(); i++) {
    const Object &obj = objects_[i];

    // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
    //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

    cv::Scalar color =
        cv::Scalar(color_list[obj.label][0], color_list[obj.label][1],
                   color_list[obj.label][2]);
    float c_mean = cv::mean(color)[0];
    cv::Scalar txt_color;
    if (c_mean > 0.5) {
      txt_color = cv::Scalar(0, 0, 0);
    } else {
      txt_color = cv::Scalar(255, 255, 255);
    }

    cv::rectangle(image, obj.rect, color * 255, 2);

    char text[256];
    sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7 * 255;

    int x = obj.rect.x;
    int y = obj.rect.y + 1;
    if (y > image.rows)
      y = image.rows;

    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        txt_bk_color, -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
  }
  return image;
}

}  // namespace perception
