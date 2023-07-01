#include "yolop.h"
#include "common/timer.h"

namespace perception {

void YOLOP::Slice(const ncnn::Mat &in, ncnn::Mat &out, int start, int end,
                  int axis) {
  ncnn::Option opt;
  opt.num_threads = 4;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = false;

  ncnn::Layer *op = ncnn::create_layer("Crop");

  // set param
  ncnn::ParamDict pd;

  ncnn::Mat axes = ncnn::Mat(1);
  axes.fill(axis);
  ncnn::Mat ends = ncnn::Mat(1);
  ends.fill(end);
  ncnn::Mat starts = ncnn::Mat(1);
  starts.fill(start);
  pd.set(9, starts); // start
  pd.set(10, ends);  // end
  pd.set(11, axes);  // axes

  op->load_param(pd);

  op->create_pipeline(opt);

  // forward
  op->forward(in, out, opt);

  op->destroy_pipeline(opt);

  delete op;
}

void YOLOP::Interp(const ncnn::Mat &in, const float &scale, const int &out_w,
                   const int &out_h, ncnn::Mat &out) {
  ncnn::Option opt;
  opt.num_threads = 4;
  opt.use_fp16_storage = false;
  opt.use_packing_layout = false;

  ncnn::Layer *op = ncnn::create_layer("Interp");

  // set param
  ncnn::ParamDict pd;
  pd.set(0, 2);     // resize_type
  pd.set(1, scale); // height_scale
  pd.set(2, scale); // width_scale
  pd.set(3, out_h); // height
  pd.set(4, out_w); // width

  op->load_param(pd);

  op->create_pipeline(opt);

  // forward
  op->forward(in, out, opt);

  op->destroy_pipeline(opt);

  delete op;
}

void YOLOP::QsortDescentInplace(std::vector<Object> &faceobjects, int left,
                                int right) {
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].prob;

  while (i <= j) {
    while (faceobjects[i].prob > p)
      i++;

    while (faceobjects[j].prob < p)
      j--;

    if (i <= j) {
      // swap
      std::swap(faceobjects[i], faceobjects[j]);

      i++;
      j--;
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (left < j)
        QsortDescentInplace(faceobjects, left, j);
    }
#pragma omp section
    {
      if (i < right)
        QsortDescentInplace(faceobjects, i, right);
    }
  }
}

void YOLOP::QsortDescentInplace(std::vector<Object> &faceobjects) {
  if (faceobjects.empty())
    return;

  QsortDescentInplace(faceobjects, 0, faceobjects.size() - 1);
}

void YOLOP::NmsSortedBboxes(const std::vector<Object> &faceobjects,
                            std::vector<int> &picked, float nms_threshold) {
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].rect.area();
  }

  for (int i = 0; i < n; i++) {
    const Object &a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object &b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = GetIntersectionArea(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold)
        keep = 0;
    }

    if (keep)
      picked.push_back(i);
  }
}

void YOLOP::GenerateProposals(const ncnn::Mat &anchors, int stride,
                              const ncnn::Mat &in_pad,
                              const ncnn::Mat &feat_blob, float prob_threshold,
                              std::vector<Object> &objects) {
  const int num_grid = feat_blob.h;

  int num_grid_x;
  int num_grid_y;
  if (in_pad.w > in_pad.h) {
    num_grid_x = in_pad.w / stride;
    num_grid_y = num_grid / num_grid_x;
  } else {
    num_grid_y = in_pad.h / stride;
    num_grid_x = num_grid / num_grid_y;
  }

  const int num_class = feat_blob.w - 5;

  const int num_anchors = anchors.w / 2;

  for (int q = 0; q < num_anchors; q++) {
    const float anchor_w = anchors[q * 2];
    const float anchor_h = anchors[q * 2 + 1];

    const ncnn::Mat feat = feat_blob.channel(q);

    for (int i = 0; i < num_grid_y; i++) {
      for (int j = 0; j < num_grid_x; j++) {
        const float *featptr = feat.row(i * num_grid_x + j);

        // find class index with max class score
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int k = 0; k < num_class; k++) {
          float score = featptr[5 + k];
          if (score > class_score) {
            class_index = k;
            class_score = score;
          }
        }

        float box_score = featptr[4];

        float confidence = sigmoid(box_score) * sigmoid(class_score);

        if (confidence >= prob_threshold) {
          // yolov5/models/yolo.py Detect forward
          // y = x[i].sigmoid()
          // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
          // self.grid[i].to(x[i].device)) * self.stride[i]  # xy y[..., 2:4] =
          // (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

          float dx = sigmoid(featptr[0]);
          float dy = sigmoid(featptr[1]);
          float dw = sigmoid(featptr[2]);
          float dh = sigmoid(featptr[3]);

          float pb_cx = (dx * 2.f - 0.5f + j) * stride;
          float pb_cy = (dy * 2.f - 0.5f + i) * stride;

          float pb_w = pow(dw * 2.f, 2) * anchor_w;
          float pb_h = pow(dh * 2.f, 2) * anchor_h;

          float x0 = pb_cx - pb_w * 0.5f;
          float y0 = pb_cy - pb_h * 0.5f;
          float x1 = pb_cx + pb_w * 0.5f;
          float y1 = pb_cy + pb_h * 0.5f;

          Object obj;
          obj.rect.x = x0;
          obj.rect.y = y0;
          obj.rect.width = x1 - x0;
          obj.rect.height = y1 - y0;
          obj.label = class_index;
          obj.prob = confidence;

          objects.push_back(obj);
        }
      }
    }
  }
}

cv::Mat YOLOP::GetVisualization(const cv::Mat &bgr) {

  cv::Mat image = bgr.clone();

  for (size_t i = 0; i < objects_.size(); i++) {
    const Object &obj = objects_[i];

    // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label,
    // obj.prob,
    //         obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

    cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 0));

    char text[256];
    sprintf(text, "%.1f%%", obj.prob * 100);

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = obj.rect.x;
    int y = obj.rect.y - label_size.height - baseLine;
    if (y < 0)
      y = 0;
    if (x + label_size.width > image.cols)
      x = image.cols - label_size.width;

    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        cv::Scalar(255, 255, 255), -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }
  const float *da_ptr = (float *)da_seg_mask_.data;
  const float *ll_ptr = (float *)ll_seg_mask_.data;
  int w = da_seg_mask_.w;
  int h = da_seg_mask_.h;
  for (int i = 0; i < h; i++) {
    cv::Vec3b *image_ptr = image.ptr<cv::Vec3b>(i);
    for (int j = 0; j < w; j++) {
      // 可行驶区域
      if (da_ptr[i * w + j] < da_ptr[w * h + i * w + j]) {
        image_ptr[j] = cv::Vec3b(0, 255, 0);
      }
      // 车道线
      if (std::round(ll_ptr[i * w + j]) == 1.0) {
        image_ptr[j] = cv::Vec3b(255, 0, 0);
      }
    }
  }
  return image;
}

std::vector<Object> YOLOP::GetObjects() { return objects_; }

cv::Mat YOLOP::GetLaneMask() {
  int w = ll_seg_mask_.w;
  int h = ll_seg_mask_.h;
  cv::Mat lane_mask = cv::Mat::zeros(h, w, CV_8UC1);
  const float *ll_ptr = (float *)ll_seg_mask_.data;
  for (int i = 0; i < h; i++) {
    uchar *image_ptr = lane_mask.ptr<uchar>(i);
    for (int j = 0; j < w; j++) {
      if (std::round(ll_ptr[i * w + j]) == 1.0) {
        image_ptr[j] = 255;
      }
    }
  }
  return lane_mask;
}

cv::Mat YOLOP::GetAccessibleAreaMask() {
  int w = da_seg_mask_.w;
  int h = da_seg_mask_.h;
  cv::Mat accessible_area = cv::Mat::zeros(h, w, CV_8UC1);
  const float *da_ptr = (float *)da_seg_mask_.data;
  for (int i = 0; i < h; i++) {
    uchar *image_ptr = accessible_area.ptr<uchar>(i);
    for (int j = 0; j < w; j++) {
      if (da_ptr[i * w + j] < da_ptr[w * h + i * w + j]) {
        image_ptr[j] = 255;
      }
    }
  }
  return accessible_area;
}

void YOLOP::Detect(cv::Mat &bgr) {
  int img_w = bgr.cols;
  int img_h = bgr.rows;
  
  // letterbox pad to multiple of MAX_STRIDE
  int w = img_w;
  int h = img_h;
  float scale = 1.f;
  if (w > h) {
    scale = (float)target_size_ / w;
    w = target_size_;
    h = h * scale;
  } else {
    scale = (float)target_size_ / h;
    h = target_size_;
    w = w * scale;
  }

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(
      bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

  // pad to target_size rectangle
  // yolov5/utils/datasets.py letterbox
  int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
  int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
  ncnn::Mat in_pad;
  ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2,
                         wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
  in_pad.substract_mean_normalize(0, norm_vals);

  ncnn::Extractor ex = yolopv2.create_extractor();

  ex.input("images", in_pad);

  std::vector<Object> proposals;

  // stride 8
  {
    ncnn::Mat out;
    ex.extract("det0", out);

    ncnn::Mat anchors(6);
    anchors[0] = 12.f;
    anchors[1] = 16.f;
    anchors[2] = 19.f;
    anchors[3] = 36.f;
    anchors[4] = 40.f;
    anchors[5] = 28.f;

    std::vector<Object> objects8;
    GenerateProposals(anchors, 8, in, out, prob_threshold_, objects8);

    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
  }

  // stride 16
  {
    ncnn::Mat out;
    ex.extract("det1", out);

    ncnn::Mat anchors(6);
    anchors[0] = 36.f;
    anchors[1] = 75.f;
    anchors[2] = 76.f;
    anchors[3] = 55.f;
    anchors[4] = 72.f;
    anchors[5] = 146.f;

    std::vector<Object> objects16;
    GenerateProposals(anchors, 16, in, out, prob_threshold_, objects16);

    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
  }

  // stride 32
  {
    ncnn::Mat out;
    ex.extract("det2", out);

    ncnn::Mat anchors(6);
    anchors[0] = 142.f;
    anchors[1] = 110.f;
    anchors[2] = 192.f;
    anchors[3] = 243.f;
    anchors[4] = 459.f;
    anchors[5] = 401.f;

    std::vector<Object> objects32;
    GenerateProposals(anchors, 32, in, out, prob_threshold_, objects32);

    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
  }

  ncnn::Mat da, ll;
  {
    ex.extract("677", da);
    ex.extract("769", ll);
    Slice(da, da_seg_mask_, hpad / 2, in_pad.h - hpad / 2, 1);
    Slice(ll, ll_seg_mask_, hpad / 2, in_pad.h - hpad / 2, 1);
    Slice(da_seg_mask_, da_seg_mask_, wpad / 2, in_pad.w - wpad / 2, 2);
    Slice(ll_seg_mask_, ll_seg_mask_, wpad / 2, in_pad.w - wpad / 2, 2);
    Interp(da_seg_mask_, 1 / scale, 0, 0, da_seg_mask_);
    Interp(ll_seg_mask_, 1 / scale, 0, 0, ll_seg_mask_);
  }

  // sort all proposals by score from highest to lowest
  QsortDescentInplace(proposals);

  // apply nms with nms_threshold
  std::vector<int> picked;
  NmsSortedBboxes(proposals, picked, nms_threshold_);

  int count = picked.size();

  objects_.resize(count);
  for (int i = 0; i < count; i++) {
    objects_[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects_[i].rect.x - (wpad / 2)) / scale;
    float y0 = (objects_[i].rect.y - (hpad / 2)) / scale;
    float x1 =
        (objects_[i].rect.x + objects_[i].rect.width - (wpad / 2)) / scale;
    float y1 =
        (objects_[i].rect.y + objects_[i].rect.height - (hpad / 2)) / scale;

    // clip
    x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

    objects_[i].rect.x = x0;
    objects_[i].rect.y = y0;
    objects_[i].rect.width = x1 - x0;
    objects_[i].rect.height = y1 - y0;
  }
}

} // namespace perception