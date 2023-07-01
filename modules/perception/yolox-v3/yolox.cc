#include "yolox.h"

namespace perception {

YOLOX::YOLOX(const std::string &engine_file_path) {
  cudaSetDevice(DEVICE);
  char *trtModelStream = nullptr;
  size_t size = 0;

  std::ifstream file(engine_file_path, std::ios::binary);
  if (file.good()) {
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
  }

  runtime = createInferRuntime(gLogger);
  engine = runtime->deserializeCudaEngine(trtModelStream, size);
  context = engine->createExecutionContext();
  assert(runtime != nullptr);
  assert(engine != nullptr);
  assert(context != nullptr);
  delete[] trtModelStream;

  auto out_dims = engine->getBindingDimensions(1);

  for (int j = 0; j < out_dims.nbDims; j++) {
    output_size *= out_dims.d[j];
  }
  prob = new float[output_size];
}

YOLOX::~YOLOX() {
  context->destroy();
  engine->destroy();
  runtime->destroy();
}

void YOLOX::Detect(cv::Mat &image) {
  int img_w = image.cols;
  int img_h = image.rows;
  cv::Mat image_ptr = StaticResize(image);
  float *blob = BlobFromImage(image_ptr);
  float scale = std::min(INPUT_W / (img_w * 1.0), INPUT_H / (img_h * 1.0));
  DoInference(*context, blob, prob, output_size, image_ptr.size());
  DecodeOutputs(prob, objects, scale, img_w, img_h);
  delete blob;
}

cv::Mat YOLOX::StaticResize(cv::Mat &img) {
  float r = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
  // r = std::min(r, 1.0f);
  int unpad_w = r * img.cols;
  int unpad_h = r * img.rows;
  cv::Mat re(unpad_h, unpad_w, CV_8UC3);
  cv::resize(img, re, re.size());
  cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
  re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
  return out;
}

void YOLOX::GenerateGridsAndStride(std::vector<int> &strides,
                                   std::vector<GridAndStride> &grid_strides) {
  for (auto stride : strides) {
    int num_grid_y = INPUT_H / stride;
    int num_grid_x = INPUT_W / stride;
    for (int g1 = 0; g1 < num_grid_y; g1++) {
      for (int g0 = 0; g0 < num_grid_x; g0++) {
        grid_strides.push_back((GridAndStride){g0, g1, stride});
      }
    }
  }
}

void YOLOX::QsortDescentInplace(std::vector<Object> &faceobjects, int left,
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

void YOLOX::QsortDescentInplace(std::vector<Object> &objects) {
  if (objects.empty())
    return;

  QsortDescentInplace(objects, 0, objects.size() - 1);
}

void YOLOX::NmsSortedBboxes(const std::vector<Object> &faceobjects,
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

void YOLOX::GenerateYoloxProposals(std::vector<GridAndStride> grid_strides,
                                   float *feat_blob, float prob_threshold,
                                   std::vector<Object> &objects) {
  const int num_anchors = grid_strides.size();

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

    // yolox/models/yolo_head.py decode logic
    float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
    float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
    float w = exp(feat_blob[basic_pos + 2]) * stride;
    float h = exp(feat_blob[basic_pos + 3]) * stride;
    float x0 = x_center - w * 0.5f;
    float y0 = y_center - h * 0.5f;

    float box_objectness = feat_blob[basic_pos + 4];
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++) {
      float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > prob_threshold) {
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = class_idx;
        obj.prob = box_prob;

        objects.push_back(obj);
      }

    } // class loop

  } // point anchor loop
}

float* YOLOX::BlobFromImage(cv::Mat &img) {
  float *blob = new float[img.total() * 3];
  int channels = 3;
  int img_h = img.rows;
  int img_w = img.cols;
  for (size_t c = 0; c < channels; c++) {
    for (size_t h = 0; h < img_h; h++) {
      for (size_t w = 0; w < img_w; w++) {
        blob[c * img_w * img_h + h * img_w + w] =
            (float)img.at<cv::Vec3b>(h, w)[c];
      }
    }
  }
  return blob;
}

void YOLOX::DecodeOutputs(float *prob, std::vector<Object> &objects,
                          float scale, const int img_w, const int img_h) {
  std::vector<Object> proposals;
  std::vector<int> strides = {8, 16, 32};
  std::vector<GridAndStride> grid_strides;
  GenerateGridsAndStride(strides, grid_strides);
  GenerateYoloxProposals(grid_strides, prob, BBOX_CONF_THRESH, proposals);

  QsortDescentInplace(proposals);

  std::vector<int> picked;
  NmsSortedBboxes(proposals, picked, NMS_THRESH);

  int count = picked.size();

  objects.resize(count);
  for (int i = 0; i < count; i++) {
    objects[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects[i].rect.x) / scale;
    float y0 = (objects[i].rect.y) / scale;
    float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
    float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

    // clip
    x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

    objects[i].rect.x = x0;
    objects[i].rect.y = y0;
    objects[i].rect.width = x1 - x0;
    objects[i].rect.height = y1 - y0;
  }
}

cv::Mat YOLOX::GetVisualization(const cv::Mat &bgr) {

  cv::Mat image = bgr.clone();

  for (size_t i = 0; i < objects.size(); i++) {
    const Object &obj = objects[i];

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

void YOLOX::DoInference(IExecutionContext &context, float *input, float *output,
                        const int output_size, cv::Size input_shape) {
  const ICudaEngine &engine = context.getEngine();

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbBindings() == 2);
  void *buffers[2];

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

  assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
  const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
  assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
  int mBatchSize = engine.getMaxBatchSize();

  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex],
                   3 * input_shape.height * input_shape.width * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));

  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA
  // output back to host
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
                        3 * input_shape.height * input_shape.width *
                            sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  context.enqueue(1, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex],
                        output_size * sizeof(float), cudaMemcpyDeviceToHost,
                        stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}
} // namespace perception