#pragma once
#include <pcl/common/centroid.h>
#include <pcl/conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>

#include "segment.h"

namespace cloud_processing {
class SegmentPlane : public CloudSegmentBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SegmentPlane();
  ~SegmentPlane() = default;

  bool Segment(CloudTypePtr& input_cloud_ptr, CloudTypePtr& segmented_cloud_ptr,
               CloudTypePtr& complementary_cloud_ptr) override;

  std::vector<CloudType> GetPlaneCloudList();

 private:
  std::vector<CloudType> cloud_plane_list_;
  CloudType cloud_plane_;
  CloudType cloud_nonplane_;
};
}  // namespace cloud_processing
