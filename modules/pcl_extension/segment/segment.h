#pragma once
#include "common/logger.hpp"
#include "pattern.hpp"

namespace pcl_extension {

class CloudSegmentBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  virtual ~CloudSegmentBase() = default;

  virtual bool Segment(CloudTypePtr& input_cloud_ptr,
                       CloudTypePtr& segmented_cloud_ptr,
                       CloudTypePtr& complementary_cloud_ptr) = 0;
};

}  // namespace pcl_extension
