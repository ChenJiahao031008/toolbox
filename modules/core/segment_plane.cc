#include "segment_plane.h"

namespace cloud_processing {

SegmentPlane::SegmentPlane() {}

bool SegmentPlane::Segment(CloudTypePtr& input_cloud_ptr,
                           CloudTypePtr& segmented_cloud_ptr,
                           CloudTypePtr& complementary_cloud_ptr) {
  GINFO << "-------------SegmentGround Begin.-------------";
  pcl::copyPointCloud(*input_cloud_ptr, cloud_nonplane_);

  std::vector<int> mapping;
  pcl::removeNaNFromPointCloud(cloud_nonplane_, cloud_nonplane_, mapping);

  pcl::SACSegmentation<PointType> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PARALLEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(500);
  seg.setDistanceThreshold(0.05);

  int raw_num_points = cloud_nonplane_.size();

  int segment_count = 1;
  cloud_plane_.clear();
  while (segment_count < 8) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    auto cloud_nonplane_ptr = cloud_nonplane_.makeShared();
    seg.setInputCloud(cloud_nonplane_ptr);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() < 100) break;

    CloudTypePtr cloud_plane_ptr(new CloudType);
    pcl::ExtractIndices<PointType> extractor;
    extractor.setInputCloud(cloud_nonplane_ptr);
    extractor.setIndices(inliers);
    extractor.setNegative(false);  // 提取内点
    extractor.filter(*cloud_plane_ptr);
    cloud_plane_list_.emplace_back(*cloud_plane_ptr);
    cloud_plane_ += *cloud_plane_ptr;

    cloud_nonplane_.clear();
    extractor.setNegative(true);  // 提取外点
    extractor.filter(cloud_nonplane_);
    GINFO << "Current Segment Num: " << segment_count++;
  }
  *segmented_cloud_ptr = cloud_plane_;
  segmented_cloud_ptr->width = cloud_plane_.size();
  segmented_cloud_ptr->height = 1;
  segmented_cloud_ptr->is_dense = true;
  segmented_cloud_ptr->points.resize(segmented_cloud_ptr->width *
                                         segmented_cloud_ptr->height);

  *complementary_cloud_ptr = cloud_nonplane_;
  complementary_cloud_ptr->width = cloud_nonplane_.size();
  complementary_cloud_ptr->height = 1;
  complementary_cloud_ptr->is_dense = true;
  complementary_cloud_ptr->points.resize(complementary_cloud_ptr->width *
                                         complementary_cloud_ptr->height);
  GINFO << "-------------SegmentGround Done.-------------";
  return true;
}

std::vector<CloudType> SegmentPlane::GetPlaneCloudList(){
  return cloud_plane_list_;
}

}  // namespace cloud_processing
