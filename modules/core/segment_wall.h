#pragma once
#include <pcl/common/centroid.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <iostream>

#include "segment.h"

#define NUM_HEURISTIC_MAX_PTS_IN_PATCH 50000

namespace cloud_processing {

using Ring = std::vector<CloudType>;
using Zone = std::vector<Ring>;

class SegmentWallConfig {
public:
  SegmentWallConfig(){};
  SegmentWallConfig(const std::string &config_path);
public:
  int num_iter = 3;
  int num_zones = 4;
  int num_min_pts = 10;
  int num_lpr = 20;
  double redundance_seeds = 0.15;
  double uprightness_thr = 0.707;
  double global_elevation_thr = -0.5;
  double adaptive_seed_selection_margin = -1.1;
  double thr_distance = 0.125;
  double noise_bound = 0.2;
  std::vector<int> num_sectors_each_zone{16, 32, 54, 32};
  std::vector<int> num_rings_each_zone{2, 4, 4, 4};
  std::vector<double> range_divide_level{2.7, 12.3625, 22.025, 41.35, 80.0};
  std::vector<double> flatness_thr{0.0, 0.000125, 0.000185, 0.000185};
};

struct PCAResult {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3f singular_values = Eigen::Vector3f::Zero();
  Eigen::Vector4f pc_mean = Eigen::Vector4f::Zero();
  Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
  Eigen::MatrixXf normal;
  double plane_d = 0.0;
};

class SegmentWall : public CloudSegmentBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SegmentWall(SegmentWallConfig& config);
  ~SegmentWall();

  bool Segment(CloudTypePtr& input_cloud_ptr, CloudTypePtr& segmented_cloud_ptr,
               CloudTypePtr& complementary_cloud_ptr) override;

 private:
  void DivideModel(CloudTypePtr& input_cloud_ptr,
                   std::vector<Zone>& cloud_model);

  void ExtractPiecewiseWall(const int zone_idx, const CloudType& all_pc,
                            CloudType& wall_pc, CloudType& non_wall_pc,
                            bool is_h_available = true);
  void EstimatePlane(const CloudType& wall_pc);

  bool ExtractInitialSeeds(const int zone_idx, const CloudType& src_pc,
                           CloudType& init_seeds, bool is_h_available = true);

 private:
  CloudType cloud_wall_;
  CloudType cloud_nonwall_;

  SegmentWallConfig config_;
  PCAResult pca_;

  bool initialized_ = false;
  double sensor_height_ = 1.75;
  float thr_plane_d_;
  std::vector<Zone> cloud_model_;
  std::vector<double> ring_sizes_{};
  std::vector<double> sector_sizes_{};
};
}  // namespace cloud_processing
