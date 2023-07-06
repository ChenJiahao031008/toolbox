#pragma once
#include <pcl/common/centroid.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <iostream>

#include "segment.h"

#define NUM_HEURISTIC_MAX_PTS_IN_PATCH 50000

namespace pcl_extension {

using Ring = std::vector<CloudType>;
using Zone = std::vector<Ring>;

class SegmentGroundConfig {
public:
  SegmentGroundConfig(){};
  SegmentGroundConfig(const std::string &config_path);
public:
  int num_iter = 3;
  int num_zones = 4;
  int num_min_pts = 10;
  int num_lpr = 20;
  int num_sectors_for_ATAT = 20;
  double max_r_for_ATAT = 5.0;
  double redundance_seeds = 0.5;
  double uprightness_thr = 0.707;
  double global_elevation_thr = -0.5;
  double adaptive_seed_selection_margin = -1.1;
  double thr_distance = 0.125;
  double noise_bound = 0.2;
  std::vector<int> num_sectors_each_zone{16, 32, 54, 32};
  std::vector<int> num_rings_each_zone{2, 4, 4, 4};
  std::vector<double> range_divide_level{2.7, 12.3625, 22.025, 41.35, 80.0};
  std::vector<double> elevation_thresholds{0.5, 0.8, 1.0, 1.1};
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

class SegmentGround : public CloudSegmentBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SegmentGround(SegmentGroundConfig& config);
  ~SegmentGround();

  bool Segment(CloudTypePtr& input_cloud_ptr, CloudTypePtr& segmented_cloud_ptr,
               CloudTypePtr& complementary_cloud_ptr) override;

 private:
  void DivideModel(CloudTypePtr& input_cloud_ptr,
                   std::vector<Zone>& cloud_model);

  void ExtractPiecewiseGround(const int zone_idx, const CloudType& all_pc,
                              CloudType& ground_pc, CloudType& non_ground_pc,
                              bool is_h_available = true);
  void EstimatePlane(const CloudType& ground_pc);
  void ExtractInitialSeeds(const int zone_idx, const CloudType& src_pc,
                           CloudType& init_seeds, bool is_h_available = true);
  void EstimateSensorHeight(CloudType& cloud_in);
  double ConsensusSetBasedHeightEstimation(const Eigen::RowVectorXd& X,
                                           const Eigen::RowVectorXd& ranges,
                                           const Eigen::RowVectorXd& weights);

 private:
  CloudType cloud_ground_;
  CloudType cloud_nonground_;

  SegmentGroundConfig config_;
  PCAResult pca_;

  bool initialized_ = false;
  double sensor_height_ = 1.75;
  float thr_plane_d_;
  std::vector<Zone> cloud_model_;
  std::vector<double> ring_sizes_{};
  std::vector<double> sector_sizes_{};
};
}  // namespace pcl_extension
