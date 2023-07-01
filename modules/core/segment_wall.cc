#include "segment_wall.h"

namespace cloud_processing {

SegmentWallConfig::SegmentWallConfig(const std::string& config_path) {
  YAML::Node config = YAML::LoadFile(config_path);
  num_iter = config["num_iter"].as<int>();
  num_zones = config["num_zones"].as<int>();
  num_min_pts = config["num_min_pts"].as<int>();
  num_lpr = config["num_lpr"].as<int>();
  redundance_seeds = config["redundance_seeds"].as<double>();
  uprightness_thr = config["uprightness_thr"].as<double>();
  thr_distance = config["thr_distance"].as<double>();
  noise_bound = config["noise_bound"].as<double>();
  global_elevation_thr = config["global_elevation_thr"].as<double>();
  adaptive_seed_selection_margin =
      config["adaptive_seed_selection_margin"].as<double>();
  num_sectors_each_zone =
      config["num_sectors_each_zone"].as<std::vector<int>>();
  num_rings_each_zone = config["num_rings_each_zone"].as<std::vector<int>>();
  range_divide_level = config["range_divide_level"].as<std::vector<double>>();
  flatness_thr = config["flatness_thr"].as<std::vector<double>>();
}

SegmentWall::SegmentWall(SegmentWallConfig& config) : config_(config) {
  ring_sizes_ = {
      (config_.range_divide_level[1] - config_.range_divide_level[0]) /
          config_.num_rings_each_zone[0],
      (config_.range_divide_level[2] - config_.range_divide_level[1]) /
          config_.num_rings_each_zone[1],
      (config_.range_divide_level[3] - config_.range_divide_level[2]) /
          config_.num_rings_each_zone[2],
      (config_.range_divide_level[4] - config_.range_divide_level[3]) /
          config_.num_rings_each_zone[3]};

  sector_sizes_ = {2 * M_PI / config_.num_sectors_each_zone.at(0),
                   2 * M_PI / config_.num_sectors_each_zone.at(1),
                   2 * M_PI / config_.num_sectors_each_zone.at(2),
                   2 * M_PI / config_.num_sectors_each_zone.at(3)};
  // zones: 环形大区;
  // sector: 每个zone中环向分割格子;
  // ring: 每个zone中轴向分割线束;
  for (int i = 0; i < config_.num_zones; ++i) {
    Zone z;
    CloudType cloud;
    Ring ring;
    // zones: 环形大区; sector: 每个zone中环向分割格子; ring:
    // 每个zone中轴向分割线束;
    for (int j = 0; j < config_.num_sectors_each_zone[i]; j++) {
      ring.emplace_back(cloud);
    }
    for (int k = 0; k < config_.num_rings_each_zone[i]; k++) {
      z.emplace_back(ring);
    }
    cloud_model_.push_back(z);
  }
  cloud_wall_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);
  cloud_nonwall_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);

  GINFO << "------------Parameter Loading Begin.-----------";
  GINFO << "config.num_iter : " << config_.num_iter;
  GINFO << "config.num_zones : " << config_.num_zones;
  GINFO << "config.num_min_pts : " << config_.num_min_pts;
  GINFO << "config.num_lpr : " << config_.num_lpr;
  GINFO << "config.redundance_seeds : " << config_.redundance_seeds;
  GINFO << "config.uprightness_thr : " << config_.uprightness_thr;
  GINFO << "config.global_elevation_thr : " << config_.global_elevation_thr;
  GINFO << "config.thr_distance : " << config_.thr_distance;
  GINFO << "config.noise_bound : " << config_.noise_bound;
  GINFO << "config.adaptive_seed_selection_margin : "
        << config_.adaptive_seed_selection_margin;
  GINFO << "config.num_sectors_each_zone : " << config_.num_sectors_each_zone[0]
        << " " << config_.num_sectors_each_zone[1] << " "
        << config_.num_sectors_each_zone[2] << " "
        << config_.num_sectors_each_zone[3];
  GINFO << "config.num_rings_each_zone : " << config_.num_rings_each_zone[0]
        << " " << config_.num_rings_each_zone[1] << " "
        << config_.num_rings_each_zone[2] << " "
        << config_.num_rings_each_zone[3];
  GINFO << "config.range_divide_level : " << config_.range_divide_level[0]
        << " " << config_.range_divide_level[1] << " "
        << config_.range_divide_level[2] << " " << config_.range_divide_level[3]
        << " " << config_.range_divide_level[4];
  GINFO << "config.flatness_thr : " << config_.flatness_thr[0] << " "
        << config_.flatness_thr[1] << " " << config_.flatness_thr[2] << " "
        << config_.flatness_thr[3];
  GINFO << "------------Parameter Loading Done.------------";
}

SegmentWall::~SegmentWall() {}

bool SegmentWall::Segment(CloudTypePtr& input_cloud_ptr,
                          CloudTypePtr& segmented_cloud_ptr,
                          CloudTypePtr& complementary_cloud_ptr) {
  GINFO << "-------------SegmentWall Begin.-------------";

  DivideModel(input_cloud_ptr, cloud_model_);

  for (int k = 0; k < config_.num_zones; ++k) {
    auto zone = cloud_model_[k];
    for (uint16_t ring_idx = 0; ring_idx < config_.num_rings_each_zone[k];
         ++ring_idx) {
      for (uint16_t sector_idx = 0;
           sector_idx < config_.num_sectors_each_zone[k]; ++sector_idx) {
        // GINFO << zone[ring_idx][sector_idx].points.size();
        if (zone[ring_idx][sector_idx].points.size() > config_.num_min_pts) {
          // Region-wise sorting is adopted（根据z轴排序:提取墙面时候从高到低）
          std::sort(
              zone[ring_idx][sector_idx].points.begin(),
              zone[ring_idx][sector_idx].end(),
              [&](PointType a, PointType b) -> bool { return a.z > b.z; });

          CloudType regionwise_wall_;
          CloudType regionwise_nonwall_;
          ExtractPiecewiseWall(k, zone[ring_idx][sector_idx], regionwise_wall_,
                               regionwise_nonwall_);

          const double surface_variable =
              pca_.singular_values.minCoeff() /
              (pca_.singular_values(0) + pca_.singular_values(1) +
               pca_.singular_values(2));

          if (config_.flatness_thr[ring_idx + 2 * k] > surface_variable) {
            cloud_wall_ += regionwise_wall_;
            cloud_nonwall_ += regionwise_nonwall_;
          }else{
            cloud_nonwall_ += regionwise_wall_;
            cloud_nonwall_ += regionwise_nonwall_;
          }
        }
      }
    }
  }

  *segmented_cloud_ptr = cloud_wall_;
  segmented_cloud_ptr->width = cloud_wall_.size();
  segmented_cloud_ptr->height = 1;
  segmented_cloud_ptr->is_dense = true;
  segmented_cloud_ptr->points.resize(segmented_cloud_ptr->width *
                                     segmented_cloud_ptr->height);

  *complementary_cloud_ptr = cloud_nonwall_;
  complementary_cloud_ptr->width = cloud_nonwall_.size();
  complementary_cloud_ptr->height = 1;
  complementary_cloud_ptr->is_dense = true;
  complementary_cloud_ptr->points.resize(complementary_cloud_ptr->width *
                                         complementary_cloud_ptr->height);

  GINFO << "-------------SegmentWall Done.-------------";
  return true;
}

void SegmentWall::DivideModel(CloudTypePtr& input_cloud_ptr,
                              std::vector<Zone>& cloud_model) {
  for (auto const& pt : input_cloud_ptr->points) {
    double r = std::sqrt(std::pow(pt.x, 2) + std::pow(pt.y, 2));
    double atan_value = std::atan2(pt.x, pt.y);
    double theta = atan_value > 0 ? atan_value : atan_value + 2 * M_PI;
    if ((r <= config_.range_divide_level[4]) &&
        (r > config_.range_divide_level[0])) {
      int ring_idx, sector_idx;
      if (r < config_.range_divide_level[1]) {  // In First rings
        ring_idx =
            std::min(static_cast<int>(((r - config_.range_divide_level[0]) /
                                       ring_sizes_[0])),
                     config_.num_rings_each_zone[0] - 1);
        sector_idx = std::min(static_cast<int>((theta / sector_sizes_[0])),
                              config_.num_sectors_each_zone[0] - 1);
        cloud_model[0][ring_idx][sector_idx].points.emplace_back(pt);
      } else if (r < config_.range_divide_level[2]) {
        ring_idx =
            std::min(static_cast<int>(((r - config_.range_divide_level[1]) /
                                       ring_sizes_[1])),
                     config_.num_rings_each_zone[1] - 1);
        sector_idx = std::min(static_cast<int>((theta / sector_sizes_[1])),
                              config_.num_sectors_each_zone[1] - 1);
        cloud_model[1][ring_idx][sector_idx].points.emplace_back(pt);
      } else if (r < config_.range_divide_level[3]) {
        ring_idx =
            std::min(static_cast<int>(((r - config_.range_divide_level[2]) /
                                       ring_sizes_[2])),
                     config_.num_rings_each_zone[2] - 1);
        sector_idx = std::min(static_cast<int>((theta / sector_sizes_[2])),
                              config_.num_sectors_each_zone[2] - 1);
        cloud_model[2][ring_idx][sector_idx].points.emplace_back(pt);
      } else {  // Far!
        ring_idx =
            std::min(static_cast<int>(((r - config_.range_divide_level[3]) /
                                       ring_sizes_[3])),
                     config_.num_rings_each_zone[3] - 1);
        sector_idx = std::min(static_cast<int>((theta / sector_sizes_[3])),
                              config_.num_sectors_each_zone[3] - 1);

        cloud_model[3][ring_idx][sector_idx].points.emplace_back(pt);
      }
    }
  }
}

void SegmentWall::ExtractPiecewiseWall(const int zone_idx,
                                       const CloudType& all_pc,
                                       CloudType& wall_pc,
                                       CloudType& non_wall_pc,
                                       bool is_h_available) {
  // 0. Initialization
  if (!wall_pc.empty()) wall_pc.clear();
  if (!non_wall_pc.empty()) non_wall_pc.clear();

  if (!ExtractInitialSeeds(zone_idx, all_pc, wall_pc, is_h_available)) {
    non_wall_pc = all_pc;
    return;
  }

  for (int i = 0; i < config_.num_iter; i++) {
    EstimatePlane(wall_pc);
    wall_pc.clear();

    Eigen::MatrixXf points(all_pc.points.size(), 3);
    int j = 0;
    for (auto& p : all_pc.points) {
      points.row(j++) << p.x, p.y, p.z;
    }
    Eigen::VectorXf result = points * pca_.normal;
    for (int r = 0; r < result.rows(); r++) {
      if (i < config_.num_iter - 1) {
        if (result[r] < thr_plane_d_) {
          wall_pc.points.push_back(all_pc[r]);
        }
      } else {  // Final stage
        if (result[r] < thr_plane_d_) {
          wall_pc.points.push_back(all_pc[r]);
        } else {
          if (i == config_.num_iter - 1) {
            non_wall_pc.push_back(all_pc[r]);
          }
        }
      }
    }
  }
  // tmp_wall_pc.clear();
}

void SegmentWall::EstimatePlane(const CloudType& wall_pc) {
  pcl::computeMeanAndCovarianceMatrix(wall_pc, pca_.covariance, pca_.pc_mean);
  // Singular Value Decomposition: SVD
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(
      pca_.covariance, Eigen::DecompositionOptions::ComputeFullU);
  pca_.singular_values = svd.singularValues();

  // use the least singular vector as normal
  pca_.normal = (svd.matrixU().col(2));
  // mean wall_pc seeds value
  Eigen::Vector3f seeds_mean = pca_.pc_mean.head<3>();
  // according to normal.T*[x,y,z] = -d
  pca_.plane_d = -(pca_.normal.transpose() * seeds_mean)(0, 0);
  // set distance threhold to `th_dist - d`
  thr_plane_d_ = config_.thr_distance - pca_.plane_d;
}

bool SegmentWall::ExtractInitialSeeds(const int zone_idx,
                                      const CloudType& src_pc,
                                      CloudType& init_seeds,
                                      bool is_h_available) {
  init_seeds.points.clear();

  // LPR is the mean of low point representative
  double sum = 0;

  // Calculate the mean height value.
  int cnt = 0;
  for (int i = 0; i < src_pc.points.size() && cnt < config_.num_lpr;
       i++) {
    sum += std::sqrt(std::pow(src_pc.points[i].x, 2) +
                     std::pow(src_pc.points[i].y, 2));
    cnt++;
  }
  // 均值
  double mean_r = cnt != 0 ? sum / cnt : 0;  // in case divide by 0

  // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
  // th_seeds = 0.5 冗余量
  for (int i = 0; i < src_pc.points.size(); i++) {
    double r = std::sqrt(std::pow(src_pc.points[i].x, 2) +
                         std::pow(src_pc.points[i].y, 2));
    if (std::fabs(r - mean_r) < config_.redundance_seeds) {
      init_seeds.points.push_back(src_pc.points[i]);
    }
  }
  if (init_seeds.size() * 1.0 / src_pc.size() < 0.25) {
    return false;
  }
  return true;
}

}  // namespace cloud_processing
