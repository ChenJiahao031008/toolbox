#include "segment_ground.h"

namespace cloud_processing {

SegmentGroundConfig::SegmentGroundConfig(const std::string& config_path) {
  YAML::Node config = YAML::LoadFile(config_path);
  num_iter = config["num_iter"].as<int>();
  num_zones = config["num_zones"].as<int>();
  num_min_pts = config["num_min_pts"].as<int>();
  num_lpr = config["num_lpr"].as<int>();
  num_sectors_for_ATAT = config["num_sectors_for_ATAT"].as<int>();
  max_r_for_ATAT = config["max_r_for_ATAT"].as<double>();
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
  elevation_thresholds =
      config["elevation_thresholds"].as<std::vector<double>>();
}

SegmentGround::SegmentGround(SegmentGroundConfig& config) : config_(config) {
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
  cloud_ground_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);
  cloud_nonground_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);

  GINFO << "------------Parameter Loading Begin.-----------";
  GINFO << "config.num_iter : " << config_.num_iter;
  GINFO << "config.num_zones : " << config_.num_zones;
  GINFO << "config.num_min_pts : " << config_.num_min_pts;
  GINFO << "config.num_lpr : " << config_.num_lpr;
  GINFO << "config.num_sectors_for_ATAT : " << config_.num_sectors_for_ATAT;
  GINFO << "config.max_r_for_ATAT : " << config_.max_r_for_ATAT;
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
  GINFO << "config.elevation_thresholds : " << config_.elevation_thresholds[0]
        << " " << config_.elevation_thresholds[1] << " "
        << config_.elevation_thresholds[2] << " "
        << config_.elevation_thresholds[3];
  GINFO << "config.flatness_thr : " << config_.flatness_thr[0] << " "
        << config_.flatness_thr[1] << " " << config_.flatness_thr[2] << " "
        << config_.flatness_thr[3];
  GINFO << "------------Parameter Loading Done.------------";
}

SegmentGround::~SegmentGround() {}

bool SegmentGround::Segment(CloudTypePtr& input_cloud_ptr,
                            CloudTypePtr& segmented_cloud_ptr,
                            CloudTypePtr& complementary_cloud_ptr) {
  GINFO << "-------------SegmentGround Begin.-------------";

  // 全地形自动高度T估计器
  if (!initialized_) {
    EstimateSensorHeight(*input_cloud_ptr);
    initialized_ = true;
  }
  GINFO << "Sensor Height: " << sensor_height_;

  int i = 0;
  while (i < input_cloud_ptr->points.size()) {
    if (input_cloud_ptr->points[i].z < -sensor_height_ - 2.0) {
      std::iter_swap(input_cloud_ptr->points.begin() + i,
                     input_cloud_ptr->points.end() - 1);
      input_cloud_ptr->points.pop_back();
    } else {
      ++i;
    }
  }

  // 清空所有point
  for (int k = 0; k < config_.num_zones; ++k) {
    for (int i = 0; i < config_.num_sectors_each_zone[k]; i++) {
      for (int j = 0; j < config_.num_rings_each_zone[k]; j++) {
        if (!cloud_model_[k][j][i].points.empty())
          cloud_model_[k][j][i].points.clear();
      }
    }
  }
  cloud_ground_.clear();
  cloud_nonground_.clear();

  DivideModel(input_cloud_ptr, cloud_model_);

  int concentric_idx = 0;  // 第几环的索引（从整体来看而非从zone开始）
  for (int k = 0; k < config_.num_zones; ++k) {
    auto zone = cloud_model_[k];
    for (uint16_t ring_idx = 0; ring_idx < config_.num_rings_each_zone[k];
         ++ring_idx) {
      for (uint16_t sector_idx = 0;
           sector_idx < config_.num_sectors_each_zone[k]; ++sector_idx) {
        // GINFO << zone[ring_idx][sector_idx].points.size();
        if (zone[ring_idx][sector_idx].points.size() > config_.num_min_pts) {
          // Region-wise sorting is adopted（根据z轴排序）
          std::sort(
              zone[ring_idx][sector_idx].points.begin(),
              zone[ring_idx][sector_idx].end(),
              [&](PointType a, PointType b) -> bool { return a.z < b.z; });

          CloudType regionwise_ground_;
          CloudType regionwise_nonground_;
          ExtractPiecewiseGround(k, zone[ring_idx][sector_idx],
                                 regionwise_ground_, regionwise_nonground_);

          // Status of each patch
          // used in checking uprightness, elevation, and flatness, respectively
          const double ground_z_vec = std::abs(pca_.normal(2, 0));
          const double ground_z_elevation = pca_.pc_mean(2, 0);
          // surface_variable 对应 论文中的\sigma_{n}
          // 严格来看是一种曲率变化，详见https://blog.csdn.net/qq_34213260/article/details/107099993
          const double surface_variable =
              pca_.singular_values.minCoeff() /
              (pca_.singular_values(0) + pca_.singular_values(1) +
               pca_.singular_values(2));

          // 垂直度判断
          if (ground_z_vec < config_.uprightness_thr) {
            // All points are rejected
            cloud_nonground_ += regionwise_ground_;
            cloud_nonground_ += regionwise_nonground_;
          } else {  // satisfy uprightness
            // 只有 concentric_idx < config.elevation_thresholds.size()
            // 才计算曲率？ 不明白为什么分开算？
            if (concentric_idx < config_.elevation_thresholds.size()) {
              // 如果没有这个elevation_thresholds.size()限制，否则明显数组越界了，
              // 此时k=0（因为concentric_idx是整体而言的）
              // z的高度阈值（但是不明白为什么取那么高）
              // 默认elevation_thresholds:  [0.523, 0.746, 0.879, 1.125]
              if (ground_z_elevation >
                  -sensor_height_ +
                      config_.elevation_thresholds[ring_idx + 2 * k]) {
                // 平坦度限制，小于阈值则认为才是平面
                // a/(a+b) -> 1-b/(a+b)
                // a->up, 1-b/(a+b)->up
                if (config_.flatness_thr[ring_idx + 2 * k] > surface_variable) {
                  cloud_ground_ += regionwise_ground_;
                  cloud_nonground_ += regionwise_nonground_;
                } else {
                  cloud_nonground_ += regionwise_ground_;
                  cloud_nonground_ += regionwise_nonground_;
                }
              } else {
                cloud_ground_ += regionwise_ground_;
                cloud_nonground_ += regionwise_nonground_;
              }
            } else {
              if ((ground_z_elevation > config_.global_elevation_thr)) {
                cloud_nonground_ += regionwise_ground_;
                cloud_nonground_ += regionwise_nonground_;
              } else {
                cloud_ground_ += regionwise_ground_;
                cloud_nonground_ += regionwise_nonground_;
              }
            }
          }
        }
      }
      ++concentric_idx;
    }
  }

  *segmented_cloud_ptr = cloud_ground_;
  segmented_cloud_ptr->width = cloud_ground_.size();
  segmented_cloud_ptr->height = 1;
  segmented_cloud_ptr->is_dense = true;
  segmented_cloud_ptr->points.resize(segmented_cloud_ptr->width *
                                     segmented_cloud_ptr->height);

  *complementary_cloud_ptr = cloud_nonground_;
  complementary_cloud_ptr->width = cloud_nonground_.size();
  complementary_cloud_ptr->height = 1;
  complementary_cloud_ptr->is_dense = true;
  complementary_cloud_ptr->points.resize(complementary_cloud_ptr->width *
                                         complementary_cloud_ptr->height);

  GINFO << "-------------SegmentGround Done.-------------";
  return true;
}

void SegmentGround::DivideModel(CloudTypePtr& input_cloud_ptr,
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

void SegmentGround::ExtractPiecewiseGround(const int zone_idx,
                                           const CloudType& all_pc,
                                           CloudType& ground_pc,
                                           CloudType& non_ground_pc,
                                           bool is_h_available) {
  // 0. Initialization
  if (!ground_pc.empty()) ground_pc.clear();
  if (!non_ground_pc.empty()) non_ground_pc.clear();
  // CloudType tmp_ground_pc;

  // 1. set seeds!
  // 当is_h_available为false时，extract_piecewiseground提取默认的前20个点作为初始种子点
  // 小于种子点+冗余量的阈值时认为是地面点（ground_pc_）。这一点感觉很疑惑，为什么是前20个？
  ExtractInitialSeeds(zone_idx, all_pc, ground_pc, is_h_available);

  // 2. Extract ground_pc
  // config_.num_iter: 迭代次数默认为3
  for (int i = 0; i < config_.num_iter; i++) {
    // 计算平面法向量
    EstimatePlane(ground_pc);
    // 清空地面点，为新一轮迭代做准备
    ground_pc.clear();

    // pointcloud to matrix
    Eigen::MatrixXf points(all_pc.points.size(), 3);
    int j = 0;
    for (auto& p : all_pc.points) {
      points.row(j++) << p.x, p.y, p.z;
    }
    // tmp_ground_pc plane model
    Eigen::VectorXf result = points * pca_.normal;
    // threshold filter
    for (int r = 0; r < result.rows(); r++) {
      if (i < config_.num_iter - 1) {
        if (result[r] < thr_plane_d_) {
          // tmp_ground_pc.points.push_back(all_pc[r]);
        }
      } else {  // Final stage
        if (result[r] < thr_plane_d_) {
          ground_pc.points.push_back(all_pc[r]);
        } else {
          if (i == config_.num_iter - 1) {
            non_ground_pc.push_back(all_pc[r]);
          }
        }
      }
    }
  }
  // tmp_ground_pc.clear();
}

void SegmentGround::EstimatePlane(const CloudType& ground_pc) {
  pcl::computeMeanAndCovarianceMatrix(ground_pc, pca_.covariance, pca_.pc_mean);
  // Singular Value Decomposition: SVD
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(
      pca_.covariance, Eigen::DecompositionOptions::ComputeFullU);
  pca_.singular_values = svd.singularValues();

  // use the least singular vector as normal
  pca_.normal = (svd.matrixU().col(2));
  // mean ground_pc seeds value
  Eigen::Vector3f seeds_mean = pca_.pc_mean.head<3>();

  // according to normal.T*[x,y,z] = -d
  pca_.plane_d = -(pca_.normal.transpose() * seeds_mean)(0, 0);
  // set distance threhold to `th_dist - d`
  thr_plane_d_ = config_.thr_distance - pca_.plane_d;
}

void SegmentGround::ExtractInitialSeeds(const int zone_idx,
                                        const CloudType& src_pc,
                                        CloudType& init_seeds,
                                        bool is_h_available) {
  init_seeds.points.clear();

  // LPR is the mean of low point representative
  double sum = 0;
  int cnt = 0;

  int init_idx = 0;
  // Empirically, adaptive seed selection applying to Z1 is fine
  if (is_h_available) {
    static double lowest_h_margin_in_close_zone =
        (sensor_height_ == 0.0)
            ? -0.1
            : config_.adaptive_seed_selection_margin * sensor_height_;
    if (zone_idx == 0) {
      for (int i = 0; i < src_pc.points.size(); i++) {
        if (src_pc.points[i].z < lowest_h_margin_in_close_zone) {
          ++init_idx;
        } else {
          break;
        }
      }
    }
  }

  // Calculate the mean height value.
  // num_lpr: 20 最多20个点
  for (int i = init_idx; i < src_pc.points.size() && cnt < config_.num_lpr;
       i++) {
    sum += src_pc.points[i].z;
    cnt++;
  }
  // 均值
  double mean_z_height = cnt != 0 ? sum / cnt : 0;  // in case divide by 0

  // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
  // th_seeds = 0.5 冗余量
  for (int i = 0; i < src_pc.points.size(); i++) {
    if (src_pc.points[i].z < mean_z_height + config_.redundance_seeds) {
      init_seeds.points.push_back(src_pc.points[i]);
    }
  }
}

void SegmentGround::EstimateSensorHeight(CloudType& cloud_in) {
  // ATAT: All-Terrain Automatic HeighT estimator
  // default config_.num_sectors_for_ATAT = 20
  Ring ring_for_ATAT;
  ring_for_ATAT.resize(config_.num_sectors_for_ATAT);
  for (auto const& pt : cloud_in.points) {
    int ring_idx, sector_idx;
    double r = std::sqrt(std::pow(pt.x, 2) + std::pow(pt.y, 2));
    float sector_size_for_ATAT = 2 * M_PI / config_.num_sectors_for_ATAT;

    // max_r_for_ATAT = 5.0
    if ((r <= config_.max_r_for_ATAT) && (r > config_.range_divide_level[0])) {
      double atan_value = std::atan2(pt.y, pt.x);
      double theta = atan_value > 0 ? atan_value : atan_value + 2 * M_PI;

      sector_idx = std::min(static_cast<int>((theta / sector_size_for_ATAT)),
                            config_.num_sectors_for_ATAT);
      ring_for_ATAT[sector_idx].points.emplace_back(pt);
    }
  }

  // Assign valid measurements and corresponding linearities/planarities
  // 线性度 点云更趋向与直线，越小越好
  // 平面度 点云更趋向与平面，越大越好
  std::vector<double> ground_elevations_wrt_the_origin;
  std::vector<double> linearities;
  std::vector<double> planarities;
  for (int i = 0; i < config_.num_sectors_for_ATAT; ++i) {
    if (ring_for_ATAT[i].size() < config_.num_min_pts) {
      continue;
    }

    CloudType dummy_est_ground;
    CloudType dummy_est_non_ground;
    // false: is_h_available 选择第0个sector
    ExtractPiecewiseGround(0, ring_for_ATAT[i], dummy_est_ground,
                           dummy_est_non_ground, false);

    const double ground_z_vec = std::abs(pca_.normal(2, 0));
    const double ground_z_elevation = pca_.pc_mean(2, 0);
    const double linearity =
        (pca_.singular_values(0) - pca_.singular_values(1)) /
        pca_.singular_values(0);
    const double planarity =
        (pca_.singular_values(1) - pca_.singular_values(2)) /
        pca_.singular_values(0);

    // Check whether the vector is sufficiently upright and flat
    // config_.uprightness_thr 垂直度阈值
    if (ground_z_vec > config_.uprightness_thr && linearity < 0.9) {
      ground_elevations_wrt_the_origin.push_back(ground_z_elevation);
      linearities.push_back(linearity);
      planarities.push_back(planarity);
    }
  }

  // Setting for consensus set-based height estimation
  int N = ground_elevations_wrt_the_origin.size();
  Eigen::Matrix<double, 1, Eigen::Dynamic> values = Eigen::MatrixXd::Ones(1, N);
  Eigen::Matrix<double, 1, Eigen::Dynamic> ranges =
      config_.noise_bound * Eigen::MatrixXd::Ones(1, N);
  Eigen::Matrix<double, 1, Eigen::Dynamic> weights =
      1.0 / (config_.noise_bound * config_.noise_bound) *
      Eigen::MatrixXd::Ones(1, N);
  for (int i = 0; i < N; ++i) {
    values(0, i) = ground_elevations_wrt_the_origin[i];
    ranges(0, i) = ranges(0, i) * linearities[i];
    weights(0, i) = weights(0, i) * planarities[i] * planarities[i];
  }

  double estimated_h =
      ConsensusSetBasedHeightEstimation(values, ranges, weights);

  // Note that these are opposites
  sensor_height_ = -estimated_h;
}

double SegmentGround::ConsensusSetBasedHeightEstimation(
    const Eigen::RowVectorXd& X, const Eigen::RowVectorXd& ranges,
    const Eigen::RowVectorXd& weights) {
  // check input parameters
  bool dimension_inconsistent =
      (X.rows() != ranges.rows()) || (X.cols() != ranges.cols());

  bool only_one_element = (X.rows() == 1) && (X.cols() == 1);
  assert(!dimension_inconsistent);
  assert(!only_one_element);  // TODO: admit a trivial solution

  // values(0, i) = ground_elevations_wrt_the_origin[i]; 对应 X
  // ranges(0, i) = ranges(0, i) * linearities[i];
  // weights(0, i) = weights(0, i) * planarities[i] * planarities[i];

  int N = X.cols();
  std::vector<std::pair<double, int>> h;
  for (size_t i = 0; i < N; ++i) {
    h.push_back(std::make_pair(X(i) - ranges(i), i + 1));
    h.push_back(std::make_pair(X(i) + ranges(i), -i - 1));
  }

  // ascending order
  std::sort(h.begin(), h.end(),
            [](std::pair<double, int> a, std::pair<double, int> b) {
              return a.first < b.first;
            });

  int nr_centers = 2 * N;
  Eigen::RowVectorXd x_hat = Eigen::MatrixXd::Zero(1, nr_centers);
  Eigen::RowVectorXd x_cost = Eigen::MatrixXd::Zero(1, nr_centers);

  double ranges_inverse_sum = ranges.sum();
  double dot_X_weights = 0;
  double dot_weights_consensus = 0;
  int consensus_set_cardinal = 0;
  double sum_xi = 0;
  double sum_xi_square = 0;

  for (size_t i = 0; i < nr_centers; ++i) {
    int idx = int(std::abs(h.at(i).second)) - 1;  // Indices starting at 1
    int epsilon = (h.at(i).second > 0) ? 1 : -1;

    consensus_set_cardinal += epsilon;
    dot_weights_consensus += epsilon * weights(idx);
    dot_X_weights += epsilon * weights(idx) * X(idx);
    ranges_inverse_sum -= epsilon * ranges(idx);
    sum_xi += epsilon * X(idx);
    sum_xi_square += epsilon * X(idx) * X(idx);

    x_hat(i) = dot_X_weights / dot_weights_consensus;

    double residual = consensus_set_cardinal * x_hat(i) * x_hat(i) +
                      sum_xi_square - 2 * sum_xi * x_hat(i);
    x_cost(i) = residual + ranges_inverse_sum;
  }

  size_t min_idx;
  x_cost.minCoeff(&min_idx);
  double estimate_temp = x_hat(min_idx);
  return estimate_temp;
}

}  // namespace cloud_processing
