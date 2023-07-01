#pragma once
#define PCL_NO_PRECOMPILE

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace cloud_processing {

struct PointXYZIRT {
  PCL_ADD_POINT4D;
  uint8_t intensity;
  uint16_t ring = 0;
  double timestamp = 0;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

}

POINT_CLOUD_REGISTER_POINT_STRUCT(
    cloud_processing::PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity, intensity)(
        uint16_t, ring, ring)(double, timestamp, timestamp))

using PointType = pcl::PointXYZI;
using CloudType = pcl::PointCloud<PointType>;
using CloudTypePtr = CloudType::Ptr;

using LoaderPointType = cloud_processing::PointXYZIRT;
using LoaderCloudType = pcl::PointCloud<LoaderPointType>;
using LoaderCloudTypePtr = LoaderCloudType::Ptr;
