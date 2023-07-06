#pragma once
#define PCL_NO_PRECOMPILE

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl_extension {

struct PointXYZIRT {
  PCL_ADD_POINT4D;
  uint8_t intensity;
  uint16_t ring = 0;
  double timestamp = 0;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

}

POINT_CLOUD_REGISTER_POINT_STRUCT(
    pcl_extension::PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity, intensity)(
        uint16_t, ring, ring)(double, timestamp, timestamp))

using PointType = pcl::PointXYZI;
using CloudType = pcl::PointCloud<PointType>;
using CloudTypePtr = CloudType::Ptr;

using LoaderPointType = pcl_extension::PointXYZIRT;
using LoaderCloudType = pcl::PointCloud<LoaderPointType>;
using LoaderCloudTypePtr = LoaderCloudType::Ptr;
