#include "modules/pcl_extension/segment/segment_plane.h"
#include "modules/pcl_extension/segment/segment_ground.h"

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

#include "common/file_manager.h"
#include "common/logger.hpp"

#ifndef LOG_CORE_DUMP_CAPTURE
#define BACKWARD_HAS_DW 1
#endif

using namespace std;
using namespace pcl_extension;

int main(int argc, char **argv) {
  std::string current_dir = common::FileManager::GetCurrentDir();
  GINFO << "Current Dir: " << current_dir;

  pcl::PointCloud<pcl::PointXYZINormal> cloud_input;
  CloudTypePtr cloud_convert_ptr(new CloudType);
  if (pcl::io::loadPCDFile<pcl::PointXYZINormal>(
          current_dir + "/../examples/testdata/0.pcd", cloud_input) == -1) {
    PCL_ERROR("Couldn't read file example.pcd\n");
    return (-1);
  }
  for (auto &pt : cloud_input) {
    PointType pt_;
    pt_.x = pt.x;
    pt_.y = pt.y;
    pt_.z = pt.z;
    pt_.intensity = pt.intensity;
    cloud_convert_ptr->points.emplace_back(pt_);
  }

  // SegmentGroundConfig config;
  SegmentGroundConfig config(current_dir + "/../examples/testyaml/config.yaml");
  SegmentGround seg(config);
  CloudTypePtr cloud_ground_ptr(new CloudType);
  CloudTypePtr cloud_complementary_ptr(new CloudType);
  seg.Segment(cloud_convert_ptr, cloud_ground_ptr, cloud_complementary_ptr);

  pcl::io::savePCDFileASCII<PointType>(current_dir + "/../examples/testdata/1.pcd",
                                       *cloud_ground_ptr);
  pcl::io::savePCDFileASCII<PointType>(current_dir + "/../examples/testdata/2.pcd",
                                       *cloud_complementary_ptr);
  return 0;
}
