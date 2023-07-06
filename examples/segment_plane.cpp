#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

#include "common/logger.hpp"
#include "modules/pcl_extension/segment/segment_ground.h"
#include "modules/pcl_extension/segment/segment_plane.h"
#include "common/file_manager.h"

#ifndef LOG_CORE_DUMP_CAPTURE
#define BACKWARD_HAS_DW 1
#endif

using namespace std;
using namespace pcl_extension;

int main(int argc, char **argv) {
  std::string current_dir = common::FileManager::GetCurrentDir();
  GINFO << "Current Dir: " << current_dir;

  CloudTypePtr cloud_input_ptr(new CloudType);
  if (pcl::io::loadPCDFile<PointType>(
          current_dir + "/../examples/testdata/2.pcd", *cloud_input_ptr) ==
      -1) {
    PCL_ERROR("Couldn't read file example.pcd\n");
    return (-1);
  }

  CloudTypePtr cloud_plane_ptr(new CloudType);
  CloudTypePtr cloud_non_plane_ptr(new CloudType);
  SegmentPlane seg;
  seg.Segment(cloud_input_ptr, cloud_plane_ptr, cloud_non_plane_ptr);

  pcl::io::savePCDFileASCII<PointType>(
      current_dir + "/../examples/testdata/2-1.pcd", *cloud_plane_ptr);
  pcl::io::savePCDFileASCII<PointType>(
      current_dir + "/../examples/testdata/2-2.pcd", *cloud_non_plane_ptr);

  return 0;
}
