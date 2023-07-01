#pragma once
#include <boost/algorithm/string.hpp>
#include <boost/dll/runtime_symbol_info.hpp>
#include <fstream>
#include <iostream>

#include "logger.hpp"
namespace common {

enum class OutputMode {
  NoRecursionAll = 0,
  RecursionFile = 1,
  RecursionFolder = 2,
  RecursionAll = 3
};

class FileManager {
 public:
  FileManager() = default;
  ~FileManager() = default;

  // 读取文件夹中的所有文件夹和文件名
  // mode=0:不允许递归, 输出文件和文件夹; mode=1:允许递归, 但是只输出文件名;
  // mode=2:允许递归, 但是只输出文件夹; mode=3: 允许递归, 输出文件名和文件路径;
  static void ReadFolder(const std::string &folder,
                         std::vector<std::string> &files,
                         common::OutputMode mode = OutputMode::RecursionAll);

  // (推荐)拷贝文件或者文件夹: 如果源为普通文件, 目标为文件夹形式,
  // 则会自动创建同名普通文件, 如果目标为普通文件, 则会覆盖目标文件.
  static bool CopyFiles(const std::string &src, const std::string &dst);
  static bool MoveFiles(const std::string &src, const std::string &dst);

  // 拷贝文件夹, 支持递归拷贝
  static bool CopyDirectory(const std::string &strSourceDir,
                            const std::string &strDestDir);

  // 获取文件地址中的文件名
  static bool GetFileNameInPath(const std::string &path, std::string &filename);

  // 获得文件地址中的扩展名
  static std::string GetFileExtension(const std::string &path);

  // 获取终端的当前目录
  static std::string GetTerminalDir();
  static std::string GetCurrentDir();
};

}  // namespace common
