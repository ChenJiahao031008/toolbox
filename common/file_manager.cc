#include "file_manager.h"

namespace common {

void FileManager::ReadFolder(const std::string &folder,
                             std::vector<std::string> &files,
                             common::OutputMode mode) {
  const fs::path folder_path(folder);
  if (fs::is_directory(folder_path)) {
    for (fs::directory_iterator it(folder_path); it != fs::directory_iterator();
         ++it) {
      const fs::path cur_pos = folder_path / (it->path()).filename();

      if (fs::is_directory(cur_pos)) {
        if (mode != OutputMode::RecursionFile)
          files.emplace_back(cur_pos.string());
        if (mode != OutputMode::NoRecursionAll)
          ReadFolder(cur_pos.string(), files);
      } else if (fs::is_regular_file(cur_pos)) {
        if (mode != OutputMode::RecursionFolder)
          files.emplace_back(cur_pos.string());
      }
    }
  } else {
    GERROR << folder << " is not a directory.";
  }
}

bool FileManager::CopyFiles(const std::string &src, const std::string &dst) {
#if defined __has_include
#if __has_include(<filesystem>)
  std::error_code ec;
#else
  boost::system::error_code ec;
#endif
#endif
  const fs::path dst_path(dst);
  const fs::path src_path(src);

  // 待拷贝文件为普通文件，则直接拷贝
  if (fs::is_regular_file(src_path)) {
    // dst is not exist
    if (!fs::exists(dst_path))
      fs::create_directories(dst_path.parent_path());

    // case1: src is a file, dst is also a file
    if (fs::is_regular_file(dst_path) || !fs::exists(dst_path)) {
#if defined __has_include
#if __has_include(<filesystem>)
      fs::copy_file(src_path, dst_path, fs::copy_options::overwrite_existing,
                    ec);
#else
      fs::copy_file(src_path, dst_path, fs::copy_option::overwrite_if_exists,
                    ec);
#endif
#endif
      if (ec) {
        GERROR << "Copy File Failed: " << ec.message();
        return false;
      }
    }
    // case2: src is a file, dst is a directory
    else if (fs::is_directory(dst_path)) {
      fs::path dst_file = dst_path / src_path.filename();
#if defined __has_include
#if __has_include(<filesystem>)
      fs::copy_file(src_path, dst_file, fs::copy_options::overwrite_existing,
                    ec);
#else
      fs::copy_file(src_path, dst_file, fs::copy_option::overwrite_if_exists,
                    ec);
#endif
#endif
      if (ec) {
        GERROR << "Copy File Failed: " << ec.message();
        return false;
      }
    }
    // case3: src is a file, dst is neither file nor directory
    else {
      if (ec) {
        GERROR << "Copy File Failed: dst is neither file nor directory. ec: "
               << ec.message();
        return false;
      }
    }
    return true;
  }

  // 待拷贝文件为目录，则递归拷贝
  if (fs::is_directory(src_path)) {
    if (!fs::exists(dst_path)) {
      fs::create_directories(dst_path);
    }
    for (fs::directory_iterator it(src_path); it != fs::directory_iterator();
         ++it) {
      const fs::path newSrc = src_path / (it->path()).filename();
      const fs::path newDst = dst_path / (it->path()).filename();
      // GINFO << "newSrc: " << newSrc.string() << "\n newDst: " <<
      // newDst.string();
      if (fs::is_directory(newSrc)) {
        CopyFiles(newSrc.string(), newDst.string());
      } else if (fs::is_regular_file(newSrc)) {
#if defined __has_include
#if __has_include(<filesystem>)
        fs::copy_file(newSrc, newDst, fs::copy_options::overwrite_existing, ec);

#else
        fs::copy_file(newSrc, newDst, fs::copy_option::overwrite_if_exists, ec);
#endif
#endif
        if (ec) {
          GERROR << "Copy File Failed: " << ec.message();
          return false;
        }
      }
    }
    return true;
  }

  GERROR << "Error: unrecognized file." << src;
  return false;
}

bool FileManager::MoveFiles(const std::string &src, const std::string &dst) {
#if defined __has_include
#if __has_include(<filesystem>)
  std::error_code ec;
#else
  boost::system::error_code ec;
#endif
#endif
  const fs::path dst_path(dst);
  const fs::path src_path(src);

  // 待拷贝文件为普通文件，则直接拷贝
  if (fs::is_regular_file(src_path)) {
    // dst is not exist
    if (!fs::exists(dst_path))
      fs::create_directories(dst_path.parent_path());

    // case1: src is a file, dst is also a file
    if (fs::is_regular_file(dst_path) || !fs::exists(dst_path)) {
#if defined __has_include
#if __has_include(<filesystem>)
      fs::rename(src_path, dst_path, ec);
#else
      fs::rename(src_path, dst_path, ec);
#endif
#endif
      if (ec) {
        GERROR << "Copy File Failed: " << ec.message();
        return false;
      }
    }
    // case2: src is a file, dst is a directory
    else if (fs::is_directory(dst_path)) {
      fs::path dst_file = dst_path / src_path.filename();
#if defined __has_include
#if __has_include(<filesystem>)
      fs::rename(src_path, dst_file, ec);

#else
      fs::rename(src_path, dst_file, ec);
#endif
#endif
      if (ec) {
        GERROR << "Copy File Failed: " << ec.message();
        return false;
      }
    }
    // case3: src is a file, dst is neither file nor directory
    else {
      if (ec) {
        GERROR << "Copy File Failed: dst is neither file nor directory. ec: "
               << ec.message();
        return false;
      }
    }
    return true;
  }

  // 待拷贝文件为目录，则递归拷贝
  if (fs::is_directory(src_path)) {
    if (!fs::exists(dst_path)) {
      fs::create_directories(dst_path);
    }
    for (fs::directory_iterator it(src_path); it != fs::directory_iterator();
         ++it) {
      const fs::path newSrc = src_path / (it->path()).filename();
      const fs::path newDst = dst_path / (it->path()).filename();
      if (fs::is_directory(newSrc)) {
        CopyFiles(newSrc.string(), newDst.string());
      } else if (fs::is_regular_file(newSrc)) {
#if defined __has_include
#if __has_include(<filesystem>)
        fs::rename(newSrc, newDst, ec);
#else
        fs::rename(newSrc, newDst, ec);
#endif
#endif
        if (ec) {
          GERROR << "Copy File Failed: " << ec.message();
          return false;
        }
      }
    }
    return true;
  }

  GERROR << "Error: unrecognized file." << src;
  return false;
}

bool FileManager::CopyDirectory(const std::string &strSourceDir,
                                const std::string &strDestDir) {
#if defined __has_include
#if __has_include(<filesystem>)
  std::error_code ec;
#else
  boost::system::error_code ec;
#endif
#endif
  // 设置遍历结束标志，用recursive_directory_iterator即可循环的遍历目录
  fs::recursive_directory_iterator end;

  for (fs::recursive_directory_iterator pos(strSourceDir); pos != end; ++pos) {
    // 过滤掉目录和子目录为空的情况
    if (fs::is_directory(*pos))
      continue;
    std::string strAppPath = fs::path(*pos).string();
    std::string strRestorePath;
    // replace_first_copy: 在strAppPath中查找strSourceDir字符串
    // 找到则用strDestDir替换, 替换后的字符串保存在一个输出迭代器中
    boost::algorithm::replace_first_copy(std::back_inserter(strRestorePath),
                                         strAppPath, strSourceDir, strDestDir);
    if (!fs::exists(fs::path(strRestorePath).parent_path())) {
      fs::create_directories(fs::path(strRestorePath).parent_path(), ec);
    }
#if defined __has_include
#if __has_include(<filesystem>)
    fs::copy_file(strAppPath, strRestorePath,
                  fs::copy_options::overwrite_existing, ec);
#else
    fs::copy_file(strAppPath, strRestorePath,
                  fs::copy_option::overwrite_if_exists, ec);
#endif
#endif
  }
  if (ec)
    return false;

  return true;
}

bool FileManager::GetFileNameInPath(const std::string &path,
                                    std::string &filename) {
  const fs::path _path_(path);
  if (fs::is_regular_file(_path_)) {
    filename = _path_.filename().string();
    std::string extension = _path_.extension().string();
    filename.erase(filename.size() - extension.size());
    return true;
  } else
    return false;
}

std::string FileManager::GetFileExtension(const std::string &path) {
  const fs::path _path_(path);
  std::string extension_name = _path_.extension().string();
  return extension_name;
}

std::string FileManager::GetTerminalDir() {
#if defined __has_include
#if __has_include(<filesystem>)
  auto curr_path = std::filesystem::current_path();
#else
  auto curr_path = boost::filesystem::current_path();
#endif
#endif
  return curr_path.string();
}

std::string FileManager::GetCurrentDir() {
  std::string path;
  std::string curr_program_path = boost::dll::program_location().string();
  const fs::path curr_path(curr_program_path);
  path = curr_path.parent_path().string();
  return path;
}

} // namespace common
