#pragma once
#include <glog/logging.h>
#include <glog/raw_logging.h>
#include <stdlib.h>
#include <cstdarg>
#include <string>

#define MODULE_NAME "MAIN_MODULE"

#if defined __has_include
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#endif
#endif

/*  调整glog默认配置  */
#define DEFAULT_CONFIG
#ifdef DEFAULT_CONFIG
namespace common
{
class Logger
{
public:
    Logger(int &argsize, char **&program);
    ~Logger();
}; // namespace Logger
} // namespace common
#endif

#define LEFT_BRACKET "["
#define RIGHT_BRACKET "]"


#define GDEBUG_MODULE(module) \
    VLOG(4) << LEFT_BRACKET << module << RIGHT_BRACKET << "[DEBUG] "
#define GDEBUG GDEBUG_MODULE(MODULE_NAME)
#define GINFO GLOG_MODULE(MODULE_NAME, INFO)
#define GWARN GLOG_MODULE(MODULE_NAME, WARN)
#define GERROR GLOG_MODULE(MODULE_NAME, ERROR)
#define GFATAL GLOG_MODULE(MODULE_NAME, FATAL)

#ifndef GLOG_MODULE_STREAM
#define GLOG_MODULE_STREAM(log_severity) GLOG_MODULE_STREAM_##log_severity
#endif

#ifndef GLOG_MODULE
#define GLOG_MODULE(module, log_severity) \
    GLOG_MODULE_STREAM(log_severity)      \
    (module)
#endif

#define GLOG_MODULE_STREAM_INFO(module)                           \
    google::LogMessage(__FILE__, __LINE__, google::INFO).stream() \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define GLOG_MODULE_STREAM_WARN(module)                              \
    google::LogMessage(__FILE__, __LINE__, google::WARNING).stream() \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define GLOG_MODULE_STREAM_ERROR(module)                           \
    google::LogMessage(__FILE__, __LINE__, google::ERROR).stream() \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define GLOG_MODULE_STREAM_FATAL(module)                           \
    google::LogMessage(__FILE__, __LINE__, google::FATAL).stream() \
        << LEFT_BRACKET << module << RIGHT_BRACKET

#define GINFO_IF(cond) ALOG_IF(INFO, cond, MODULE_NAME)
#define GWARN_IF(cond) ALOG_IF(WARN, cond, MODULE_NAME)
#define GERROR_IF(cond) ALOG_IF(ERROR, cond, MODULE_NAME)
#define GFATAL_IF(cond) ALOG_IF(FATAL, cond, MODULE_NAME)
#define ALOG_IF(severity, cond, module) \
    !(cond) ? (void)0                   \
            : google::LogMessageVoidify() & GLOG_MODULE(module, severity)

#define ACHECK(cond) CHECK(cond) << LEFT_BRACKET << MODULE_NAME << RIGHT_BRACKET

#define GINFO_EVERY(freq) \
    LOG_EVERY_N(INFO, freq) << LEFT_BRACKET << MODULE_NAME << RIGHT_BRACKET
#define GWARN_EVERY(freq) \
    LOG_EVERY_N(WARNING, freq) << LEFT_BRACKET << MODULE_NAME << RIGHT_BRACKET
#define GERROR_EVERY(freq) \
    LOG_EVERY_N(ERROR, freq) << LEFT_BRACKET << MODULE_NAME << RIGHT_BRACKET

#if !defined(RETURN_IF_NULL)
#define RETURN_IF_NULL(ptr)              \
    if (ptr == nullptr)                  \
    {                                    \
        GWARN << #ptr << " is nullptr."; \
        return;                          \
    }
#endif

#if !defined(RETURN_VAL_IF_NULL)
#define RETURN_VAL_IF_NULL(ptr, val)     \
    if (ptr == nullptr)                  \
    {                                    \
        GWARN << #ptr << " is nullptr."; \
        return val;                      \
    }
#endif

#if !defined(RETURN_IF)
#define RETURN_IF(condition)               \
    if (condition)                         \
    {                                      \
        GWARN << #condition << " is met."; \
        return;                            \
    }
#endif

#if !defined(RETURN_VAL_IF)
#define RETURN_VAL_IF(condition, val)      \
    if (condition)                         \
    {                                      \
        GWARN << #condition << " is met."; \
        return val;                        \
    }
#endif

#if !defined(_RETURN_VAL_IF_NULL2__)
#define _RETURN_VAL_IF_NULL2__
#define RETURN_VAL_IF_NULL2(ptr, val) \
    if (ptr == nullptr)               \
    {                                 \
        return (val);                 \
    }
#endif

#if !defined(_RETURN_VAL_IF2__)
#define _RETURN_VAL_IF2__
#define RETURN_VAL_IF2(condition, val) \
    if (condition)                     \
    {                                  \
        return (val);                  \
    }
#endif

#if !defined(_RETURN_IF2__)
#define _RETURN_IF2__
#define RETURN_IF2(condition) \
    if (condition)            \
    {                         \
        return;               \
    }
#endif
