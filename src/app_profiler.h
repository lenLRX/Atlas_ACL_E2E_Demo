#ifndef __APP_PROFILER_H__
#define __APP_PROFILER_H__

#include <atomic>
#include <thread>
#include <chrono>
#include <string>
#include <nlohmann/json.hpp>

#include "util.h"

using json = nlohmann::json;

class AppProfiler {
public:
   static void Start();
   static void ShutDown();
   static bool Active();
   static void RecordEvent(json jevent);
private:
   static void ProfilerThread();

   static AppProfiler& GetInstance();
   std::atomic<bool> active{false};
   ThreadSafeQueue<json> queue;
   std::thread worker_thread;
   std::chrono::time_point<std::chrono::steady_clock> start_tp;
};


class AppProfileGuard {
public:
  AppProfileGuard(const char* name, const char* fname, int lineno, bool raii);
  ~AppProfileGuard();
  void AddBeginRecord();
  void AddEndRecord();
private:
  void AddRecord(const char* name, const char* fname, int lineno,
                 const std::string& type, const std::string& tname,
                 const std::string& sname) const;
  std::string record_name;
  const char* record_file_name;
  int record_file_lineno;
  std::string thread_name;
  std::string stream_name;
  bool raii;
};

#define APP_PROFILE(name) \
  auto __CONCAT__(temp_app_perf_obj_, __LINE__) = \
  AppProfileGuard(#name, __FILE__, __LINE__, true)

const std::string& GetThreadName();
const std::string& GetStreamName();

void SetThreadName(const std::string& thread_name);
void SetStreamName(const std::string& stream_name);


#endif//__APP_PROFILER_H__