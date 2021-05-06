#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>

#include "app_profiler.h"
#include "signal_handler.h"

void AppProfiler::ProfilerThread() {
  auto &profiler = AppProfiler::GetInstance();
  auto pid = getpid();
  std::stringstream fname_ss;
  fname_ss << profiler.perf_log_dir << "/"
           << "perflog." << pid << ".json";
  std::ofstream profile_file(fname_ss.str().c_str());
  profile_file << "[";
  bool first_record = true;

  auto start_us_tp = std::chrono::time_point_cast<std::chrono::microseconds>(
      profiler.start_tp);
  uint64_t start_us_count = start_us_tp.time_since_epoch().count();

  while (AppProfiler::Active()) {
    json record = profiler.queue.pop();
    if (record.count("shutdown")) {
      break;
    } else {
      if (first_record) {
        first_record = false;
      } else {
        profile_file << ",\n";
      }
      record["ts"] = record["ts"].get<uint64_t>() - start_us_count;
      profile_file << record;
    }
  }

  profile_file << "]\n";
  profile_file.close();
}

void AppProfiler::Start() {
  auto &profiler = AppProfiler::GetInstance();
  profiler.start_tp = std::chrono::steady_clock::now();
  profiler.active = true;
  profiler.worker_thread = std::move(std::thread(AppProfiler::ProfilerThread));
  SingalHandler::Register(AppProfiler::ShutDown);
  std::atexit(AppProfiler::ShutDown);
}

void AppProfiler::SetLogDir(const std::string &logdir) {
  auto &profiler = AppProfiler::GetInstance();
  profiler.perf_log_dir = logdir;
}

void AppProfiler::ShutDown() {
  auto &profiler = AppProfiler::GetInstance();
  if (AppProfiler::Active()) {
    profiler.active = false;
    AppProfiler::RecordEvent({{"shutdown", true}});
    profiler.worker_thread.join();
    std::cout << "AppProfiler shutdown" << std::endl;
  }
}

bool AppProfiler::Active() {
  auto &profiler = AppProfiler::GetInstance();
  return profiler.active.load();
}

void AppProfiler::RecordEvent(json jevent) {
  auto &profiler = AppProfiler::GetInstance();
  if (AppProfiler::Active()) {
    profiler.queue.push(jevent);
  }
}

AppProfiler &AppProfiler::GetInstance() {
  static AppProfiler instance;
  return instance;
}

AppProfileGuard::AppProfileGuard(const char *name, const char *fname,
                                 int lineno, bool raii)
    : record_name(name), record_file_name(fname), record_file_lineno(lineno),
      thread_name(GetThreadName()), stream_name(GetStreamName()), raii(raii) {
  if (raii) {
    AddBeginRecord();
  }
}

AppProfileGuard::~AppProfileGuard() {
  if (raii) {
    AddEndRecord();
  }
}

void AppProfileGuard::AddBeginRecord() {
  auto current_tp = std::chrono::steady_clock::now();
  start_us =
      std::chrono::time_point_cast<std::chrono::microseconds>(current_tp);
}

void AppProfileGuard::AddEndRecord() {
  AddRecord(record_name.c_str(), record_file_name, record_file_lineno,
            thread_name, stream_name);
}

void AppProfileGuard::AddRecord(const char *name, const char *fname, int lineno,
                                const std::string &tname,
                                const std::string &sname) const {
  auto current_tp = std::chrono::steady_clock::now();
  auto us_tp =
      std::chrono::time_point_cast<std::chrono::microseconds>(current_tp);
  auto start_count = start_us.time_since_epoch().count();
  auto end_count = us_tp.time_since_epoch().count();

  json record{{"name", name},
              {"ph", "X"},
              {"tid", tname},
              {"pid", sname},
              {"ts", start_count},
              {"dur", end_count - start_count},
              {"args", {{"file", fname}, {"lineno", lineno}}}};

  AppProfiler::RecordEvent(record);
}

thread_local std::string profiler_thread_name;
thread_local std::string profiler_stream_name;

const std::string &GetThreadName() { return profiler_thread_name; }

const std::string &GetStreamName() { profiler_stream_name; }

void SetThreadName(const std::string &thread_name) {
  profiler_thread_name = thread_name;
}

void SetStreamName(const std::string &stream_name) {
  profiler_stream_name = stream_name;
}
