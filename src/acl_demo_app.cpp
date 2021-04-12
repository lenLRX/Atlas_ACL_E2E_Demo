#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <signal.h>
#include <string>

#include "app_profiler.h"
#include "util.h"
#include "yolov3_stream.h"

using json = nlohmann::json;

std::vector<std::thread> CreateStreamByConfig(const json &jconfig) {
  std::vector<std::thread> vec_threads;
  if (!jconfig.is_array()) {
    std::cerr << "error while parsing config, config is not array!"
              << std::endl;
    exit(-1);
  }
  for (const json &jstream : jconfig) {
    vec_threads.push_back(MakeYolov3Stream(jstream));
  }
  return vec_threads;
}

int main(int argc, char **argv) {
  signal(SIGINT, exit);
  CHECK_ACL(aclInit(nullptr));
  if (argc != 3) {
    std::cerr << "invalid arguments!\n"
              << "usage: ./build/acl_demo_app -c config.json" << std::endl;
    return -1;
  }
  std::string option{argv[1]};
  if (option != "-c") {
    std::cerr << "invalid option!\n"
              << "expected -c but got: " << option << std::endl;
    return -1;
  }

  std::string config_file_path{argv[2]};
  std::ifstream config_fs(config_file_path.c_str());
  if (!config_fs.good()) {
    std::cerr << "failed to open config file: " << config_file_path
              << std::endl;
    return -1;
  }

  json j;
  config_fs >> j;

  json streams = j.at("streams");

  auto app_config = j.at("config");
  if (app_config.count("app_perf")) {
    bool perf = app_config.at("app_perf");
    if (perf) {
      if (app_config.count("perflog_path")) {
        AppProfiler::SetLogDir(app_config.at("perflog_path"));
      }
      AppProfiler::Start();
    }
  }

  auto threads = CreateStreamByConfig(streams);
  for (auto &t : threads) {
    t.join();
  }
}