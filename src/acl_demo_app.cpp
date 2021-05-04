#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <signal.h>
#include <string>

#include "app_profiler.h"
#include "signal_handler.h"
#include "stream_factory.h"
#include "util.h"

using json = nlohmann::json;

std::vector<std::thread> CreateStreamByConfig(const json &jconfig) {
  std::vector<std::thread> vec_threads;
  if (!jconfig.is_array()) {
    std::cerr << "error while parsing config, config is not array!"
              << std::endl;
    exit(-1);
  }
  int stream_id = 0;
  for (const json &jstream : jconfig) {
    vec_threads.push_back(StreamFactory::MakeStream(jstream, stream_id));
    ++stream_id;
  }
  return vec_threads;
}

int main(int argc, char **argv) {
  SingalHandler::RegisterSignal();
  CHECK_ACL(aclInit(nullptr));
  uint32_t dev_count = 0;
  CHECK_ACL(aclrtGetDeviceCount(&dev_count));
  std::cout << "total dev count: " << dev_count << std::endl;
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