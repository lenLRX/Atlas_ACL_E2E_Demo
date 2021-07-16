#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "acl_model.h"
#include "util.h"

using json = nlohmann::json;

void TestCase(json test_cfg, int test_iter) {
  std::string model_path = test_cfg.at("test_model");
  int batch_size = test_cfg.at("batch_size");
  aclrtStream stream;
  CHECK_ACL(aclrtCreateStream(&stream));
  ACLModel model(stream);
  model.Init(model_path.c_str());

  // prepare input output
  ACLModel::DevBufferVec input;
  ACLModel::DevBufferVec output;

  for (size_t input_size : model.GetInputBufferSizes()) {
    void *buf;
    CHECK_ACL(aclrtMalloc(&buf, input_size, ACL_MEM_MALLOC_HUGE_FIRST));
    DeviceBufferPtr input_buffer;
    input_buffer = std::make_shared<DeviceBuffer>(
        buf, input_size, DeviceBuffer::DevMemDeleter());
    input.push_back(input_buffer);
  }

  for (size_t output_size : model.GetOutputBufferSizes()) {
    void *buf;
    CHECK_ACL(aclrtMalloc(&buf, output_size, ACL_MEM_MALLOC_HUGE_FIRST));
    DeviceBufferPtr output_buffer;
    output_buffer = std::make_shared<DeviceBuffer>(
        buf, output_size, DeviceBuffer::DevMemDeleter());
    output.push_back(output_buffer);
  }

  std::chrono::steady_clock::time_point start = steady_clock::now();
  for (int i = 0; i < test_iter; ++i) {
    model.Infer(input, output);
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));
  std::chrono::steady_clock::time_point end = steady_clock::now();
  auto duration = end - start;
  microseconds duration_us = duration_cast<microseconds>(duration);
  auto duration_s = duration_us.count() / 1000.f / 1000.f;
  auto batch_time = duration_s / test_iter;
  auto fps = 1 / batch_time * batch_size;
  std::cout << "model: " << model_path
            << " batch_size: " << batch_size << " fps: " << fps
            << std::endl;
}

int main(int argc, char **argv) {
  CHECK_ACL(aclInit(nullptr));
  if (argc != 3) {
    std::cerr << "invalid arguments!\n"
              << "usage: ./build/benchmark_demo -c config.json" << std::endl;
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

  CHECK_ACL(aclrtSetDevice(0));
  aclrtContext ctx;
  CHECK_ACL(aclrtCreateContext(&ctx, 0));
  CHECK_ACL(aclrtSetCurrentContext(ctx));

  json j;
  config_fs >> j;
  json tests = j.at("tests");
  int test_iter = j.at("test_iter");
  for (auto &test_cfg : tests) {
    TestCase(test_cfg, test_iter);
  }
}