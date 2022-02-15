#ifndef __YOLOX_STREAM_H__
#define __YOLOX_STREAM_H__

#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "acl_model.h"
#include "app_profiler.h"

using json = nlohmann::json;

class YoloXPreProcess {
public:
  // input type: <raw image, resized image (YUV420SP)>
  using InTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;
  // output type: <raw image, resized image (RGB)>
  using OutTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;

  YoloXPreProcess(int w, int h, bool enable_neon);

  OutTy Process(InTy bufferx2);
  OutTy ProcessWithNeon(InTy input);
  OutTy ProcessWithoutNeon(InTy input);

private:
  int width;
  int height;
  bool enable_neon;
};

class YoloXModel {
public:
  // input type: <raw image, resized image>
  using InTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;
  // output type: <<confs, box info>, raw image>
  using OutTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;

  YoloXModel(const std::string &path, aclrtStream stream);
  OutTy Process(InTy bufferx2);

private:
  ACLModel yolox_model;
  aclrtStream model_stream;
};

class YoloXPostProcess {
public:
  // input type: <<confs, box info>, raw image>
  using InTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;
  using OutTy = DeviceBufferPtr;
  YoloXPostProcess(int width, int height, int model_width, int model_height);
  OutTy Process(InTy input);

private:
  int width;
  int height;
  float h_ratio;
  float w_ratio;
};

std::thread MakeYoloXStream(json config);

#endif //__YOLOX_STREAM_H__