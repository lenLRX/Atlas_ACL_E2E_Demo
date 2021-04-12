#ifndef __YOLOV3_STREAM_H__
#define __YOLOV3_STREAM_H__

#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "acl_model.h"
#include "app_profiler.h"

using json = nlohmann::json;

class Yolov3Model {
public:
  // input type: <raw image, resized image>
  using InTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;
  // input type: <box info, resized image>
  using OutTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;

  Yolov3Model(const std::string &path, aclrtStream stream);
  OutTy Process(InTy bufferx2);

private:
  ACLModel yolov3_model;
  aclrtStream model_stream;
};

class Yolov3PostProcess {
public:
  using InTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;
  using OutTy = DeviceBufferPtr;
  Yolov3PostProcess() = default;
  OutTy Process(InTy input);
};

std::thread MakeYolov3Stream(json config);

#endif //__YOLOV3_STREAM_H__