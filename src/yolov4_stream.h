#ifndef __YOLOv4_STREAM_H__
#define __YOLOv4_STREAM_H__

#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "acl_model.h"
#include "app_profiler.h"

using json = nlohmann::json;

class Yolov4Model {
public:
  // input type: <raw image, resized image>
  using InTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;
  // output type: <<confs, box info>, raw image>
  using OutTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;

  Yolov4Model(const std::string &path, aclrtStream stream);
  OutTy Process(InTy bufferx2);

private:
  ACLModel yolov4_model;
  aclrtStream model_stream;
};

class Yolov4PostProcess {
public:
  // input type: <<confs, box info>, raw image>
  using InTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;
  using OutTy = DeviceBufferPtr;
  Yolov4PostProcess(int width, int height);
  OutTy Process(InTy input);

private:
  int width;
  int height;
  float h_ratio;
  float w_ratio;
};

std::thread MakeYolov4Stream(json config);

#endif //__YOLOv4_STREAM_H__