#ifndef __YOLOv5_STREAM_H__
#define __YOLOv5_STREAM_H__

#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "acl_model.h"
#include "app_profiler.h"

using json = nlohmann::json;

// use aicore aipp to convert YUV420SP to RGB
class Yolov5PreProcess {
public:
  // input type: <raw image, resized image (YUV420SP)>
  using InTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;
  // output type: <raw image, resized image (RGB)>
  using OutTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;

  Yolov5PreProcess(int w, int h);

  OutTy Process(InTy bufferx2);

private:
  int width;
  int height;
};

class Yolov5Model {
public:
  // input type: <raw image, resized image>
  using InTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;
  // output type: <<confs, box info>, raw image>
  using OutTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;

  Yolov5Model(const std::string &path, aclrtStream stream);
  OutTy Process(InTy bufferx2);

private:
  ACLModel yolov5_model;
  aclrtStream model_stream;
};

class Yolov5PostProcess {
public:
  // input type: <<confs, box info>, raw image>
  using InTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;
  using OutTy = DeviceBufferPtr;
  Yolov5PostProcess(int width, int height, int model_width, int model_height);
  OutTy Process(InTy input);

private:
  int width;
  int height;
  float h_ratio;
  float w_ratio;
};

std::thread MakeYolov5Stream(json config);

#endif //__YOLOv5_STREAM_H__