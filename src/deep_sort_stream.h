#ifndef __DEEP_SORT_STREAM_H__
#define __DEEP_SORT_STREAM_H__

#include <Python.h>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "acl_model.h"
#include "app_profiler.h"
#include "drawing.h"
#include "vpc_batch_crop.h"
#include "yolov3_stream.h"

using json = nlohmann::json;

using BoxInfo = std::tuple<std::vector<int32_t>, std::vector<float>>;

class DeepSortCropProcess {
public:
  // input type: <boxinfo, raw image>
  using InTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;
  // output type: <boxinfo, raw image, croped images batches>
  using OutTy = std::tuple<BoxInfo, DeviceBufferPtr, ACLModel::DevBufferVec>;
  DeepSortCropProcess(aclrtStream stream, int width, int height);
  OutTy Process(InTy input);

private:
  aclrtStream crop_stream;
  VPCBatchCrop crop_engine;
  int width;
  int height;
  float h_ratio;
  float w_ratio;
};

class DeepSortModel {
public:
  // input type: <boxinfo, raw image, croped images batches>
  using InTy = std::tuple<BoxInfo, DeviceBufferPtr, ACLModel::DevBufferVec>;
  // output type: <box info, raw image, deep sort features>
  using OutTy = std::tuple<BoxInfo, DeviceBufferPtr, uint8_t *>;

  DeepSortModel(const std::string &path, aclrtStream stream);
  OutTy Process(InTy input_tup);

private:
  ACLModel deepsort_model;
  aclrtStream model_stream;
};

struct TrackResult {
  TrackResult(int track_id, int x1, int y1, int x2, int y2)
      : track_id(track_id), x1(x1), y1(y1), x2(x2), y2(y2) {}
  int track_id;
  int x1;
  int y1;
  int x2;
  int y2;
};

using TrackResultVec = std::vector<TrackResult>;

class DeepSortTracker {
public:
  // input type: <box info, raw image, deep sort features>
  using InTy = std::tuple<BoxInfo, DeviceBufferPtr, uint8_t *>;
  // output type: <tracker box, raw image>
  using OutTy = std::tuple<TrackResultVec, DeviceBufferPtr>;
  DeepSortTracker();
  OutTy Process(InTy input_tup);

private:
  PyObject *tracker_ctx;
  PyObject *update_tracker_fn;
  PyObject *query_tracking_fn;
};

class DeepSortPostProcess {
public:
  // input type: <tracker box, raw image>
  using InTy = std::tuple<TrackResultVec, DeviceBufferPtr>;
  // output type: <labeled image>
  using OutTy = DeviceBufferPtr;
  DeepSortPostProcess(int width, int height);
  OutTy Process(InTy input);

private:
  int width;
  int height;

  std::vector<YUVColor> colors;
};

std::thread MakeDeepSortStream(json config);

#endif //__DEEP_SORT_STREAM_H__