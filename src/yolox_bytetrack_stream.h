#ifndef __YOLOX_BYTETRACK_STREAM_H__
#define __YOLOX_BYTETRACK_STREAM_H__

#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "acl_model.h"
#include "app_profiler.h"
#include "bytetrack.h"
#include "yolox_stream.h"

using json = nlohmann::json;

enum ByteTrackImplType { BYTETRACK_CPP, BYTETRACK_PY };

class ByteTrackDetectInfo {
public:
  ByteTrackDetectInfo(PyObject *dets) : dets(dets) {}
  ByteTrackDetectInfo(const std::vector<Object> &obj) : objects(obj) {}
  PyObject *dets;
  std::vector<Object> objects;
};

class ByteTrackYoloXPostProcess {
public:
  // input type: <preds, raw image>
  using InTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;
  // output type: <labeled image>
  using OutTy = std::tuple<ByteTrackDetectInfo *, DeviceBufferPtr>;
  ByteTrackYoloXPostProcess(int model_height, int model_width, int box_num,
                            int class_num, float nms_thr, float score_thr,
                            ByteTrackImplType impl_type);
  OutTy Process(InTy input_tup);

private:
  void InitPY();
  void InitCPP();

  OutTy ProcessPY(InTy input_tup);
  OutTy ProcessCPP(InTy input_tup);

  int model_h;
  int model_w;
  int box_num;
  int class_num;
  float nms_thr;
  float score_thr;
  std::vector<GridAndStride> grid_strides;
  ByteTrackImplType impl_type;
};

class ByteTrackTracker {
public:
  // input type: <preds, raw image>
  using InTy = std::tuple<ByteTrackDetectInfo *, DeviceBufferPtr>;
  // output type: <labeled image>
  using OutTy = DeviceBufferPtr;
  ByteTrackTracker(double framerate, int raw_img_height, int raw_img_width,
                   int model_height, int model_width, int box_num,
                   int class_num, float nms_thr, float score_thr,
                   ByteTrackImplType impl_type);
  OutTy Process(InTy input_tup);

private:
  void InitPY();
  void InitCPP();

  OutTy ProcessPY(InTy input_tup);
  OutTy ProcessCPP(InTy input_tup);

  BYTETracker cpp_tracker;
  PyObject *bytetracker{nullptr};
  double framerate;
  int raw_img_h;
  int raw_img_w;
  int model_h;
  int model_w;
  float h_ratio;
  float w_ratio;
  float nms_thr;
  float score_thr;
  int box_num;
  int class_num;
  std::vector<YUVColor> colors;
  ByteTrackImplType impl_type;
};

class ByteTrackPreProcess {
public:
  // input type: <raw image, resized image>
  using InTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;
  // output type: <raw image, resized image>
  using OutTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;

  ByteTrackPreProcess(int model_height, int model_width,
                      const std::vector<float> &vec_mean,
                      const std::vector<float> &vec_std);

  OutTy Process(InTy bufferx2);

private:
  int height;
  int width;
  std::vector<float> mean;
  std::vector<float> std;
};

#endif //__YOLOX_BYTETRACK_STREAM_H__