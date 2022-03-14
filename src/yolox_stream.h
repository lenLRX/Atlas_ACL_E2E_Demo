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

  YoloXPreProcess(int w, int h, bool enable_neon, float scale);

  OutTy Process(InTy bufferx2);
  OutTy ProcessWithNeon(InTy input);
  OutTy ProcessWithoutNeon(InTy input);

private:
  int width;
  int height;
  bool enable_neon;
  float scale;
};

class YoloXModel {
public:
  // input type: <raw image, resized image>
  using InTy = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;
  // output type: <preds, raw image>
  using OutTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;

  YoloXModel(const std::string &path, aclrtStream stream);
  OutTy Process(InTy bufferx2);

private:
  ACLModel yolox_model;
  aclrtStream model_stream;
};

class YoloXPostProcess {
public:
  // input type: <preds, raw image>
  using InTy = std::tuple<ACLModel::DevBufferVec, DeviceBufferPtr>;
  using OutTy = DeviceBufferPtr;
  YoloXPostProcess(int width, int height, int model_width, int model_height,
                   int box_num, int class_num);
  OutTy Process(InTy input);

private:
  int width;
  int height;
  int box_num;
  int class_num;
  float h_ratio;
  float w_ratio;
};

// https://github.com/numpy/numpy/issues/11925
class YoloXPyEnv {
public:
  static YoloXPyEnv &GetInstance() {
    static YoloXPyEnv env;
    return env;
  }

  PyObject *GetPostProcessFn() const { return post_processing_fn; }

private:
  YoloXPyEnv() {
    Py_Initialize();
    { // expansion of macro import_array
      if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError,
                        "numpy.core.multiarray failed to import");
      }
    }
    numpy = PyImport_ImportModule("numpy");
    yolox_module = PyImport_ImportModule("yolox");
    if (yolox_module == NULL) {
      PyErr_Print();
      return;
    }

    post_processing_fn =
        PyObject_GetAttrString(yolox_module, "post_processing");
    if (post_processing_fn == NULL) {
      PyErr_Print();
      return;
    }

    PyEval_InitThreads();
    _save = PyEval_SaveThread();
  }

  ~YoloXPyEnv() {
    PyEval_RestoreThread(_save);
    Py_FinalizeEx();
  }

  PyThreadState *_save;
  PyObject *numpy;
  PyObject *yolox_module;
  PyObject *post_processing_fn;
};

std::thread MakeYoloXStream(json config);

#endif //__YOLOX_STREAM_H__