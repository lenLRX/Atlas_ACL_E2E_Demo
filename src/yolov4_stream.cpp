#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "acl_model.h"
#include "camera_input.h"
#include "drawing.h"
#include "dvpp_decoder.h"
#include "dvpp_encoder.h"
#include "ffmpeg_input.h"
#include "ffmpeg_output.h"
#include "util.h"
#include "vpc_resize.h"

#include "acl_cb_thread.h"
#include "app_profiler.h"
#include "device_manager.h"
#include "signal_handler.h"
#include "stream_factory.h"
#include "task_node.h"
#include "yolov4_stream.h"

#define CHECK_PY_ERR(obj)                                                      \
  if (obj == NULL) {                                                           \
    PyErr_Print();                                                             \
    throw std::runtime_error("CHECK_PY_ERR");                                  \
  }

// https://github.com/numpy/numpy/issues/11925
class Yolov4PyEnv {
public:
  static Yolov4PyEnv &GetInstance() {
    static Yolov4PyEnv env;
    return env;
  }

  PyObject *GetPostProcessFn() const { return post_processing_fn; }

private:
  Yolov4PyEnv() {
    Py_Initialize();
    { // expansion of macro import_array
      if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError,
                        "numpy.core.multiarray failed to import");
      }
    }
    numpy = PyImport_ImportModule("numpy");
    yolov4_module = PyImport_ImportModule("yolov4");
    if (yolov4_module == NULL) {
      PyErr_Print();
      return;
    }

    post_processing_fn =
        PyObject_GetAttrString(yolov4_module, "post_processing");
    if (post_processing_fn == NULL) {
      PyErr_Print();
      return;
    }

    PyEval_InitThreads();
    _save = PyEval_SaveThread();
  }

  ~Yolov4PyEnv() {
    PyEval_RestoreThread(_save);
    Py_FinalizeEx();
  }

  PyThreadState *_save;
  PyObject *numpy;
  PyObject *yolov4_module;
  PyObject *post_processing_fn;
};

using namespace std::chrono_literals;

const static int yolov4_model_size = 416;

Yolov4Model::Yolov4Model(const std::string &path, aclrtStream stream)
    : yolov4_model(stream), model_stream(stream) {
  yolov4_model.Init(path.c_str());
  std::cout << "Model Info:" << std::endl;
  std::cout << yolov4_model.ToString();
}

Yolov4Model::OutTy Yolov4Model::Process(Yolov4Model::InTy bufferx2) {
  APP_PROFILE(Yolov4Model);
  auto output_buffers = yolov4_model.Infer({std::get<1>(bufferx2)});
  return {output_buffers, std::get<0>(bufferx2)};
}

Yolov4PostProcess::Yolov4PostProcess(int width, int height)
    : width(width), height(height) {
  h_ratio = height / (float)yolov4_model_size;
  w_ratio = width / (float)yolov4_model_size;
}

Yolov4PostProcess::OutTy Yolov4PostProcess::Process(InTy input) {
  APP_PROFILE(Yolov4PostProcess);
  Yolov4PyEnv &env = Yolov4PyEnv::GetInstance();
  const auto &infer_results = std::get<0>(input);
  auto image_buffer = std::get<1>(input);

  float *confs = (float *)infer_results[0]->GetHostPtr();
  float *boxes = (float *)infer_results[1]->GetHostPtr();

  PyGILGuard py_gil_guard;

  npy_intp confs_dim[3] = {1, 10647, 80};
  const int confs_nd = 3;

  PyObject *confs_arr =
      PyArray_SimpleNewFromData(confs_nd, confs_dim, NPY_FLOAT32, confs);
  PyArray_CLEARFLAGS((PyArrayObject *)confs_arr, NPY_ARRAY_OWNDATA);

  npy_intp boxes_dim[4] = {1, 10647, 1, 4};
  const int boxes_nd = 4;

  PyObject *boxes_arr =
      PyArray_SimpleNewFromData(boxes_nd, boxes_dim, NPY_FLOAT32, boxes);
  PyArray_CLEARFLAGS((PyArrayObject *)boxes_arr, NPY_ARRAY_OWNDATA);

  PyObject *post_processing_fn = env.GetPostProcessFn();

  const float conf_thresh = 0.4;
  const float nms_thresh = 0.6;
  PyObject *post_processing_arg =
      Py_BuildValue("(O,O,f,f)", confs_arr, boxes_arr, conf_thresh, nms_thresh);

  Py_XDECREF(confs_arr);
  Py_XDECREF(boxes_arr);

  if (post_processing_arg == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to make post processing args");
  }

  PyObject *bboxes =
      PyObject_Call(post_processing_fn, post_processing_arg, NULL);

  Py_XDECREF(post_processing_arg);

  if (bboxes == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to call post processing");
  }

  // std::cout << "result box num:" << box_out_num << std::endl;

  YUV420SPImage img((uint8_t *)image_buffer->GetHostPtr(), height, width);
  YUVColor box_color(0, 0, 0xff); // Red?

  PyObject *first_batch = PyList_GetItem(bboxes, 0);

  Py_ssize_t bboxes_size = PyList_Size(first_batch);
  for (Py_ssize_t i = 0; i < bboxes_size; ++i) {
    PyObject *box = PyList_GetItem(first_batch, i);
    CHECK_PY_ERR(box);
    PyObject *py_x1 = PyList_GetItem(box, 0);
    CHECK_PY_ERR(py_x1);
    int x1 = std::round(PyFloat_AsDouble(py_x1) * width);
    PyObject *py_y1 = PyList_GetItem(box, 1);
    CHECK_PY_ERR(py_y1);
    int y1 = std::round(PyFloat_AsDouble(py_y1) * height);
    PyObject *py_x2 = PyList_GetItem(box, 2);
    CHECK_PY_ERR(py_x2);
    int x2 = std::round(PyFloat_AsDouble(py_x2) * width);
    PyObject *py_y2 = PyList_GetItem(box, 3);
    CHECK_PY_ERR(py_y2);
    int y2 = std::round(PyFloat_AsDouble(py_y2) * height);

    PyObject *py_label = PyList_GetItem(box, 6);
    CHECK_PY_ERR(py_label);
    int label = PyLong_AsLong(py_label);

    img.DrawRect(x1, y1, x2, y2, box_color, 3);
    img.DrawText(x1, y2, yolov3_label_zh_cn[label + 1], box_color);
  }

  Py_XDECREF(bboxes);

  return image_buffer;
}

void Yolov4StreamThread(json config, int id) {
  std::string input_addr = config.at("src");
  std::string output_addr = config.at("dst");
  std::string model_path = config.at("yolov4_model_path");
  bool is_null_output = output_addr == "null";
  bool hardware_enc = false;
  if (config.count("hw_encoder")) {
    hardware_enc = config.at("hw_encoder") && (!is_null_output);
  }

  std::string stream_name = StreamName(input_addr, id);

  AclCallBackThread cb_decoder_thread(stream_name, "DVPP_DECODER");
  AclCallBackThread cb_encoder_thread(stream_name, "DVPP_ENCODER");

  aclrtContext ctx = DeviceManager::AllocateCtx();
  CHECK_ACL(aclrtSetCurrentContext(ctx));
  aclrtStream decoder_stream;
  CHECK_ACL(aclrtCreateStream(&decoder_stream));
  CHECK_ACL(aclrtSubscribeReport(cb_decoder_thread.GetPid(), decoder_stream));

  aclrtStream encoder_stream;
  CHECK_ACL(aclrtCreateStream(&encoder_stream));
  CHECK_ACL(aclrtSubscribeReport(cb_encoder_thread.GetPid(), encoder_stream));

  const int queue_size = 4;

  FFMPEGInput ffmpeg_input;
  CameraInput camera_input;

  int camera_id = ParseCameraInput(input_addr);
  int width, height;
  if (camera_id < 0) {
    ffmpeg_input.Init(input_addr.c_str());
    width = ffmpeg_input.GetWidth();
    height = ffmpeg_input.GetHeight();
  } else {
    camera_input.Init(camera_id);
    width = camera_input.GetWidth();
    height = camera_input.GetHeight();
  }

  using buf_tup_t = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;

  ThreadSafeQueueWithCapacity<AVPacket> decoder_input_queue(queue_size);

  DvppDecoder decoder;
  TaskNode<DvppDecoder, AVPacket, void> decoder_node(&decoder, "DvppDecoder",
                                                     stream_name);
  decoder_node.SetInputQueue(&decoder_input_queue);

  ThreadSafeQueueWithCapacity<DeviceBufferPtr> resize_input_queue(queue_size);
  decoder.SetOutputQueue(&resize_input_queue);
  decoder_node.Start(ctx);

  if (camera_id < 0) {
    decoder.Init(cb_decoder_thread.GetPid(), height, width,
                 ffmpeg_input.GetProfile());
    decoder.SetDeviceCtx(&ctx);
  }

  aclrtStream resize_stream;
  CHECK_ACL(aclrtCreateStream(&resize_stream));
  VPCResizeEngine resize_engine(resize_stream);
  resize_engine.Init(height, width, yolov4_model_size, yolov4_model_size);

  TaskNode<VPCResizeEngine, DeviceBufferPtr, buf_tup_t> resize_engine_node(
      &resize_engine, "VPCResizeEngine", stream_name);
  resize_engine_node.SetInputQueue(&resize_input_queue);

  ThreadSafeQueueWithCapacity<buf_tup_t> yolov4_input_queue(queue_size);
  resize_engine_node.SetOutputQueue(&yolov4_input_queue);
  resize_engine_node.Start(ctx);

  aclrtStream model_stream;
  CHECK_ACL(aclrtCreateStream(&model_stream));
  Yolov4Model yolov4_model(model_path, model_stream);

  TaskNode<Yolov4Model, Yolov4Model::InTy, Yolov4Model::OutTy>
      yolov4_model_node(&yolov4_model, "Yolov4Model", stream_name);

  yolov4_model_node.SetInputQueue(&yolov4_input_queue);

  ThreadSafeQueueWithCapacity<Yolov4Model::OutTy> yolov4_output_queue(
      queue_size);
  yolov4_model_node.SetOutputQueue(&yolov4_output_queue);
  yolov4_model_node.Start(ctx);

  NullOutput<Yolov4PostProcess::InTy> null_out;
  TaskNode<NullOutput<Yolov4PostProcess::InTy>, Yolov4PostProcess::InTy, void>
      null_out_node(&null_out, "NullOutput", stream_name);

  ThreadSafeQueueWithCapacity<Yolov4Model::OutTy> dummy_output_queue(
      queue_size);

  Yolov4PostProcess post_process(width, height);
  TaskNode<Yolov4PostProcess, Yolov4PostProcess::InTy, Yolov4PostProcess::OutTy>
      post_process_node(&post_process, "Yolov4PostProcess", stream_name);

  if (!is_null_output) {
    post_process_node.SetInputQueue(&yolov4_output_queue);
    null_out_node.SetInputQueue(&dummy_output_queue);
  } else {
    post_process_node.SetInputQueue(&dummy_output_queue);
    null_out_node.SetInputQueue(&yolov4_output_queue);
  }

  dummy_output_queue.ShutDown();

  null_out_node.Start(ctx);

  ThreadSafeQueueWithCapacity<Yolov4PostProcess::OutTy>
      yolov4_post_output_queue(queue_size);
  post_process_node.SetOutputQueue(&yolov4_post_output_queue);
  post_process_node.Start(ctx);

  FFMPEGOutput ffmpeg_output;
  if (!is_null_output) {
    if (camera_id < 0) {
      ffmpeg_output.Init(output_addr, height, width,
                         ffmpeg_input.GetFramerate());
    } else {
      ffmpeg_output.Init(output_addr, height, width, camera_input.GetFPS());
    }
  }

  TaskNode<FFMPEGOutput, DeviceBufferPtr, void> ffmpeg_sw_output_node(
      &ffmpeg_output, "FFMPEGSoftwareOutput", stream_name);

  TaskNode<FFMPEGOutput, DvppEncoder::OutTy, void> ffmpeg_hw_output_node(
      &ffmpeg_output, "FFMPEGHardwareOutput", stream_name);

  DvppEncoder encoder;
  TaskNode<DvppEncoder, DeviceBufferPtr, void> encoder_node(
      &encoder, "DvppEncoder", stream_name);
  ThreadSafeQueueWithCapacity<DvppEncoder::OutTy> encoder_output_queue(
      queue_size);
  if (hardware_enc) {
    encoder.Init(cb_encoder_thread.GetPid(), height, width);
    encoder.SetOutputQueue(&encoder_output_queue);
    encoder_node.SetInputQueue(&yolov4_post_output_queue);
    ffmpeg_hw_output_node.SetInputQueue(&encoder_output_queue);
    ffmpeg_hw_output_node.Start(ctx);
    encoder_node.Start(ctx);
  } else {
    ffmpeg_sw_output_node.SetInputQueue(&yolov4_post_output_queue);
    ffmpeg_sw_output_node.Start(ctx);
  }

  if (camera_id < 0) {
    TaskNode<FFMPEGInput, void, void> ffmpeg_input_node(
        &ffmpeg_input, "FFMPEGInput", stream_name);
    ffmpeg_input.SetOutputQueue(&decoder_input_queue);
    SingalHandler::Register([&]() { ffmpeg_input.Stop(); });
    ffmpeg_input_node.Start(ctx);
    ffmpeg_input_node.Join();
    decoder.ShutDown();
  } else {
    TaskNode<CameraInput, void, void> camera_input_node(
        &camera_input, "CameraInput", stream_name);
    camera_input.SetOutputQueue(&resize_input_queue);
    SingalHandler::Register([&]() { camera_input.Stop(); });
    camera_input_node.Start(ctx);
    camera_input_node.Join();
  }

  resize_engine_node.Join();
  yolov4_model_node.Join();
  null_out_node.Join();
  post_process_node.Join();
  if (hardware_enc) {
    encoder_node.Join();
    encoder.Destory();
    ffmpeg_hw_output_node.Join();
  } else {
    ffmpeg_sw_output_node.Join();
  }

  if (camera_id < 0) {
    decoder_node.Join();
    decoder.Destory();
  }

  resize_engine.Destory();
  ffmpeg_output.Close();

  CHECK_ACL(aclrtUnSubscribeReport(cb_decoder_thread.GetPid(), decoder_stream));
  CHECK_ACL(aclrtUnSubscribeReport(cb_encoder_thread.GetPid(), encoder_stream));
  std::cout << "End of stream input: " << stream_name << std::endl;
}

REGSITER_STREAM(yolov4_demo, [](json config, int id) -> std::thread {
  return std::thread(Yolov4StreamThread, config, id);
});
