#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <fstream>
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
#include "dev_mem_pool.h"
#include "device_manager.h"
#include "signal_handler.h"
#include "stream_factory.h"
#include "task_node.h"
#include "yolox_stream.h"
#include "focus_op.h"

#define CHECK_PY_ERR(obj)                                                      \
  if (obj == NULL) {                                                           \
    PyErr_Print();                                                             \
    throw std::runtime_error("CHECK_PY_ERR");                                  \
  }

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

using namespace std::chrono_literals;

YoloXPreProcess::YoloXPreProcess(int w, int h, bool enable_neon)
    : width(w), height(h), enable_neon(enable_neon) {}


YoloXPreProcess::OutTy
YoloXPreProcess::Process(YoloXPreProcess::InTy bufferx2) {
#ifdef __ARM_NEON
  const bool has_neon = true;
#else
  const bool has_neon = false;
#endif
  if (has_neon && enable_neon && width % 16 == 0) {
    return ProcessWithNeon(bufferx2);
  }
  return ProcessWithoutNeon(bufferx2);
}

YoloXPreProcess::OutTy
YoloXPreProcess::ProcessWithNeon(YoloXPreProcess::InTy bufferx2) {
  APP_PROFILE(YoloXPreProcessWithNeon);
  uint8_t *host_buffer = (uint8_t *)std::get<1>(bufferx2)->GetHostPtr();

  int input_size = height * width * 3 * sizeof(float);

  void *buf = DevMemPool::AllocDevMem(input_size);

  auto dev_buffer_ptr = std::make_shared<DeviceBuffer>(
      buf, input_size, DeviceBuffer::DevMemDeleter());

  CvtFocusNEONFuse(height, width, (float *)dev_buffer_ptr->GetHostPtr(),
                   (uint8_t *)host_buffer, 1.0);

  dev_buffer_ptr->CopyToDevice();

  return {std::get<0>(bufferx2), dev_buffer_ptr};
}

YoloXPreProcess::OutTy
YoloXPreProcess::ProcessWithoutNeon(YoloXPreProcess::InTy bufferx2) {
  APP_PROFILE(YoloXPreProcessWithoutNeon);
  uint8_t *host_buffer = (uint8_t *)std::get<1>(bufferx2)->GetHostPtr();
  cv::Mat input_img(height * 3 / 2, width, CV_8UC1, host_buffer);
  cv::Mat output_img(height, width, CV_8UC3);
  {
    APP_PROFILE(COLOR_YUV2RGB_NV12);
    cv::cvtColor(input_img, output_img, cv::COLOR_YUV2RGB_NV12);
  }

  // YUV420SP2RGBNEON(height, width, output_img.data, input_img.data);
  // cv::imwrite("rgb.jpg", output_img);

  cv::Mat focus_format_img(height, width, CV_8UC3);

  FocusTransform<uint8_t>(height, width, (uint8_t *)focus_format_img.data,
                          (uint8_t *)output_img.data);
  // FocusTransformNEON(height, width, (uint8_t*)focus_format_img.data,
  // (uint8_t*)output_img.data);

  int input_size = height * width * 3 * sizeof(float);

  void *buf = DevMemPool::AllocDevMem(input_size);

  auto dev_buffer_ptr = std::make_shared<DeviceBuffer>(
      buf, input_size, DeviceBuffer::DevMemDeleter());

  cv::Mat output_float(height, width, CV_32FC3, dev_buffer_ptr->GetHostPtr());
  {
    APP_PROFILE(IMG_U8_TO_FLOAT);
    focus_format_img.convertTo(output_float, CV_32FC3);
  }
  // yolox does not need to scale
  {
    //APP_PROFILE(IMG_DIV_255);
    //output_float *= 0.00392156862745098;
  }

  // FocusTransform<float>(height, width, (float*)dev_buffer_ptr->GetHostPtr(),
  // (float*)output_float.data);

  dev_buffer_ptr->CopyToDevice();

  return {std::get<0>(bufferx2), dev_buffer_ptr};
}


YoloXModel::YoloXModel(const std::string &path, aclrtStream stream)
    : yolox_model(stream), model_stream(stream) {
  yolox_model.Init(path.c_str());
  std::cout << "YOLOX Model Info:" << std::endl;
  std::cout << yolox_model.ToString();
}

YoloXModel::OutTy YoloXModel::Process(YoloXModel::InTy bufferx2) {
  APP_PROFILE(YoloXModel);
  auto output_buffers = yolox_model.Infer({std::get<1>(bufferx2)});
  return {output_buffers, std::get<0>(bufferx2)};
}

YoloXPostProcess::YoloXPostProcess(int width, int height, int model_width,
                                     int model_height)
    : width(width), height(height) {
  h_ratio = height / (float)model_height;
  w_ratio = width / (float)model_width;
}

YoloXPostProcess::OutTy YoloXPostProcess::Process(InTy input) {
  APP_PROFILE(YoloXPostProcess);
  YoloXPyEnv &env = YoloXPyEnv::GetInstance();
  const auto &infer_results = std::get<0>(input);
  auto image_buffer = std::get<1>(input);

  float *pred = (float *)infer_results[0]->GetHostPtr();

  PyGILGuard py_gil_guard;

  npy_intp pred_dim[3] = {1, 8400, 85};
  const int pred_nd = 3;

  PyObject *pred_arr =
      PyArray_SimpleNewFromData(pred_nd, pred_dim, NPY_FLOAT32, pred);
  PyArray_CLEARFLAGS((PyArrayObject *)pred_arr, NPY_ARRAY_OWNDATA);

  PyObject *post_processing_fn = env.GetPostProcessFn();

  PyObject *post_processing_arg = Py_BuildValue("(O)", pred_arr);

  Py_XDECREF(pred_arr);

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

  Py_ssize_t bboxes_size = PyList_Size(bboxes);
  for (Py_ssize_t i = 0; i < bboxes_size; ++i) {
    PyObject *box = PyList_GetItem(bboxes, i);
    CHECK_PY_ERR(box);
    PyObject *py_x1 = PyList_GetItem(box, 0);
    CHECK_PY_ERR(py_x1);
    int x1 = std::round(PyFloat_AsDouble(py_x1) * w_ratio);
    PyObject *py_y1 = PyList_GetItem(box, 1);
    CHECK_PY_ERR(py_y1);
    int y1 = std::round(PyFloat_AsDouble(py_y1) * h_ratio);
    PyObject *py_x2 = PyList_GetItem(box, 2);
    CHECK_PY_ERR(py_x2);
    int x2 = std::round(PyFloat_AsDouble(py_x2) * w_ratio);
    PyObject *py_y2 = PyList_GetItem(box, 3);
    CHECK_PY_ERR(py_y2);
    int y2 = std::round(PyFloat_AsDouble(py_y2) * h_ratio);

    PyObject *py_label = PyList_GetItem(box, 5);
    CHECK_PY_ERR(py_label);
    int label = PyFloat_AsDouble(py_label);

    img.DrawRect(x1, y1, x2, y2, box_color, 3);
    img.DrawText(x1, y2, yolov3_label_zh_cn[label + 1], box_color);
  }

  Py_XDECREF(bboxes);

  return image_buffer;
}

void YoloXStreamThread(json config, int id) {
  std::string input_addr = config.at("src");
  std::string output_addr = config.at("dst");
  std::string model_path = config.at("yolox_model_path");

  int model_height = config.at("model_height");
  int model_width = config.at("model_width");
  bool is_null_output = output_addr == "null";
  bool hardware_enc = false;
  if (config.count("hw_encoder")) {
    hardware_enc = config.at("hw_encoder") && (!is_null_output);
  }

  bool enable_neon = true;
  if (config.count("enable_neon")) {
    enable_neon = config.at("enable_neon");
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

  if (camera_id < 0) {
    decoder_node.Start(ctx);
    decoder.Init(cb_decoder_thread.GetPid(), height, width,
                 ffmpeg_input.GetProfile());
    decoder.SetDeviceCtx(&ctx);
  }

  aclrtStream resize_stream;
  CHECK_ACL(aclrtCreateStream(&resize_stream));
  VPCResizeEngine resize_engine(resize_stream);
  resize_engine.Init(height, width, model_height, model_width);

  TaskNode<VPCResizeEngine, DeviceBufferPtr, buf_tup_t> resize_engine_node(
      &resize_engine, "VPCResizeEngine", stream_name);
  resize_engine_node.SetInputQueue(&resize_input_queue);

  ThreadSafeQueueWithCapacity<buf_tup_t> preprocess_input_queue(queue_size);
  resize_engine_node.SetOutputQueue(&preprocess_input_queue);
  resize_engine_node.Start(ctx);

  YoloXPreProcess yolox_preprocess(model_width, model_height, enable_neon);

  TaskNode<YoloXPreProcess, YoloXPreProcess::InTy, YoloXPreProcess::OutTy>
      yolox_preprocess_node(&yolox_preprocess, "YoloxPreProcess",
                             stream_name);

  ThreadSafeQueueWithCapacity<buf_tup_t> yolox_input_queue(queue_size);

  yolox_preprocess_node.SetInputQueue(&preprocess_input_queue);
  yolox_preprocess_node.SetOutputQueue(&yolox_input_queue);
  yolox_preprocess_node.Start(ctx);

  aclrtStream model_stream;
  CHECK_ACL(aclrtCreateStream(&model_stream));
  YoloXModel yolox_model(model_path, model_stream);

  TaskNode<YoloXModel, YoloXModel::InTy, YoloXModel::OutTy>
      yolox_model_node(&yolox_model, "YoloXModel", stream_name);


  yolox_model_node.SetInputQueue(&yolox_input_queue);
  

  ThreadSafeQueueWithCapacity<YoloXModel::OutTy> yolox_output_queue(
      queue_size);
  yolox_model_node.SetOutputQueue(&yolox_output_queue);
  yolox_model_node.Start(ctx);

  NullOutput<YoloXPostProcess::InTy> null_out;
  TaskNode<NullOutput<YoloXPostProcess::InTy>, YoloXPostProcess::InTy, void>
      null_out_node(&null_out, "NullOutput", stream_name);

  ThreadSafeQueueWithCapacity<YoloXModel::OutTy> dummy_output_queue(
      queue_size);

  YoloXPostProcess post_process(width, height, model_width, model_height);
  TaskNode<YoloXPostProcess, YoloXPostProcess::InTy, YoloXPostProcess::OutTy>
      post_process_node(&post_process, "YoloXPostProcess", stream_name);

  if (!is_null_output) {
    post_process_node.SetInputQueue(&yolox_output_queue);
    null_out_node.SetInputQueue(&dummy_output_queue);
  } else {
    post_process_node.SetInputQueue(&dummy_output_queue);
    null_out_node.SetInputQueue(&yolox_output_queue);
  }

  dummy_output_queue.ShutDown();

  null_out_node.Start(ctx);

  ThreadSafeQueueWithCapacity<YoloXPostProcess::OutTy>
      yolox_post_output_queue(queue_size);
  post_process_node.SetOutputQueue(&yolox_post_output_queue);
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
    encoder_node.SetInputQueue(&yolox_post_output_queue);
    ffmpeg_hw_output_node.SetInputQueue(&encoder_output_queue);
    ffmpeg_hw_output_node.Start(ctx);
    encoder_node.Start(ctx);
  } else {
    ffmpeg_sw_output_node.SetInputQueue(&yolox_post_output_queue);
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
  yolox_preprocess_node.Join();
  yolox_model_node.Join();
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

REGSITER_STREAM(yolox_demo, [](json config, int id) -> std::thread {
  return std::thread(YoloXStreamThread, config, id);
});
