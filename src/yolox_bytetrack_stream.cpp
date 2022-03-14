#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"

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
#include "focus_op.h"
#include "signal_handler.h"
#include "stream_factory.h"
#include "task_node.h"
#include "yolox_bytetrack_stream.h"
#include "yolox_stream.h"

// https://github.com/numpy/numpy/issues/11925
class ByteTrackPyEnv {
public:
  static ByteTrackPyEnv &GetInstance() {
    static ByteTrackPyEnv env;
    return env;
  }

  PyObject *GetInitFn() const { return tracker_init_fn; }

  PyObject *GetUpdateFn() const { return tracker_update_fn; }

  PyObject *GetPostProcessFn() const { return yolox_post_process_fn; }

private:
  ByteTrackPyEnv() {
    Py_Initialize();
    { // expansion of macro import_array
      if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError,
                        "numpy.core.multiarray failed to import");
      }
    }
    numpy = PyImport_ImportModule("numpy");
    byte_track_module = PyImport_ImportModule("bytetrack");
    if (byte_track_module == NULL) {
      PyErr_Print();
      return;
    }

    tracker_init_fn = PyObject_GetAttrString(byte_track_module, "init_tracker");
    if (tracker_init_fn == NULL) {
      PyErr_Print();
      return;
    }

    tracker_update_fn =
        PyObject_GetAttrString(byte_track_module, "update_tracker");
    if (tracker_update_fn == NULL) {
      PyErr_Print();
      return;
    }

    yolox_post_process_fn =
        PyObject_GetAttrString(byte_track_module, "yolox_post_process");
    if (yolox_post_process_fn == NULL) {
      PyErr_Print();
      return;
    }

    PyEval_InitThreads();
    _save = PyEval_SaveThread();
  }

  ~ByteTrackPyEnv() {
    PyEval_RestoreThread(_save);
    Py_FinalizeEx();
  }

  PyThreadState *_save;
  PyObject *numpy;
  PyObject *byte_track_module;
  PyObject *tracker_init_fn;
  PyObject *tracker_update_fn;
  PyObject *yolox_post_process_fn;
};

ByteTrackYoloXPostProcess::ByteTrackYoloXPostProcess(
    int image_height, int image_width, int box_num, int class_num,
    float nms_thr, float score_thr, ByteTrackImplType impl_type)
    : model_h(image_height), model_w(image_width), box_num(box_num),
      class_num(class_num), nms_thr(nms_thr), score_thr(score_thr),
      impl_type(impl_type) {
  if (impl_type == BYTETRACK_PY) {
    InitPY();
  } else {
    InitCPP();
  }
}

void ByteTrackYoloXPostProcess::InitCPP() {
  std::vector<int> strides{8, 16, 32};
  grid_strides = generate_grids_and_stride(model_w, model_h, strides);
}

void ByteTrackYoloXPostProcess::InitPY() {}

ByteTrackYoloXPostProcess::OutTy
ByteTrackYoloXPostProcess::Process(ByteTrackYoloXPostProcess::InTy input) {
  if (impl_type == BYTETRACK_PY) {
    return ProcessPY(input);
  }
  return ProcessCPP(input);
}

ByteTrackYoloXPostProcess::OutTy ByteTrackYoloXPostProcess::ProcessPY(
    ByteTrackYoloXPostProcess::InTy input_tup) {
  APP_PROFILE(ByteTrackYoloXPostProcessPY);

  const auto &infer_results = std::get<0>(input_tup);
  auto image_buffer = std::get<1>(input_tup);
  float *pred = (float *)infer_results[0]->GetHostPtr();

  ByteTrackPyEnv &env = ByteTrackPyEnv::GetInstance();
  PyGILGuard py_gil_guard;

  PyObject *post_process_fn = env.GetPostProcessFn();

  npy_intp pred_dim[3] = {1, box_num, class_num + 5};
  const int pred_nd = 3;

  PyObject *pred_arr =
      PyArray_SimpleNewFromData(pred_nd, pred_dim, NPY_FLOAT32, pred);
  PyArray_CLEARFLAGS((PyArrayObject *)pred_arr, NPY_ARRAY_OWNDATA);

  if (pred_arr == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to make pred_arr");
  }

  PyObject *post_process_arg = Py_BuildValue("(O,(i,i),f,f)", pred_arr, model_h,
                                             model_w, nms_thr, score_thr);

  if (post_process_arg == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to make post_process_arg args");
  }

  PyObject *dets = PyObject_Call(post_process_fn, post_process_arg, NULL);

  if (dets == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to call yolox post process");
  }

  Py_XDECREF(pred_arr);
  Py_XDECREF(post_process_arg);
  return {new ByteTrackDetectInfo(dets), image_buffer};
}

ByteTrackYoloXPostProcess::OutTy ByteTrackYoloXPostProcess::ProcessCPP(
    ByteTrackYoloXPostProcess::InTy input_tup) {
  APP_PROFILE(ByteTrackYoloXPostProcessCPP);

  const auto &infer_results = std::get<0>(input_tup);
  auto image_buffer = std::get<1>(input_tup);
  float *pred = (float *)infer_results[0]->GetHostPtr();

  return {new ByteTrackDetectInfo(detect_yolox(model_h, model_w, box_num,
                                               class_num, nms_thr, score_thr,
                                               pred, grid_strides)),
          image_buffer};
}

ByteTrackTracker::ByteTrackTracker(double framerate, int raw_img_height,
                                   int raw_img_width, int image_height,
                                   int image_width, int box_num, int class_num,
                                   float nms_thr, float score_thr,
                                   ByteTrackImplType impl_type)
    : cpp_tracker(std::round(framerate), 100), framerate(framerate),
      raw_img_h(raw_img_height), raw_img_w(raw_img_width),
      model_h(image_height), model_w(image_width), box_num(box_num),
      class_num(class_num),
      h_ratio((float)raw_img_height / (float)image_height),
      w_ratio((float)raw_img_width / (float)image_width), nms_thr(nms_thr),
      score_thr(score_thr), impl_type(impl_type) {
  if (impl_type == BYTETRACK_PY) {
    InitPY();
  } else {
    InitCPP();
  }

  colors.emplace_back(76, 84, 255);   // Red
  colors.emplace_back(149, 43, 21);   // Lime
  colors.emplace_back(29, 255, 107);  // Blue
  colors.emplace_back(225, 0, 148);   // Yellow
  colors.emplace_back(178, 171, 0);   // Cyan
  colors.emplace_back(105, 212, 234); // Magenta
}

void ByteTrackTracker::InitPY() {
  ByteTrackPyEnv &env = ByteTrackPyEnv::GetInstance();
  PyGILGuard py_gil_guard;

  PyObject *init_fn = env.GetInitFn();

  PyObject *init_arg = Py_BuildValue("(d,f,f)", framerate, nms_thr, score_thr);

  bytetracker = PyObject_Call(init_fn, init_arg, NULL);

  if (bytetracker == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to init bytetracker");
  }

  Py_XDECREF(init_arg);
}

void ByteTrackTracker::InitCPP() {}

ByteTrackTracker::OutTy
ByteTrackTracker::Process(ByteTrackTracker::InTy input) {
  if (impl_type == BYTETRACK_PY) {
    return ProcessPY(input);
  }
  return ProcessCPP(input);
}

ByteTrackTracker::OutTy
ByteTrackTracker::ProcessPY(ByteTrackTracker::InTy input_tup) {
  APP_PROFILE(ByteTrackTrackerPY);

  ByteTrackDetectInfo *detect_input = std::get<0>(input_tup);

  PyObject *dets = detect_input->dets;
  auto image_buffer = std::get<1>(input_tup);

  ByteTrackPyEnv &env = ByteTrackPyEnv::GetInstance();

  PyGILGuard py_gil_guard;

  PyObject *tracker_upd_fn = env.GetUpdateFn();

  PyObject *tracker_upd_arg =
      Py_BuildValue("(O,O,(i,i),(i,i))", bytetracker, dets, model_h, model_w,
                    model_h, model_w);

  if (tracker_upd_arg == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to make tracker_upd_arg args");
  }

  // PyObject_Print(tracker_upd_arg, stdout, 0);

  PyObject *update_result =
      PyObject_Call(tracker_upd_fn, tracker_upd_arg, NULL);

  Py_XDECREF(dets);
  Py_XDECREF(tracker_upd_arg);

  if (update_result == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to update tracker");
  }

  YUV420SPImage img((uint8_t *)image_buffer->GetHostPtr(), raw_img_h,
                    raw_img_w);

  PyObject *tlwh_list = PyTuple_GetItem(update_result, 0);
  CHECK_PY_ERR(tlwh_list);

  PyObject *ids_list = PyTuple_GetItem(update_result, 1);
  CHECK_PY_ERR(ids_list);

  PyObject *score_list = PyTuple_GetItem(update_result, 2);
  CHECK_PY_ERR(score_list);

  Py_ssize_t object_num = PyList_Size(tlwh_list);

  for (Py_ssize_t i = 0; i < object_num; ++i) {
    PyObject *tlwh = PyList_GetItem(tlwh_list, i);
    CHECK_PY_ERR(tlwh);
    PyObject *py_x1 = PyList_GetItem(tlwh, 0);
    CHECK_PY_ERR(py_x1);
    float fp_x1 = PyFloat_AsDouble(py_x1) * w_ratio;
    int x1 = std::round(fp_x1);
    PyObject *py_y1 = PyList_GetItem(tlwh, 1);
    CHECK_PY_ERR(py_y1);
    float fp_y1 = PyFloat_AsDouble(py_y1) * h_ratio;
    int y1 = std::round(fp_y1);

    PyObject *py_w = PyList_GetItem(tlwh, 2);
    CHECK_PY_ERR(py_w);
    float fp_w = PyFloat_AsDouble(py_w) * w_ratio;
    int x2 = std::round(fp_x1 + fp_w);

    PyObject *py_h = PyList_GetItem(tlwh, 3);
    CHECK_PY_ERR(py_h);
    float fp_h = PyFloat_AsDouble(py_h) * h_ratio;
    int y2 = std::round(fp_y1 + fp_h);

    PyObject *py_track_id = PyList_GetItem(ids_list, i);
    CHECK_PY_ERR(py_track_id);
    int track_id = PyLong_AsLong(py_track_id);

    PyObject *py_score = PyList_GetItem(score_list, i);
    CHECK_PY_ERR(py_score);
    float score = PyFloat_AsDouble(py_score);

    auto &color = colors[track_id % colors.size()];
    img.DrawRect(x1, y1, x2, y2, color, 3);
    img.DrawText(x1, y2, std::to_string(track_id), color);
  }

  Py_XDECREF(update_result);

  delete detect_input;

  return image_buffer;
}

ByteTrackTracker::OutTy
ByteTrackTracker::ProcessCPP(ByteTrackTracker::InTy input_tup) {
  APP_PROFILE(ByteTrackTrackerCPP);

  ByteTrackDetectInfo *detect_input = std::get<0>(input_tup);
  auto image_buffer = std::get<1>(input_tup);

  std::vector<STrack> output_stracks =
      cpp_tracker.update(detect_input->objects);

  int output_size = output_stracks.size();

  YUV420SPImage img((uint8_t *)image_buffer->GetHostPtr(), raw_img_h,
                    raw_img_w);

  for (int i = 0; i < output_size; ++i) {
    const auto &tlwh = output_stracks[i].tlwh;
    bool vertical = tlwh[2] / tlwh[3] > 1.6;
    if (tlwh[2] * tlwh[3] > 20 && !vertical) {
      int x1 = std::round(tlwh[0] * w_ratio);
      int y1 = std::round(tlwh[1] * h_ratio);
      int x2 = std::round((tlwh[0] + tlwh[2]) * w_ratio);
      int y2 = std::round((tlwh[1] + tlwh[3]) * h_ratio);

      int track_id = output_stracks[i].track_id;

      auto &color = colors[track_id % colors.size()];

      img.DrawRect(x1, y1, x2, y2, color, 3);
      img.DrawText(x1, y2, std::to_string(track_id), color);
    }
  }

  delete detect_input;
  return image_buffer;
}

ByteTrackPreProcess::ByteTrackPreProcess(int model_height, int model_width,
                                         const std::vector<float> &vec_mean,
                                         const std::vector<float> &vec_std) {
  height = model_height / 2;
  width = model_width / 2;

  mean = vec_mean;
  // div std -> mul std
  for (float s : vec_std) {
    std.push_back(1 / s);
  }
}

ByteTrackPreProcess::OutTy
ByteTrackPreProcess::Process(ByteTrackPreProcess::InTy bufferx2) {
  APP_PROFILE(ByteTrackPreProcess);
  auto &buffer_ptr = std::get<1>(bufferx2);
  float *host_buffer = (float *)buffer_ptr->GetHostPtr();

  int focus_channel_stride = height * width;

  for (int chn_i = 0; chn_i < 12; ++chn_i) {
    float chn_mean = mean[chn_i % 3];
    float chn_std = std[chn_i % 3];
#pragma omp for simd
    for (int i = 0; i < focus_channel_stride; ++i) {
      host_buffer[i] = (host_buffer[i] - chn_mean) * chn_std;
    }
    host_buffer += focus_channel_stride;
  }

  buffer_ptr->CopyToDevice();

  return bufferx2;
}

void YoloXByteTrackStreamThread(json config, int id) {
  std::string input_addr = config.at("src");
  std::string output_addr = config.at("dst");
  std::string model_path = config.at("bytetrack_model_path");

  int model_height = config.at("model_height");
  int model_width = config.at("model_width");

  int box_num = config.value<int>("model_box_num", 13566);

  int class_num = config.value<int>("model_class_num", 1);

  float nms_thr = config.value<float>("nms_thr", 0.1);
  float score_thr = config.value<float>("score_thr", 0.7);

  bool is_null_output = output_addr == "null";
  bool hardware_enc = false;
  if (config.count("hw_encoder")) {
    hardware_enc = config.at("hw_encoder") && (!is_null_output);
  }

  bool enable_neon = config.value<bool>("enable_neon", true);

  std::vector<float> vec_mean =
      config.value<std::vector<float>>("mean", {0, 0, 0});
  std::vector<float> vec_std =
      config.value<std::vector<float>>("std", {1, 1, 1});
  float scale = config.value<float>("scale", 1.0);

  std::string str_impl_mode = config.value<std::string>("tracker_impl", "cpp");
  ByteTrackImplType impl_type =
      str_impl_mode == "cpp" ? BYTETRACK_CPP : BYTETRACK_PY;

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

  YoloXPreProcess yolox_preprocess(model_width, model_height, enable_neon,
                                   scale);

  TaskNode<YoloXPreProcess, YoloXPreProcess::InTy, YoloXPreProcess::OutTy>
      yolox_preprocess_node(&yolox_preprocess, "YoloxPreProcess", stream_name);

  ThreadSafeQueueWithCapacity<buf_tup_t> bytetrack_preprocess_input_queue(
      queue_size);
  ThreadSafeQueueWithCapacity<buf_tup_t> yolox_input_queue(queue_size);

  yolox_preprocess_node.SetInputQueue(&preprocess_input_queue);
  yolox_preprocess_node.SetOutputQueue(&bytetrack_preprocess_input_queue);
  yolox_preprocess_node.Start(ctx);

  ByteTrackPreProcess bytetrack_preprocess(model_height, model_width, vec_mean,
                                           vec_std);

  TaskNode<ByteTrackPreProcess, ByteTrackPreProcess::InTy,
           ByteTrackPreProcess::OutTy>
      bytetrack_preprocess_node(&bytetrack_preprocess, "ByteTrackPreProcess",
                                stream_name);

  bytetrack_preprocess_node.SetInputQueue(&bytetrack_preprocess_input_queue);
  bytetrack_preprocess_node.SetOutputQueue(&yolox_input_queue);
  bytetrack_preprocess_node.Start(ctx);

  aclrtStream model_stream;
  CHECK_ACL(aclrtCreateStream(&model_stream));
  YoloXModel yolox_model(model_path, model_stream);

  TaskNode<YoloXModel, YoloXModel::InTy, YoloXModel::OutTy> yolox_model_node(
      &yolox_model, "YoloXModel", stream_name);

  yolox_model_node.SetInputQueue(&yolox_input_queue);

  ThreadSafeQueueWithCapacity<YoloXModel::OutTy> yolox_output_queue(queue_size);
  yolox_model_node.SetOutputQueue(&yolox_output_queue);
  yolox_model_node.Start(ctx);

  NullOutput<ByteTrackYoloXPostProcess::InTy> null_out;
  TaskNode<NullOutput<ByteTrackYoloXPostProcess::InTy>,
           ByteTrackYoloXPostProcess::InTy, void>
      null_out_node(&null_out, "NullOutput", stream_name);

  ThreadSafeQueueWithCapacity<YoloXModel::OutTy> dummy_output_queue(queue_size);

  double tracker_framerate = 0;

  if (camera_id < 0) {
    tracker_framerate = av_q2d(ffmpeg_input.GetFramerate());
  } else {
    tracker_framerate = camera_input.GetFPS();
  }

  ByteTrackYoloXPostProcess yolox_post_process(model_height, model_width,
                                               box_num, class_num, nms_thr,
                                               score_thr, impl_type);
  TaskNode<ByteTrackYoloXPostProcess, ByteTrackYoloXPostProcess::InTy,
           ByteTrackYoloXPostProcess::OutTy>
      yolox_post_process_node(&yolox_post_process, "ByteTrackYoloXPostProcess",
                              stream_name);

  ThreadSafeQueueWithCapacity<ByteTrackYoloXPostProcess::OutTy>
      tracker_input_queue(queue_size);

  ByteTrackTracker bytetracker(tracker_framerate, height, width, model_height,
                               model_width, box_num, class_num, nms_thr,
                               score_thr, impl_type);
  TaskNode<ByteTrackTracker, ByteTrackTracker::InTy, ByteTrackTracker::OutTy>
      bytetracker_node(&bytetracker, "ByteTrackTracker", stream_name);

  if (!is_null_output) {
    yolox_post_process_node.SetInputQueue(&yolox_output_queue);
    null_out_node.SetInputQueue(&dummy_output_queue);
  } else {
    yolox_post_process_node.SetInputQueue(&dummy_output_queue);
    null_out_node.SetInputQueue(&yolox_output_queue);
  }

  dummy_output_queue.ShutDown();

  null_out_node.Start(ctx);

  bytetracker_node.SetInputQueue(&tracker_input_queue);
  yolox_post_process_node.SetOutputQueue(&tracker_input_queue);
  yolox_post_process_node.Start(ctx);

  ThreadSafeQueueWithCapacity<ByteTrackTracker::OutTy> tracker_output_queue(
      queue_size);
  bytetracker_node.SetOutputQueue(&tracker_output_queue);
  bytetracker_node.Start(ctx);

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
    encoder_node.SetInputQueue(&tracker_output_queue);
    ffmpeg_hw_output_node.SetInputQueue(&encoder_output_queue);
    ffmpeg_hw_output_node.Start(ctx);
    encoder_node.Start(ctx);
  } else {
    ffmpeg_sw_output_node.SetInputQueue(&tracker_output_queue);
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
  bytetrack_preprocess_node.Join();
  yolox_model_node.Join();
  null_out_node.Join();
  yolox_post_process_node.Join();
  bytetracker_node.Join();
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

REGSITER_STREAM(yolox_bytetrack_demo, [](json config, int id) -> std::thread {
  return std::thread(YoloXByteTrackStreamThread, config, id);
});
