#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"

#include "acl_cb_thread.h"
#include "acl_model.h"
#include "app_profiler.h"
#include "camera_input.h"
#include "deep_sort_stream.h"
#include "dev_mem_pool.h"
#include "device_manager.h"
#include "drawing.h"
#include "dvpp_decoder.h"
#include "dvpp_encoder.h"
#include "ffmpeg_input.h"
#include "ffmpeg_output.h"
#include "jpeg_encode.h"
#include "signal_handler.h"
#include "stream_factory.h"
#include "task_node.h"
#include "util.h"
#include "vpc_batch_crop.h"
#include "vpc_resize.h"

#define CHECK_PY_ERR(obj)                                                      \
  if (obj == NULL) {                                                           \
    PyErr_Print();                                                             \
    throw std::runtime_error("CHECK_PY_ERR");                                  \
  }

// https://github.com/numpy/numpy/issues/11925
class PyEnv {
public:
  static PyEnv &GetInstance() {
    static PyEnv env;
    return env;
  }

  PyObject *GetInitFn() const { return init_fn; }

  PyObject *GetMakeDetectionsFn() const { return make_detection_fn; }

private:
  PyEnv() {
    Py_Initialize();
    { // expansion of macro import_array
      if (_import_array() < 0) {
        PyErr_Print();
        PyErr_SetString(PyExc_ImportError,
                        "numpy.core.multiarray failed to import");
      }
    }
    numpy = PyImport_ImportModule("numpy");
    deepsort_module = PyImport_ImportModule("acl_deepsort_app");
    if (deepsort_module == NULL) {
      PyErr_Print();
      return;
    }

    init_fn = PyObject_GetAttrString(deepsort_module, "init_tracker");
    if (init_fn == NULL) {
      PyErr_Print();
      return;
    }

    make_detection_fn =
        PyObject_GetAttrString(deepsort_module, "make_detections");
    if (make_detection_fn == NULL) {
      PyErr_Print();
      return;
    }
    PyEval_InitThreads();
    _save = PyEval_SaveThread();
  }

  ~PyEnv() {
    PyEval_RestoreThread(_save);
    Py_FinalizeEx();
  }

  PyThreadState *_save;
  PyObject *numpy;
  PyObject *deepsort_module;
  PyObject *init_fn;
  PyObject *make_detection_fn;
};

static const int feature_h = 128;
static const int feature_w = 64;
static const int feature_pic_size = feature_w * feature_h * 3 / 2;
static const int deepsort_batch_size = 16;
static int post_nms_num = 1024;
static const int output_feature_size = 128 * sizeof(float);
const static int yolov3_model_size = 416;

DeepSortCropProcess::DeepSortCropProcess(aclrtStream stream, int width,
                                         int height)
    : crop_stream(stream), crop_engine(stream), width(width), height(height) {
  crop_engine.Init(height, width);
  h_ratio = height / (float)yolov3_model_size;
  w_ratio = width / (float)yolov3_model_size;
}

DeepSortCropProcess::OutTy DeepSortCropProcess::Process(InTy input) {
  APP_PROFILE(DeepSortCropProcess);
  const auto &infer_results = std::get<0>(input);
  auto &image_buffer = std::get<1>(input);

  void *raw_buffer = image_buffer->GetDevicePtr();

  float *box_info = (float *)infer_results[0]->GetHostPtr();
  int32_t box_out_num = ((int32_t *)infer_results[1]->GetHostPtr())[0];

  std::vector<int> vec_x1, vec_x2, vec_y1, vec_y2;
  std::vector<int> vec_dst_h, vec_dst_w;
  vec_x1.reserve(box_out_num);
  vec_x2.reserve(box_out_num);
  vec_y1.reserve(box_out_num);
  vec_y2.reserve(box_out_num);
  vec_dst_h.reserve(box_out_num);
  vec_dst_w.reserve(box_out_num);

  OutTy output;
  std::get<1>(output) = image_buffer;
  BoxInfo &output_box = std::get<0>(output);

  std::vector<int32_t> &boxes = std::get<0>(output_box); // format (x,y,w,h)
  boxes.reserve(box_out_num);
  std::vector<float> &scores = std::get<1>(output_box);
  scores.reserve(box_out_num);

  for (int32_t box_i = 0; box_i < box_out_num; ++box_i) {
    float x1 = box_info[box_out_num * 0 + box_i] * w_ratio;
    float y1 = box_info[box_out_num * 1 + box_i] * h_ratio;
    float x2 = box_info[box_out_num * 2 + box_i] * w_ratio;
    float y2 = box_info[box_out_num * 3 + box_i] * h_ratio;
    float score = box_info[box_out_num * 4 + box_i];
    float label = box_info[box_out_num * 5 + box_i];
    int ilabel = int(label) + 1;

    int even_x1 = ((int)(x1 / 2)) * 2;
    int odd_x2 = ((int)(x2 / 2)) * 2 - 1;
    if (odd_x2 < 0) {
      odd_x2 = 1;
    }
    int even_y1 = ((int)(y1 / 2)) * 2;
    int odd_y2 = ((int)(y2 / 2)) * 2 - 1;
    if (odd_y2 < 0) {
      odd_y2 = 1;
    }

    int x_width = odd_x2 - even_x1 + 1;
    if (x_width < 10) {
      // ignore small box
      continue;
    }

    int y_width = odd_y2 - even_y1 + 1;
    if (y_width < 10) {
      continue;
    }

#define __CHECK_IN_RANGE(value, bound)                                         \
  if (value < 0) {                                                             \
    continue;                                                                  \
  } else if (value > bound) {                                                  \
    continue;                                                                  \
  }

    __CHECK_IN_RANGE(even_x1, width);
    __CHECK_IN_RANGE(odd_x2, width);
    __CHECK_IN_RANGE(even_y1, height);
    __CHECK_IN_RANGE(odd_y2, height);

    boxes.push_back(even_x1);
    boxes.push_back(even_y1);
    boxes.push_back(x_width);
    boxes.push_back(y_width);
    scores.push_back(score);
    vec_x1.push_back(even_x1);
    vec_x2.push_back(odd_x2);
    vec_y1.push_back(even_y1);
    vec_y2.push_back(odd_y2);
    vec_dst_h.push_back(feature_h);
    vec_dst_w.push_back(feature_w);
  }

  int valid_box_count = vec_x1.size();
  const int batch_feature_size = feature_pic_size * deepsort_batch_size;

  ACLModel::DevBufferVec &features = std::get<2>(output);

  for (int i = 0; i < valid_box_count; i += deepsort_batch_size) {
    int batch_size = std::min(deepsort_batch_size, valid_box_count - i);
    void *device_img_info = DevMemPool::AllocDevMem(batch_feature_size);

    DeviceBufferPtr feature_batch = std::make_shared<DeviceBuffer>(
        device_img_info, batch_feature_size, DeviceBuffer::DevMemDeleter());

    std::vector<uint8_t *> vec_dst_addr;
    vec_dst_addr.reserve(batch_size);

    for (int batch_i = 0; batch_i < batch_size; ++batch_i) {
      vec_dst_addr.push_back((uint8_t *)device_img_info +
                             batch_i * feature_pic_size);
    }

    crop_engine.Crop((uint8_t *)raw_buffer, vec_x1.data() + i,
                     vec_y1.data() + i, vec_x2.data() + i, vec_y2.data() + i,
                     vec_dst_addr.data(), vec_dst_h.data() + i,
                     vec_dst_w.data() + i, batch_size);
    features.push_back(feature_batch);
  }

  return output;
}

DeepSortModel::DeepSortModel(const std::string &path, aclrtStream stream)
    : deepsort_model(stream), model_stream(stream) {
  deepsort_model.Init(path.c_str());
  std::cout << "Model Info:" << std::endl;
  std::cout << deepsort_model.ToString();
}

DeepSortModel::OutTy DeepSortModel::Process(InTy input_tup) {
  APP_PROFILE(DeepSortModel);
  BoxInfo &box_info = std::get<0>(input_tup);
  auto &image_buffer = std::get<1>(input_tup);
  auto &feature_vec = std::get<2>(input_tup);

  std::vector<int32_t> &boxes = std::get<0>(box_info); // format (x,y,w,h)
  std::vector<float> &scores = std::get<1>(box_info);

  int box_num = scores.size();

  // delete by tracker
  uint8_t *output_buffer = new uint8_t[box_num * output_feature_size];

  int batch_num = feature_vec.size();
  int output_i = 0;

  for (int i = 0; i < batch_num; ++i) {
    auto output_buffers = deepsort_model.Infer({feature_vec[i]});
    int actual_batch_size =
        std::min(deepsort_batch_size, box_num - i * deepsort_batch_size);
    for (int batch_i = 0; batch_i < actual_batch_size; ++batch_i) {
      memcpy(output_buffer + output_i * output_feature_size,
             output_buffers[0]->GetHostPtr(), output_feature_size);
      ++output_i;
    }
  }

  OutTy output;
  std::get<0>(output) = box_info;
  std::get<1>(output) = image_buffer;
  std::get<2>(output) = output_buffer;
  return output;
}

DeepSortTracker::DeepSortTracker() {
  PyEnv &env = PyEnv::GetInstance();
  PyGILGuard py_gil_guard;
  PyObject *init_fn = env.GetInitFn();
  PyObject *init_arg = PyTuple_New(0);
  tracker_ctx = PyObject_Call(init_fn, init_arg, NULL);

  if (tracker_ctx == NULL) {
    PyErr_Print();
    return;
  }

  update_tracker_fn = PyObject_GetAttrString(tracker_ctx, "update");

  if (update_tracker_fn == NULL) {
    PyErr_Print();
    return;
  }

  Py_XDECREF(init_arg);
  query_tracking_fn =
      PyObject_GetAttrString(tracker_ctx, "query_tracking_result");
  if (query_tracking_fn == NULL) {
    PyErr_Print();
    return;
  }
}

DeepSortTracker::OutTy DeepSortTracker::Process(InTy input_tup) {
  APP_PROFILE(DeepSortTracker);

  PyGILGuard py_gil_guard;

  BoxInfo &box_info = std::get<0>(input_tup);
  auto &image_buffer = std::get<1>(input_tup);
  uint8_t *feature_buffer = std::get<2>(input_tup);
  std::vector<int32_t> &boxes = std::get<0>(box_info); // format (x,y,w,h)
  std::vector<float> &scores = std::get<1>(box_info);

  TrackResultVec trackings;

  PyEnv &env = PyEnv::GetInstance();

  int feature_num = scores.size();
  npy_intp boxes_dim[2] = {feature_num, 4};
  const int boxes_nd = 2;

  PyObject *boxes_arr =
      PyArray_SimpleNewFromData(boxes_nd, boxes_dim, NPY_INT32, boxes.data());
  PyArray_CLEARFLAGS((PyArrayObject *)boxes_arr, NPY_ARRAY_OWNDATA);

  npy_intp scores_dim[1] = {feature_num};
  const int scores_nd = 1;

  PyObject *scores_arr = PyArray_SimpleNewFromData(scores_nd, scores_dim,
                                                   NPY_FLOAT32, scores.data());
  PyArray_CLEARFLAGS((PyArrayObject *)scores_arr, NPY_ARRAY_OWNDATA);

  npy_intp deepsort_dim[2] = {feature_num, 128};
  const int deepsort_nd = 2;
  PyObject *feature_arr = PyArray_SimpleNewFromData(
      deepsort_nd, deepsort_dim, NPY_FLOAT32, feature_buffer);
  PyArray_CLEARFLAGS((PyArrayObject *)feature_arr, NPY_ARRAY_OWNDATA);

  PyObject *detection_fn = env.GetMakeDetectionsFn();

  PyObject *detection_arg =
      Py_BuildValue("(O,O,O)", boxes_arr, scores_arr, feature_arr);
  Py_XDECREF(boxes_arr);
  Py_XDECREF(scores_arr);
  Py_XDECREF(feature_arr);

  if (detection_arg == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to make detection args");
  }

  PyObject *detections = PyObject_Call(detection_fn, detection_arg, NULL);
  delete[] feature_buffer;
  if (detections == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to make detections");
  }

  Py_XDECREF(detection_arg);

  PyObject *update_arg = Py_BuildValue("(O)", detections);

  if (update_arg == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to make update arg");
  }

  PyObject *upd_result = PyObject_Call(update_tracker_fn, update_arg, NULL);

  Py_XDECREF(update_arg);

  if (upd_result == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to update");
  }

  Py_XDECREF(detections);
  Py_XDECREF(upd_result);

  PyObject *query_arg = PyTuple_New(0);
  if (query_arg == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to make track arg");
  }

  // List[List[tracking_id, x1, y1 ,x2 ,y2]]
  PyObject *tracking_result = PyObject_Call(query_tracking_fn, query_arg, NULL);
  if (tracking_result == NULL) {
    PyErr_Print();
    throw std::runtime_error("Failed to make track");
  }

  Py_XDECREF(query_arg);

  Py_ssize_t track_size = PyList_Size(tracking_result);
  for (Py_ssize_t i = 0; i < track_size; ++i) {
    PyObject *track = PyList_GetItem(tracking_result, i);
    CHECK_PY_ERR(track);
    PyObject *py_track_id = PyList_GetItem(track, 0);
    CHECK_PY_ERR(py_track_id);
    int track_id = PyLong_AsLong(py_track_id);
    PyObject *py_x1 = PyList_GetItem(track, 1);
    CHECK_PY_ERR(py_x1);
    int x1 = std::round(PyFloat_AsDouble(py_x1));
    PyObject *py_y1 = PyList_GetItem(track, 2);
    CHECK_PY_ERR(py_y1);
    int y1 = std::round(PyFloat_AsDouble(py_y1));
    PyObject *py_x2 = PyList_GetItem(track, 3);
    CHECK_PY_ERR(py_x2);
    int x2 = std::round(PyFloat_AsDouble(py_x2));
    PyObject *py_y2 = PyList_GetItem(track, 4);
    CHECK_PY_ERR(py_y2);
    int y2 = std::round(PyFloat_AsDouble(py_y2));
    trackings.push_back({track_id, x1, y1, x2, y2});
  }

  Py_XDECREF(tracking_result);
  return std::make_tuple(trackings, image_buffer);
}

DeepSortPostProcess::DeepSortPostProcess(int width, int height)
    : width(width), height(height) {
  colors.emplace_back(76, 84, 255);   // Red
  colors.emplace_back(149, 43, 21);   // Lime
  colors.emplace_back(29, 255, 107);  // Blue
  colors.emplace_back(225, 0, 148);   // Yellow
  colors.emplace_back(178, 171, 0);   // Cyan
  colors.emplace_back(105, 212, 234); // Magenta
}

DeepSortPostProcess::OutTy DeepSortPostProcess::Process(InTy input) {
  APP_PROFILE(DeepSortPostProcess);

  auto &trackings = std::get<0>(input);
  auto &image_buffer = std::get<1>(input);

  YUV420SPImage img((uint8_t *)image_buffer->GetHostPtr(), height, width);

  for (const auto &track : trackings) {
    auto &color = colors[track.track_id % colors.size()];
    img.DrawRect(track.x1, track.y1, track.x2, track.y2, color, 3);
    img.DrawText(track.x1, track.y2, std::to_string(track.track_id), color);
  }
  return image_buffer;
}

void DeepSortStreamThread(json config, int id) {
  std::string input_addr = config.at("src");
  std::string output_addr = config.at("dst");
  std::string yolov3_model_path = config.at("yolov3_model_path");
  std::string deepsort_model_path = config.at("deepsort_model_path");
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
  resize_engine.Init(height, width, yolov3_model_size, yolov3_model_size);

  TaskNode<VPCResizeEngine, DeviceBufferPtr, buf_tup_t> resize_engine_node(
      &resize_engine, "VPCResizeEngine", stream_name);
  resize_engine_node.SetInputQueue(&resize_input_queue);

  ThreadSafeQueueWithCapacity<buf_tup_t> yolov3_input_queue(queue_size);
  resize_engine_node.SetOutputQueue(&yolov3_input_queue);
  resize_engine_node.Start(ctx);

  aclrtStream yolov3_model_stream;
  CHECK_ACL(aclrtCreateStream(&yolov3_model_stream));
  Yolov3Model yolov3_model(yolov3_model_path, yolov3_model_stream);

  TaskNode<Yolov3Model, Yolov3Model::InTy, Yolov3Model::OutTy>
      yolov3_model_node(&yolov3_model, "Yolov3Model", stream_name);

  yolov3_model_node.SetInputQueue(&yolov3_input_queue);

  ThreadSafeQueueWithCapacity<Yolov3Model::OutTy> crop_input_queue(queue_size);
  yolov3_model_node.SetOutputQueue(&crop_input_queue);
  yolov3_model_node.Start(ctx);

  aclrtStream deepsort_crop_stream;
  CHECK_ACL(aclrtCreateStream(&deepsort_crop_stream));
  DeepSortCropProcess deepsort_crop(deepsort_crop_stream, width, height);
  TaskNode<DeepSortCropProcess, DeepSortCropProcess::InTy,
           DeepSortCropProcess::OutTy>
      deepsort_crop_node(&deepsort_crop, "DeepSortCrop", stream_name);
  deepsort_crop_node.SetInputQueue(&crop_input_queue);
  ThreadSafeQueueWithCapacity<DeepSortCropProcess::OutTy> tracker_input_queue(
      queue_size);
  deepsort_crop_node.SetOutputQueue(&tracker_input_queue);
  deepsort_crop_node.Start(ctx);

  aclrtStream deepsort_stream;
  CHECK_ACL(aclrtCreateStream(&deepsort_stream));
  DeepSortModel deepsort_model(deepsort_model_path, deepsort_stream);
  TaskNode<DeepSortModel, DeepSortModel::InTy, DeepSortModel::OutTy>
      deepsort_model_node(&deepsort_model, "DeepSortModel", stream_name);
  deepsort_model_node.SetInputQueue(&tracker_input_queue);
  ThreadSafeQueueWithCapacity<DeepSortModel::OutTy> deepsort_tracker_queue(
      queue_size);
  deepsort_model_node.SetOutputQueue(&deepsort_tracker_queue);
  deepsort_model_node.Start(ctx);

  DeepSortTracker tracker;
  TaskNode<DeepSortTracker, DeepSortTracker::InTy, DeepSortTracker::OutTy>
      tracker_node(&tracker, "DeepSortTracker", stream_name);
  tracker_node.SetInputQueue(&deepsort_tracker_queue);
  ThreadSafeQueueWithCapacity<DeepSortTracker::OutTy> tracker_output_queue(
      queue_size);
  tracker_node.SetOutputQueue(&tracker_output_queue);
  tracker_node.Start(ctx);

  NullOutput<DeepSortPostProcess::InTy> null_out;
  TaskNode<NullOutput<DeepSortPostProcess::InTy>, DeepSortPostProcess::InTy,
           void>
      null_out_node(&null_out, "NullOutput", stream_name);

  ThreadSafeQueueWithCapacity<DeepSortTracker::OutTy> dummy_output_queue(
      queue_size);

  DeepSortPostProcess deepsort_pp(width, height);
  TaskNode<DeepSortPostProcess, DeepSortPostProcess::InTy,
           DeepSortPostProcess::OutTy>
      deepsort_pp_node(&deepsort_pp, "DeepSortPostProcess", stream_name);

  if (!is_null_output) {
    deepsort_pp_node.SetInputQueue(&tracker_output_queue);
    null_out_node.SetInputQueue(&dummy_output_queue);
  } else {
    deepsort_pp_node.SetInputQueue(&dummy_output_queue);
    null_out_node.SetInputQueue(&tracker_output_queue);
  }

  dummy_output_queue.ShutDown();

  null_out_node.Start(ctx);

  ThreadSafeQueueWithCapacity<DeepSortPostProcess::OutTy> deepsort_output_queue(
      queue_size);
  deepsort_pp_node.SetOutputQueue(&deepsort_output_queue);
  deepsort_pp_node.Start(ctx);

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
    encoder_node.SetInputQueue(&deepsort_output_queue);
    ffmpeg_hw_output_node.SetInputQueue(&encoder_output_queue);
    ffmpeg_hw_output_node.Start(ctx);
    encoder_node.Start(ctx);
  } else {
    ffmpeg_sw_output_node.SetInputQueue(&deepsort_output_queue);
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
  yolov3_model_node.Join();
  deepsort_crop_node.Join();
  tracker_node.Join();
  deepsort_model_node.Join();
  null_out_node.Join();
  deepsort_pp_node.Join();
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

REGSITER_STREAM(deep_sort_demo, [](json config, int id) -> std::thread {
  return std::thread(DeepSortStreamThread, config, id);
});
