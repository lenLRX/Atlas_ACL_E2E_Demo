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
#include "task_node.h"
#include "yolov3_stream.h"

using namespace std::chrono_literals;

const static int yolov3_model_size = 416;

Yolov3Model::Yolov3Model(const std::string &path, aclrtStream stream)
    : yolov3_model(stream), model_stream(stream) {
  yolov3_model.Init(path.c_str());
  std::cout << "Model Info:" << std::endl;
  std::cout << yolov3_model.ToString();
}

Yolov3Model::OutTy Yolov3Model::Process(Yolov3Model::InTy bufferx2) {
  APP_PROFILE(Yolov3Model);
  size_t img_info_size = yolov3_model.GetInputBufferSizes()[1];
  void *device_img_info;
  CHECK_ACL(
      aclrtMalloc(&device_img_info, img_info_size, ACL_MEM_MALLOC_HUGE_FIRST));

  float *host_img_info;
  if (IsDeviceMode()) {
    host_img_info = (float *)device_img_info;
  } else {
    CHECK_ACL(aclrtMallocHost(((void **)&host_img_info), img_info_size));
  }

  host_img_info[0] = yolov3_model_size;
  host_img_info[1] = yolov3_model_size;
  host_img_info[2] = yolov3_model_size; // scale H
  host_img_info[3] = yolov3_model_size; // scale W

  if (!IsDeviceMode()) {
    CHECK_ACL(aclrtMemcpy(device_img_info, img_info_size, host_img_info,
                          img_info_size, ACL_MEMCPY_HOST_TO_DEVICE));
  }

  DeviceBufferPtr img_info_buffer;
  img_info_buffer = std::make_shared<DeviceBuffer>(
      device_img_info, img_info_size, DeviceBuffer::DevMemDeleter());

  auto output_buffers =
      yolov3_model.Infer({std::get<1>(bufferx2), img_info_buffer});
  return {output_buffers, std::get<1>(bufferx2)};
}

Yolov3PostProcess::OutTy Yolov3PostProcess::Process(InTy input) {
  APP_PROFILE(Yolov3PostProcess);
  const auto &infer_results = std::get<0>(input);
  auto image_buffer = std::get<1>(input);

  int post_nms_num = 1024;
  float *box_info = (float *)infer_results[0]->GetHostPtr();
  int32_t box_out_num = ((int32_t *)infer_results[1]->GetHostPtr())[0];

  std::cout << "result box num:" << box_out_num << std::endl;

  YUV420SPImage img((uint8_t *)image_buffer->GetHostPtr(), yolov3_model_size,
                    yolov3_model_size);
  YUVColor box_color(0, 0, 0xff); // Red?

  for (int i = 0; i < box_out_num; ++i) {
    float x1 = box_info[box_out_num * 0 + i];
    float y1 = box_info[box_out_num * 1 + i];
    float x2 = box_info[box_out_num * 2 + i];
    float y2 = box_info[box_out_num * 3 + i];
    float score = box_info[box_out_num * 4 + i];
    float label = box_info[box_out_num * 5 + i];
    /*
    std::cout << "box info: x1: " << x1 << " y2: " << y1 << " x2: " << x2
              << " y2: " << y2 << " score: " << score
              << " label: " << yolov3_label[int(label) + 1] << std::endl;
    */
    img.DrawRect(x1, y1, x2, y2, box_color, 3);
    img.DrawText(x1, y2, yolov3_label_zh_cn[int(label) + 1], box_color);
  }

  return image_buffer;
}

static void DetectAndDraw(ACLModel *model, DeviceBufferPtr buffer) {
  APP_PROFILE(DetectAndDraw);
  size_t img_info_size = model->GetInputBufferSizes()[1];
  void *device_img_info;
  CHECK_ACL(
      aclrtMalloc(&device_img_info, img_info_size, ACL_MEM_MALLOC_HUGE_FIRST));

  float *host_img_info;
  if (IsDeviceMode()) {
    host_img_info = (float *)device_img_info;
  } else {
    CHECK_ACL(aclrtMallocHost(((void **)&host_img_info), img_info_size));
  }

  host_img_info[0] = yolov3_model_size;
  host_img_info[1] = yolov3_model_size;
  host_img_info[2] = yolov3_model_size; // scale H
  host_img_info[3] = yolov3_model_size; // scale W

  if (!IsDeviceMode()) {
    CHECK_ACL(aclrtMemcpy(device_img_info, img_info_size, host_img_info,
                          img_info_size, ACL_MEMCPY_HOST_TO_DEVICE));
  }

  DeviceBufferPtr img_info_buffer;
  img_info_buffer = std::make_shared<DeviceBuffer>(
      device_img_info, img_info_size, DeviceBuffer::DevMemDeleter());

  auto output_buffers = model->Infer({buffer, img_info_buffer});

  int post_nms_num = 1024;
  float *box_info = (float *)output_buffers[0]->GetHostPtr();
  int32_t box_out_num = ((int32_t *)output_buffers[1]->GetHostPtr())[0];

  std::cout << "result box num:" << box_out_num << std::endl;
  // PERF_TIMER();

  YUV420SPImage img((uint8_t *)buffer->GetHostPtr(), yolov3_model_size,
                    yolov3_model_size);
  YUVColor box_color(0, 0, 0xff); // Red?

  for (int i = 0; i < box_out_num; ++i) {
    float x1 = box_info[box_out_num * 0 + i];
    float y1 = box_info[box_out_num * 1 + i];
    float x2 = box_info[box_out_num * 2 + i];
    float y2 = box_info[box_out_num * 3 + i];
    float score = box_info[box_out_num * 4 + i];
    float label = box_info[box_out_num * 5 + i];
    /*
    std::cout << "box info: x1: " << x1 << " y2: " << y1 << " x2: " << x2
              << " y2: " << y2 << " score: " << score
              << " label: " << yolov3_label[int(label) + 1] << std::endl;
    */
    img.DrawRect(x1, y1, x2, y2, box_color, 3);
    img.DrawText(x1, y2, yolov3_label_zh_cn[int(label) + 1], box_color);
  }
}

void Yolov3StreamThread(json config) {
  std::string input_addr = config.at("src");
  std::string output_addr = config.at("dst");
  std::string model_path = config.at("model_path");

  CHECK_ACL(aclrtSetDevice(0));
  AclCallBackThread cb_thread;

  aclrtContext ctx;
  CHECK_ACL(aclrtCreateContext(&ctx, 0));
  CHECK_ACL(aclrtSetCurrentContext(ctx));
  aclrtStream stream;
  CHECK_ACL(aclrtCreateStream(&stream));
  CHECK_ACL(aclrtSubscribeReport(cb_thread.GetPid(), stream));

  const int queue_size = 16;

  FFMPEGInput ffmpeg_input;

  using buf_tup_t = std::tuple<DeviceBufferPtr, DeviceBufferPtr>;

  ThreadSafeQueueWithCapacity<AVPacket> decoder_input_queue(queue_size);

  DvppDecoder decoder;
  TaskNode<DvppDecoder, AVPacket, void> decoder_node(&decoder, "DvppDecoder",
                                                     input_addr);
  decoder_node.SetInputQueue(&decoder_input_queue);

  ThreadSafeQueueWithCapacity<DeviceBufferPtr> resize_input_queue(queue_size);
  decoder.SetOutputQueue(&resize_input_queue);
  decoder_node.Start(ctx);

  aclrtStream resize_stream;
  CHECK_ACL(aclrtCreateStream(&resize_stream));
  VPCResizeEngine resize_engine(resize_stream);

  TaskNode<VPCResizeEngine, DeviceBufferPtr, buf_tup_t> resize_engine_node(
      &resize_engine, "VPCResizeEngine", input_addr);
  resize_engine_node.SetInputQueue(&resize_input_queue);

  ThreadSafeQueueWithCapacity<buf_tup_t> yolov3_input_queue(queue_size);
  resize_engine_node.SetOutputQueue(&yolov3_input_queue);
  resize_engine_node.Start(ctx);

  aclrtStream model_stream;
  CHECK_ACL(aclrtCreateStream(&model_stream));
  Yolov3Model yolov3_model(model_path, model_stream);

  TaskNode<Yolov3Model, Yolov3Model::InTy, Yolov3Model::OutTy>
      yolov3_model_node(&yolov3_model, "Yolov3Model", input_addr);

  yolov3_model_node.SetInputQueue(&yolov3_input_queue);

  ThreadSafeQueueWithCapacity<Yolov3Model::OutTy> yolov3_post_input_queue(
      queue_size);
  yolov3_model_node.SetOutputQueue(&yolov3_post_input_queue);
  yolov3_model_node.Start(ctx);

  Yolov3PostProcess post_process;
  TaskNode<Yolov3PostProcess, Yolov3PostProcess::InTy, Yolov3PostProcess::OutTy>
      post_process_node(&post_process, "Yolov3PostProcess", input_addr);

  post_process_node.SetInputQueue(&yolov3_post_input_queue);
  ThreadSafeQueueWithCapacity<Yolov3PostProcess::OutTy>
      yolov3_post_output_queue(queue_size);
  post_process_node.SetOutputQueue(&yolov3_post_output_queue);
  post_process_node.Start(ctx);

  FFMPEGOutput ffmpeg_output;
  TaskNode<FFMPEGOutput, DeviceBufferPtr, void> ffmpeg_output_node(
      &ffmpeg_output, "FFMPEGOutput", input_addr);

  ffmpeg_output_node.SetInputQueue(&yolov3_post_output_queue);
  ffmpeg_output_node.Start(ctx);

  CameraInput camera_input;

  int camera_id = ParseCameraInput(input_addr);
  int width, height;
  if (camera_id < 0) {
    ffmpeg_input.Init(input_addr.c_str());
    ffmpeg_output.Init(output_addr, yolov3_model_size, yolov3_model_size,
                       ffmpeg_input.GetFramerate());
    width = ffmpeg_input.GetWidth();
    height = ffmpeg_input.GetHeight();
    decoder.Init(cb_thread.GetPid(), height, width, ffmpeg_input.GetProfile());
    decoder.SetDeviceCtx(&ctx);
  } else {
    camera_input.Init(camera_id);
    // camera_input.RegisterHandler(resize_handler);
    width = camera_input.GetWidth();
    height = camera_input.GetHeight();
    ffmpeg_output.Init(output_addr, yolov3_model_size, yolov3_model_size,
                       camera_input.GetFPS());
  }

  // DvppEncoder encoder;
  // encoder.Init(cb_thread.GetPid(), yolov3_model_size, yolov3_model_size,
  // &ffmpeg_output);

  resize_engine.Init(height, width, yolov3_model_size, yolov3_model_size);

  if (camera_id < 0) {
    TaskNode<FFMPEGInput, void, void> ffmpeg_input_node(
        &ffmpeg_input, "FFMPEGInput", input_addr);
    ffmpeg_input.SetOutputQueue(&decoder_input_queue);
    ffmpeg_input_node.Start(ctx);
    ffmpeg_input_node.Join();
  } else {
    camera_input.Run();
  }

  resize_engine_node.Join();
  yolov3_model_node.Join();
  post_process_node.Join();
  ffmpeg_output_node.Join();
  if (camera_id < 0) {
    decoder_node.Join();
    decoder.Destory();
  }

  resize_engine.Destory();
  ffmpeg_output.Close();
  // encoder.Destory();

  // CHECK_ACL(aclrtUnSubscribeReport(cb_thread.GetPid(), stream));
  // cb_thread.Join();
  std::cout << "End of stream input: " << input_addr << std::endl;
}

std::thread MakeYolov3Stream(json config) {
  return std::thread(Yolov3StreamThread, config);
}