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
#include "dev_mem_pool.h"
#include "device_manager.h"
#include "signal_handler.h"
#include "stream_factory.h"
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
  void *device_img_info = DevMemPool::AllocDevMem(img_info_size);

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
  if (!IsDeviceMode()) {
    CHECK_ACL(aclrtFreeHost(host_img_info));
  }
  return {output_buffers, std::get<0>(bufferx2)};
}

Yolov3PostProcess::Yolov3PostProcess(int width, int height)
    : width(width), height(height) {
  h_ratio = height / (float)yolov3_model_size;
  w_ratio = width / (float)yolov3_model_size;
}

Yolov3PostProcess::OutTy Yolov3PostProcess::Process(InTy input) {
  APP_PROFILE(Yolov3PostProcess);
  const auto &infer_results = std::get<0>(input);
  auto image_buffer = std::get<1>(input);

  int post_nms_num = 1024;
  float *box_info = (float *)infer_results[0]->GetHostPtr();
  int32_t box_out_num = ((int32_t *)infer_results[1]->GetHostPtr())[0];

  // std::cout << "result box num:" << box_out_num << std::endl;

  YUV420SPImage img((uint8_t *)image_buffer->GetHostPtr(), height, width);
  YUVColor box_color(0, 0, 0xff); // Red?

  for (int i = 0; i < box_out_num; ++i) {
    float x1 = box_info[box_out_num * 0 + i] * w_ratio;
    float y1 = box_info[box_out_num * 1 + i] * h_ratio;
    float x2 = box_info[box_out_num * 2 + i] * w_ratio;
    float y2 = box_info[box_out_num * 3 + i] * h_ratio;
    float score = box_info[box_out_num * 4 + i];
    float label = box_info[box_out_num * 5 + i];

    img.DrawRect(x1, y1, x2, y2, box_color, 3);
    img.DrawText(x1, y2, yolov3_label_zh_cn[int(label) + 1], box_color);
  }

  return image_buffer;
}

void Yolov3StreamThread(json config, int id) {
  std::string input_addr = config.at("src");
  std::string output_addr = config.at("dst");
  std::string model_path = config.at("model_path");
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

  if (camera_id < 0) {
    decoder_node.Start(ctx);
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

  aclrtStream model_stream;
  CHECK_ACL(aclrtCreateStream(&model_stream));
  Yolov3Model yolov3_model(model_path, model_stream);

  TaskNode<Yolov3Model, Yolov3Model::InTy, Yolov3Model::OutTy>
      yolov3_model_node(&yolov3_model, "Yolov3Model", stream_name);

  yolov3_model_node.SetInputQueue(&yolov3_input_queue);

  ThreadSafeQueueWithCapacity<Yolov3Model::OutTy> yolov3_output_queue(
      queue_size);
  yolov3_model_node.SetOutputQueue(&yolov3_output_queue);
  yolov3_model_node.Start(ctx);

  NullOutput<Yolov3PostProcess::InTy> null_out;
  TaskNode<NullOutput<Yolov3PostProcess::InTy>, Yolov3PostProcess::InTy, void>
      null_out_node(&null_out, "NullOutput", stream_name);

  ThreadSafeQueueWithCapacity<Yolov3Model::OutTy> dummy_output_queue(
      queue_size);

  Yolov3PostProcess post_process(width, height);
  TaskNode<Yolov3PostProcess, Yolov3PostProcess::InTy, Yolov3PostProcess::OutTy>
      post_process_node(&post_process, "Yolov3PostProcess", stream_name);

  if (!is_null_output) {
    post_process_node.SetInputQueue(&yolov3_output_queue);
    null_out_node.SetInputQueue(&dummy_output_queue);
  } else {
    post_process_node.SetInputQueue(&dummy_output_queue);
    null_out_node.SetInputQueue(&yolov3_output_queue);
  }

  dummy_output_queue.ShutDown();

  null_out_node.Start(ctx);

  ThreadSafeQueueWithCapacity<Yolov3PostProcess::OutTy>
      yolov3_post_output_queue(queue_size);
  post_process_node.SetOutputQueue(&yolov3_post_output_queue);
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
    encoder_node.SetInputQueue(&yolov3_post_output_queue);
    ffmpeg_hw_output_node.SetInputQueue(&encoder_output_queue);
    ffmpeg_hw_output_node.Start(ctx);
    encoder_node.Start(ctx);
  } else {
    ffmpeg_sw_output_node.SetInputQueue(&yolov3_post_output_queue);
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

REGSITER_STREAM(yolov3_demo, [](json config, int id) -> std::thread {
  return std::thread(Yolov3StreamThread, config, id);
});
