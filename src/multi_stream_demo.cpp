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

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using namespace std::chrono_literals;

const static int yolov3_model_size = 416;

// RTSP input >> DVPP decode >> DVPP resize
// >> AICORE yolov3 >> draw box >> DVPP encode >> RTMP output

void DetectAndDraw(ACLModel *model, uint8_t *buffer) {
  const auto &input_buffers = model->GetInputBuffer();
  memcpy(input_buffers[0], buffer, model->GetInputBufferSizes()[0]);
  float *img_info = (float *)input_buffers[1];
  img_info[0] = yolov3_model_size;
  img_info[1] = yolov3_model_size;
  img_info[2] = yolov3_model_size; // scale H
  img_info[3] = yolov3_model_size; // scale W

  model->Infer();

  int post_nms_num = 1024;
  const auto &output_buffers = model->GetOutputBuffer();
  float *box_info = (float *)output_buffers[0];
  int32_t box_out_num = ((int32_t *)output_buffers[1])[0];

  std::cout << "result box num:" << box_out_num << std::endl;
  // PERF_TIMER();

  YUV420SPImage img(buffer, yolov3_model_size, yolov3_model_size);
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
  }
}

void StreamThread(std::string input_addr, std::string output_addr) {
  CHECK_ACL(aclrtSetDevice(0));
  AclCallBackThread cb_thread;

  aclrtContext ctx;
  CHECK_ACL(aclrtCreateContext(&ctx, 0));
  CHECK_ACL(aclrtSetCurrentContext(ctx));
  aclrtStream stream;
  CHECK_ACL(aclrtCreateStream(&stream));
  CHECK_ACL(aclrtSubscribeReport(cb_thread.GetPid(), stream));

  ACLModel model(stream);
  model.Init("./model/sample-yolov3_pp_416.om");

  std::cout << "Model Info:" << std::endl;
  std::cout << model.ToString();

  FFMPEGInput ffmpeg_input;
  FFMPEGOutput ffmpeg_output;
  DvppDecoder decoder;
  VPCResizeEngine resize_engine(stream);
  CameraInput camera_input;

  auto resize_handler = [&](uint8_t *buffer) {
    // PERF_TIMER();
    resize_engine.Resize(buffer);
  };

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

    decoder.RegisterHandler(resize_handler);
  } else {
    camera_input.Init(camera_id);
    camera_input.RegisterHandler(resize_handler);
    width = camera_input.GetWidth();
    height = camera_input.GetHeight();
    ffmpeg_output.Init(output_addr, yolov3_model_size, yolov3_model_size, camera_input.GetFPS());
  }

  // DvppEncoder encoder;
  // encoder.Init(cb_thread.GetPid(), yolov3_model_size, yolov3_model_size,
  // &ffmpeg_output);

  resize_engine.Init(height, width, yolov3_model_size, yolov3_model_size);

  resize_engine.RegisterHandler([&](uint8_t *buffer, uint8_t* raw_buffer) {
    DetectAndDraw(&model, buffer);
    // encoder.SendFrame(buffer);
    ffmpeg_output.SendFrame(buffer);
  });

  if (camera_id < 0) {
    ffmpeg_input.RegisterHandler(
        [&](AVPacket *packet) { decoder.SendFrame(packet); });
    ffmpeg_input.Run();

    decoder.Destory();
  } else {
    camera_input.Run();
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));

  resize_engine.Destory();
  ffmpeg_output.Close();
  // encoder.Destory();

  // CHECK_ACL(aclrtUnSubscribeReport(cb_thread.GetPid(), stream));
  // cb_thread.Join();
  std::cout << "End of stream input: " << input_addr << std::endl;
}

int main(int argc, char **argv) {
  CHECK_ACL(aclInit(nullptr));

  std::string input_promot = "--input";
  std::string output_promot = "--output";

  std::vector<std::thread> streams;
  if ((argc - 1) % 4 != 0) {
    std::cerr << "Invalid cmd line option" << std::endl;
    return -1;
  }

  int stream_num = (argc - 1) / 4;
  int arg_i = 1;

  for (int i = 0; i < stream_num; ++i) {
    std::string i_promot(argv[arg_i]);
    if (i_promot != input_promot) {
      std::cerr << "invalid option: " << i_promot << std::endl;
      return -1;
    }
    ++arg_i;

    std::string input_addr(argv[arg_i]);

    ++arg_i;
    std::string o_promot(argv[arg_i]);
    if (o_promot != output_promot) {
      std::cerr << "invalid option: " << i_promot << std::endl;
      return -1;
    }

    ++arg_i;

    std::string output_addr(argv[arg_i]);
    ++arg_i;
    std::cout << "Add stream --input " << input_addr << " --output "
              << output_addr << std::endl;
    streams.emplace_back(StreamThread, input_addr, output_addr);
  }

  for (auto &t : streams) {
    t.join();
  }

  return 0;
}
