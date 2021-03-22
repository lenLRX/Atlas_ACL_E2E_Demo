#include "acl_model.h"
#include "camera_input.h"
#include "drawing.h"
#include "dvpp_decoder.h"
#include "dvpp_encoder.h"
#include "ffmpeg_input.h"
#include "ffmpeg_output.h"
#include "util.h"
#include "vpc_resize.h"
#include "vpc_batch_crop.h"
#include "jpeg_encode.h"

#include "acl_cb_thread.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <sstream>

#include "deep_sort_py.h"

using namespace std::chrono_literals;

const static int yolov3_model_size = 416;
const static int person_class_id = 1;

// RTSP input >> DVPP decode >> DVPP resize
// >> AICORE yolov3 >> draw box >> DVPP encode >> RTMP output

void DetectAndDraw(ACLModel *yolo_model,
                   ACLModel *deepsort_model,
                   VPCBatchCrop* crop_engine,
                   uint8_t *buffer,
                   uint8_t *raw_buffer,
                   int width,
                   int height) {
  aclrtStream stream = crop_engine->GetStream();
  const auto &input_buffers = yolo_model->GetInputBuffer();
  int pic_size = yolo_model->GetInputBufferSizes()[0];
  memcpy(input_buffers[0], buffer, pic_size);
  float *img_info = (float *)input_buffers[1];
  float h_ratio = height / (float)yolov3_model_size;
  float w_ratio = width / (float)yolov3_model_size;
  img_info[0] = yolov3_model_size;
  img_info[1] = yolov3_model_size;
  img_info[2] = yolov3_model_size; // scale H
  img_info[3] = yolov3_model_size; // scale W

  yolo_model->Infer();

  int raw_buffer_size = yuv420sp_size(height, width);

  int post_nms_num = 1024;
  const auto &output_buffers = yolo_model->GetOutputBuffer();
  float *box_info = (float *)output_buffers[0];
  int32_t box_out_num = ((int32_t *)output_buffers[1])[0];

  std::cout << "result box num:" << box_out_num << std::endl;
  // PERF_TIMER();

  YUV420SPImage img(raw_buffer, height, width);
  std::vector<YUVColor> colors;
  colors.emplace_back(76, 84, 255); // Red
  colors.emplace_back(149, 43, 21); // Lime
  colors.emplace_back(29, 255, 107); // Blue
  colors.emplace_back(225, 0, 148); // Yellow
  colors.emplace_back(178, 171, 0); // Cyan
  colors.emplace_back(105, 212, 234); // Magenta

  if (box_out_num > 0) {
    const int feature_h = 128;
    const int feature_w = 64;
    const int feature_pic_size = feature_w*feature_h * 3 / 2;
    const int deepsort_batch_size = 16;
    int batch_num = box_out_num / deepsort_batch_size;

    int output_index = 0;
    const int output_feature_size = 128 * sizeof(float);
    uint8_t* output_buffer = new uint8_t[box_out_num * output_feature_size];

    std::vector<int32_t> boxes; // format (x,y,w,h)
    std::vector<float> scores;

    for (int batch_i = 0;batch_i <= batch_num; batch_i++) {
      int batch_size = deepsort_batch_size;
      if (batch_i == batch_num) {
        batch_size = box_out_num % deepsort_batch_size;
      }
      const auto &deepsort_input_buffers = deepsort_model->GetInputBuffer();

      std::vector<int> vec_x1, vec_x2, vec_y1, vec_y2;
      std::vector<int> vec_dst_h, vec_dst_w;
      std::vector<uint8_t*> vec_dst_addr;
      uint8_t* dst_addr = (uint8_t*)deepsort_input_buffers[0];

      for (int i = 0; i < batch_size; ++i) {
        int box_idx = deepsort_batch_size * batch_i + i;
        float x1 = box_info[box_out_num * 0 + box_idx] * w_ratio;
        float y1 = box_info[box_out_num * 1 + box_idx] * h_ratio;
        float x2 = box_info[box_out_num * 2 + box_idx] * w_ratio;
        float y2 = box_info[box_out_num * 3 + box_idx] * h_ratio;
        float score = box_info[box_out_num * 4 + box_idx];
        float label = box_info[box_out_num * 5 + box_idx];

        int ilabel = int(label) + 1;
        /*
        std::cout << "box idx:" << box_idx << 
                   " info: x1: " << x1 << " y1: " << y1
                   << " x2: " << x2
                  << " y2: " << y2 << " score: " << score
                  << " label: " << yolov3_label[ilabel] << std::endl;
        */
        // only track person
        if (ilabel != person_class_id) {
          continue;
        }
        
        //img.DrawRect(x1, y1, x2, y2, box_color, 3);
        int even_x1 = ((int)(x1/2))*2;
        int odd_x2 = ((int)(x2/2))*2-1;
        if (odd_x2 < 0) {
          odd_x2 = 1;
        }
        int even_y1 = ((int)(y1/2))*2;
        int odd_y2 = ((int)(y2/2))*2-1;
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

#define __CHECK_IN_RANGE(value, bound) \
if (value < 0) {\
  continue;\
}\
else if (value > bound) {\
  continue;\
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
        vec_dst_addr.push_back(dst_addr);
        vec_dst_h.push_back(feature_h);
        vec_dst_w.push_back(feature_w);
        dst_addr += feature_pic_size;
      }

      batch_size = vec_x1.size();

      if (batch_size == 0) {
        continue;
      }

      // step1: crop and resize detection box
      crop_engine->Crop(
        raw_buffer,
        vec_x1.data(),
        vec_y1.data(),
        vec_x2.data(),
        vec_y2.data(),
        vec_dst_addr.data(),
        vec_dst_h.data(),
        vec_dst_w.data(),
        batch_size
      );
      
      // step2: InferFeature
      deepsort_model->Infer();
      memcpy(output_buffer + output_index * output_feature_size,
             deepsort_model->GetOutputBuffer()[0],
             output_feature_size * batch_size);
      output_index += batch_size;
    }
    std::vector<std::vector<int>> trackings;
    {
      deep_sort_py_func(output_index, boxes.data(),
                      scores.data(), output_buffer,
                      trackings);
    }
    delete[] output_buffer;

    for (const auto& track: trackings) {
      auto& color = colors[track[0]%colors.size()];
      img.DrawRect(track[1], track[2], track[3], track[4], color, 3);
      std::cout << "draw track id:" << track[0]
        << " (" << track[1] << "," << track[2] << ","
        << track[3] << "," << track[4] << ")" << std::endl;
      img.DrawText(track[1], track[4], std::to_string(track[0]), color);
    }
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

  ACLModel yolo_model(stream);
  yolo_model.Init("./model/sample-yolov3_pp_416.om");

  std::cout << "YOLOV3 Model Info:" << std::endl;
  std::cout << yolo_model.ToString();

  ACLModel deepsort_model(stream);
  deepsort_model.Init("./model/deepsort_mars.om");

  std::cout << "deepsort Model Info:" << std::endl;
  std::cout << deepsort_model.ToString();

  FFMPEGInput ffmpeg_input;
  FFMPEGOutput ffmpeg_output;
  DvppDecoder decoder;
  VPCResizeEngine resize_engine(stream);
  VPCBatchCrop crop_engine(stream);
  CameraInput camera_input;

  auto resize_handler = [&](uint8_t *buffer) {
    // PERF_TIMER();
    resize_engine.Resize(buffer);
  };

  int camera_id = ParseCameraInput(input_addr);
  int width, height;
  if (camera_id < 0) {
    ffmpeg_input.Init(input_addr.c_str());
    width = ffmpeg_input.GetWidth();
    height = ffmpeg_input.GetHeight();
    ffmpeg_output.Init(output_addr, height, width,
                       ffmpeg_input.GetFramerate());
    decoder.Init(cb_thread.GetPid(), height, width, ffmpeg_input.GetProfile());
    decoder.SetDeviceCtx(&ctx);

    decoder.RegisterHandler(resize_handler);
  } else {
    camera_input.Init(camera_id);
    camera_input.RegisterHandler(resize_handler);
    width = camera_input.GetWidth();
    height = camera_input.GetHeight();
    ffmpeg_output.Init(output_addr, height, width, 20);
  }

  // DvppEncoder encoder;
  // encoder.Init(cb_thread.GetPid(), yolov3_model_size, yolov3_model_size,
  // &ffmpeg_output);

  resize_engine.Init(height, width, yolov3_model_size, yolov3_model_size);
  crop_engine.Init(height, width);

  resize_engine.RegisterHandler([&](uint8_t *buffer, uint8_t* raw_buffer) {
    DetectAndDraw(&yolo_model, &deepsort_model, 
                  &crop_engine, buffer,
                  raw_buffer, width, height);
    // encoder.SendFrame(buffer);
    ffmpeg_output.SendFrame(raw_buffer);
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


