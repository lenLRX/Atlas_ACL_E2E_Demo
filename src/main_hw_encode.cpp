extern "C" {
#include "peripheral_api.h"
}

#include "opencv2/opencv.hpp"
#include <iostream>
#include <unistd.h>

#include "acl/acl.h"

#include "acl_cb_thread.h"
#include "acl_model.h"
#include "dvpp_encoder.h"
#include "ffmpeg_output.h"
#include "util.h"
#include "vpc_resize.h"
#include "yolov3_post.h"

const static int yolov3_model_size = 416;

class CameraCtx {
public:
  VPCResizeEngine *resize;
  FFMPEGOutput *rtmp;
  FFMPEGOutput *resize_rtmp;
  ACLModel *model;
  aclrtContext *dev_ctx;
  DvppEncoder *encoder_ctx;
};

int CameraCallBack(const void *pdata, int size, void *param) {
  // std::cerr << "CameraCallBack size: " << size << std::endl;
  CameraCtx *ctx = (CameraCtx *)param;
  CHECK_ACL(aclrtSetCurrentContext(*(ctx->dev_ctx)));

  {
      // PERF_TIMER();
      // ctx->rtmp->SendFrame((const uint8_t*)pdata);
  }

  {
    // PERF_TIMER();
    CHECK_ACL(ctx->resize->Resize((const uint8_t *)pdata));
  }

  const uint8_t *resized_buffer =
      (const uint8_t *)ctx->resize->GetOutputBuffer();
  {
    // PERF_TIMER();
    ctx->resize_rtmp->SendFrame(resized_buffer);
  }

  const auto &input_buffers = ctx->model->GetInputBuffer();
  memcpy(input_buffers[0], resized_buffer,
         ctx->model->GetInputBufferSizes()[0]);
  float *img_info = (float *)input_buffers[1];
  img_info[0] = yolov3_model_size;
  img_info[1] = yolov3_model_size;
  img_info[2] = 720;  // scale H
  img_info[3] = 1280; // scale W
  {
    // PERF_TIMER();
    ctx->model->Infer();
  }

  int post_nms_num = 1024;
  const auto &output_buffers = ctx->model->GetOutputBuffer();
  float *box_info = (float *)output_buffers[0];
  int32_t box_out_num = ((int32_t *)output_buffers[1])[0];

  cv::Mat mYUV(720 * 1.5, 1280, CV_8UC1, (void *)pdata);
  cv::Mat mRGB(720, 1280, CV_8UC3);
  cv::Mat mYUV420P(720 * 1.5, 1280, CV_8UC1);
  {
    PERF_TIMER();
    cv::cvtColor(mYUV, mRGB, CV_YUV2RGB_NV12, 3);
  }

  std::cout << "result box num:" << box_out_num << std::endl;

  for (int i = 0; i < box_out_num; ++i) {
    float x1 = box_info[box_out_num * 0 + i];
    float y1 = box_info[box_out_num * 1 + i];
    float x2 = box_info[box_out_num * 2 + i];
    float y2 = box_info[box_out_num * 3 + i];
    float score = box_info[box_out_num * 4 + i];
    float label = box_info[box_out_num * 5 + i];
    std::cout << "box info: x1: " << x1 << " y2: " << y1 << " x2: " << x2
              << " y2: " << y2 << " score: " << score
              << " label: " << yolov3_label[int(label) + 1] << std::endl;
    //<< " label: " << label << std::endl;
    cv::rectangle(mRGB, {(int)x1, (int)y1}, {(int)x2, (int)y2},
                  cv::Scalar(237, 149, 100));
  }

  {
    PERF_TIMER();
    cv::cvtColor(mRGB, mYUV420P, CV_RGB2YUV_I420, 1);
  }

  // ctx->rtmp->SendFrame((const uint8_t*)(mYUV420P.ptr()));
  ctx->encoder_ctx->SendFrame((uint8_t *)(pdata));

  return 1;
}

int main(int argc, char **argv) {
  CHECK_ACL(aclInit(nullptr));

  int ret;
  ret = MediaLibInit();
  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "MediaLibInit failed " << ret << std::endl;
    return -1;
  }

  ret = IsChipAlive(NULL);

  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "IsChipAlive failed" << std::endl;
    return -1;
  }

  ret = OpenCamera(0);

  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "OpenCamera 0 failed" << std::endl;
    return -1;
  }

  struct CameraResolution supported_resolution[HIAI_MAX_CAMERARESOLUTION_COUNT];

  ret = GetCameraProperty(0, CAMERA_PROP_SUPPORTED_RESOLUTION,
                          supported_resolution);
  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "GetCameraProperty Resolution Failed " << ret << std::endl;
    return -1;
  }

  int i = 0;
  do {
    std::cout << "Camera suppported width=" << supported_resolution[i].width
              << " height=" << supported_resolution[i].height << std::endl;
    ++i;
  } while (supported_resolution[i].width != -1 &&
           i < HIAI_MAX_CAMERARESOLUTION_COUNT);

  int fps = 20;
  ret = SetCameraProperty(0, CAMERA_PROP_FPS, &fps);
  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "SetCameraProperty 0 failed" << ret << std::endl;
    ;
    return -1;
  }

  struct CameraResolution resolution;

  resolution.height = 720;
  resolution.width = 1280;
  ret = SetCameraProperty(0, CAMERA_PROP_RESOLUTION, &resolution);
  if (ret != LIBMEDIA_STATUS_OK) {
    std::cerr << "SetCameraProperty 0 failed" << ret << std::endl;
    return -1;
  }

  FFMPEGOutput rtmp_ctx;
  ret = rtmp_ctx.Init("mystream", 720, 1280, AV_PIX_FMT_YUV420P);

  if (ret != 0) {
    std::cerr << "InitRtmp Ctx failed ret: " << ret << std::endl;
    return -1;
  }

  CHECK_ACL(aclrtSetDevice(0));

  aclrtContext ctx;
  CHECK_ACL(aclrtCreateContext(&ctx, 0));
  CHECK_ACL(aclrtSetCurrentContext(ctx));
  aclrtStream stream;
  CHECK_ACL(aclrtCreateStream(&stream));

  aclrtRunMode mode;
  CHECK_ACL(aclrtGetRunMode(&mode));
  std::cerr << "run mode:"
            << std::string(mode == ACL_DEVICE ? "device" : "host") << std::endl;

  AclCallBackThread cb_thread;

  CHECK_ACL(aclrtSubscribeReport(cb_thread.GetPid(), stream));

  VPCResizeEngine resize_engine(stream);

  resize_engine.Init(720, 1280, yolov3_model_size, yolov3_model_size);

  FFMPEGOutput resized_ctx;
  ret = resized_ctx.Init("resize", yolov3_model_size, yolov3_model_size);

  DvppEncoder encoder;
  encoder.Init(cb_thread.GetPid(), 720, 1280, &rtmp_ctx);

  ACLModel model(stream);
  model.Init("./model/sample-yolov3_pp_416.om");

  std::cout << "Model Info:" << std::endl;
  std::cout << model.ToString();

  CameraCtx camera_ctx;
  camera_ctx.resize = &resize_engine;
  camera_ctx.rtmp = &rtmp_ctx;
  camera_ctx.resize_rtmp = &resized_ctx;
  camera_ctx.dev_ctx = &ctx;
  camera_ctx.model = &model;
  camera_ctx.encoder_ctx = &encoder;

  ret = CapCamera(0, CameraCallBack, &camera_ctx);
  sleep(50000000);

  /*
  int cam_buffer_size = yuv420sp_size(720, 1280);

  uint8_t* cam_buffer = (uint8_t*)malloc(cam_buffer_size);

  CameraCapMode cap_mode = CAMERA_CAP_ACTIVE;


  ret = SetCameraProperty(0, CAMERA_PROP_CAP_MODE, &mode);

  if(ret != LIBMEDIA_STATUS_OK) {
      std::cerr << "SetCameraProperty 0 failed " << ret << std::endl;;
      return -1;
  }

  while (true) {
      ReadFrameFromCamera(0, cam_buffer, &cam_buffer_size);
      rtmp_ctx.SendFrame((const uint8_t*)cam_buffer);
      resize_engine.Resize((const uint8_t*)cam_buffer, cam_buffer_size);
      resized_ctx.SendFrame((const uint8_t*)resize_engine.GetOutputBuffer());
  }
  */

  CHECK_ACL(aclFinalize());

  return 0;
}