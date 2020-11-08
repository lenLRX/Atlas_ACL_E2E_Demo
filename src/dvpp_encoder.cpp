#include "dvpp_encoder.h"
#include "ffmpeg_output.h"

#include <iostream>

static void EncoderCallback(acldvppPicDesc *input, acldvppStreamDesc *output,
                            void *userdata) {
  uint32_t retcode = acldvppGetStreamDescRetCode(output);
  std::cerr << "Encoder Callback retcode:" << retcode << std::endl;

  void *data_ptr = acldvppGetStreamDescData(output);
  uint32_t data_size = acldvppGetStreamDescSize(output);

  std::cerr << "Encoder Buffer size: " << data_size << std::endl;

  FFMPEGOutput *ctx = (FFMPEGOutput *)userdata;
  ctx->SendEncodedFrame(data_ptr, data_size);
}

DvppEncoder::DvppEncoder() {
  channel_desc = aclvencCreateChannelDesc();
  frame_config = aclvencCreateFrameConfig();
}

DvppEncoder::~DvppEncoder() {}

void DvppEncoder::Destory() {
  aclvencDestroyChannel(channel_desc);
  aclvencDestroyChannelDesc(channel_desc);
  aclvencDestroyFrameConfig(frame_config);
  std::cout << "DvppEncoder::~DvppEncoder End" << std::endl;
}

aclError DvppEncoder::Init(const pthread_t thread_id, int h, int w,
                           FFMPEGOutput *ctx) {
  height = h;
  width = w;
  size = (width * height * 3) / 2;
  std::cout << "[DvppEncoder::Init] height:" << height << " width: " << width
            << " size: " << size << std::endl;
  rtmp_ctx = ctx;
  CHECK_ACL(aclvencSetChannelDescThreadId(channel_desc, thread_id));
  CHECK_ACL(aclvencSetChannelDescCallback(channel_desc, &EncoderCallback));
  CHECK_ACL(aclvencSetChannelDescEnType(channel_desc,
                                        H264_HIGH_LEVEL)); // H265-main level
  CHECK_ACL(aclvencSetChannelDescPicFormat(channel_desc,
                                           PIXEL_FORMAT_YUV_SEMIPLANAR_420));
  CHECK_ACL(aclvencSetChannelDescPicHeight(channel_desc, h));
  CHECK_ACL(aclvencSetChannelDescPicWidth(channel_desc, w));
  CHECK_ACL(aclvencSetChannelDescKeyFrameInterval(channel_desc, 1));

  CHECK_ACL(aclvencCreateChannel(channel_desc));

  CHECK_ACL(aclvencSetFrameConfigForceIFrame(frame_config, 0));
  CHECK_ACL(aclvencSetFrameConfigEos(frame_config, 0));

  return ACL_ERROR_NONE;
}

aclError DvppEncoder::SendFrame(uint8_t *data) {
  acldvppPicDesc *pic_desc = acldvppCreatePicDesc();
  CHECK_ACL(acldvppSetPicDescData(pic_desc, data));
  CHECK_ACL(acldvppSetPicDescSize(pic_desc, size));
  CHECK_ACL(acldvppSetPicDescFormat(pic_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
  CHECK_ACL(acldvppSetPicDescWidth(pic_desc, width));
  CHECK_ACL(acldvppSetPicDescHeight(pic_desc, height));
  CHECK_ACL(acldvppSetPicDescWidthStride(pic_desc, width));
  CHECK_ACL(acldvppSetPicDescHeightStride(pic_desc, height));
  CHECK_ACL(aclvencSendFrame(channel_desc, pic_desc, nullptr, frame_config,
                             rtmp_ctx));
}
