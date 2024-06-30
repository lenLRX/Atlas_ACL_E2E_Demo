#include "dvpp_encoder.h"
#include "ffmpeg_output.h"

#include <iostream>
#include <mutex>
#include <thread>

class EncoderContext {
public:
  EncoderContext(DeviceBufferPtr buf, DvppEncoder *encoder)
      : buffer(buf), encoder(encoder) {}
  // hold reference until encode is done
  DeviceBufferPtr buffer;
  DvppEncoder *encoder;
};

// only 1 stream can use VENC in an process
class EncoderLock {
public:
  bool Lock() {
    std::lock_guard<std::mutex> guard(mtx);
    if (!locked) {
      locked = true;
      return true;
    }
    return false;
  }

  void Unlock() { locked = false; }

  static EncoderLock &GetInstance() {
    static EncoderLock lock;
    return lock;
  }

private:
  EncoderLock() = default;

  std::mutex mtx;
  bool locked{false};
};

static void EncoderCallback(acldvppPicDesc *input, acldvppStreamDesc *output,
                            void *userdata) {
  uint32_t retcode = acldvppGetStreamDescRetCode(output);
  // std::cerr << "Encoder Callback retcode:" << retcode << std::endl;

  void *data_ptr = acldvppGetStreamDescData(output);
  uint32_t data_size = acldvppGetStreamDescSize(output);

  void *host_buffer = malloc(data_size);
  if (IsDeviceMode()) {
    memcpy(host_buffer, data_ptr, data_size);
  } else {
    CHECK_ACL(aclrtMemcpy(host_buffer, data_size, data_ptr, data_size,
                          ACL_MEMCPY_DEVICE_TO_HOST));
  }

  EncoderContext *ctx = (EncoderContext *)userdata;
  DvppEncoder *encoder = ctx->encoder;
  auto *queue = encoder->GetOutputQueue();
  queue->push(std::make_tuple(host_buffer, data_size));
  delete ctx;
  CHECK_ACL(acldvppDestroyPicDesc(input));
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
  EncoderLock::GetInstance().Unlock();
  // std::cout << "DvppEncoder::~DvppEncoder End" << std::endl;
}

aclError DvppEncoder::Init(const pthread_t thread_id, int h, int w) {
  bool locked = EncoderLock::GetInstance().Lock();
  if (!locked) {
    std::cerr << "only one stream with hw encoder is supported, please check "
                 "your config!"
              << std::endl;
    throw std::runtime_error("only one VENC supported");
  }

  height = h;
  width = w;
  size = (width * height * 3) / 2;
  std::cout << "[DvppEncoder::Init] height:" << height << " width: " << width
            << " size: " << size << std::endl;
  CHECK_ACL(aclvencSetChannelDescThreadId(channel_desc, thread_id));
  CHECK_ACL(aclvencSetChannelDescCallback(channel_desc, &EncoderCallback));
  CHECK_ACL(aclvencSetChannelDescEnType(channel_desc,
                                        H264_MAIN_LEVEL)); // H265-main level
  CHECK_ACL(aclvencSetChannelDescPicFormat(channel_desc,
                                           PIXEL_FORMAT_YUV_SEMIPLANAR_420));
  CHECK_ACL(aclvencSetChannelDescPicHeight(channel_desc, h));
  CHECK_ACL(aclvencSetChannelDescPicWidth(channel_desc, w));
  CHECK_ACL(aclvencSetChannelDescKeyFrameInterval(channel_desc, 12));
  // CHECK_ACL(aclvencSetChannelDescMaxBitRate(channel_desc, 100));

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
  CHECK_ACL(
      aclvencSendFrame(channel_desc, pic_desc, nullptr, frame_config, this));
  return ACL_ERROR_NONE;
}

void DvppEncoder::Process(DeviceBufferPtr buffer) {
  APP_PROFILE(DvppEncoder::Process);
  // copy buffer to device if it is modified by host (box drawing)
  buffer->CopyToDevice();
  acldvppPicDesc *pic_desc = acldvppCreatePicDesc();
  EncoderContext *ctx = new EncoderContext(buffer, this);
  CHECK_ACL(acldvppSetPicDescData(pic_desc, buffer->GetDevicePtr()));
  CHECK_ACL(acldvppSetPicDescSize(pic_desc, size));
  CHECK_ACL(acldvppSetPicDescFormat(pic_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
  CHECK_ACL(acldvppSetPicDescWidth(pic_desc, width));
  CHECK_ACL(acldvppSetPicDescHeight(pic_desc, height));
  CHECK_ACL(acldvppSetPicDescWidthStride(pic_desc, width));
  CHECK_ACL(acldvppSetPicDescHeightStride(pic_desc, height));
  // IFrame every 16 frame
  if (frame_count % 16 == 0) {
    CHECK_ACL(aclvencSetFrameConfigForceIFrame(frame_config, 1));
  } else {
    CHECK_ACL(aclvencSetFrameConfigForceIFrame(frame_config, 0));
  }
  ++frame_count;
  CHECK_ACL(
      aclvencSendFrame(channel_desc, pic_desc, nullptr, frame_config, ctx));
}

void DvppEncoder::SetOutputQueue(ThreadSafeQueueWithCapacity<OutTy> *queue) {
  output_queue = queue;
}

ThreadSafeQueueWithCapacity<DvppEncoder::OutTy> *DvppEncoder::GetOutputQueue() {
  return output_queue;
}
