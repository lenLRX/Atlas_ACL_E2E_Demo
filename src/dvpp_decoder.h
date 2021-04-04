#ifndef __DVPP_DECODER_H__
#define __DVPP_DECODER_H__

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "acl_cb_thread.h"

#include "util.h"
#include "acl_model.h"

#include <functional>
#include <memory>


class DvppDecoder {
public:
  DvppDecoder() = default;
  ~DvppDecoder();
  void Destory();
  aclError Init(const pthread_t thread_id, int h, int w,
                acldvppStreamFormat profile = H264_HIGH_LEVEL);
  aclError SendFrame(AVPacket *packet);
  void RegisterHandler(std::function<void(DeviceBufferPtr)> handler);
  const std::function<void(DeviceBufferPtr)> &GetHandler();
  void SetDeviceCtx(aclrtContext *ctx);
  aclrtContext *GetDeviceCtx();

  int GetHeight();
  int GetWidth();
  int GetOutputBufferSize();

private:
  static int GetChannelId();
  int height;
  int width;
  int output_size;
  int timestamp;
  aclvdecChannelDesc *channel_desc;
  aclvdecFrameConfig *frame_config;
  std::function<void(DeviceBufferPtr)> buffer_handler;
  aclrtContext *dev_ctx;
};

#endif // __DVPP_DECODER_H__