#ifndef __DVPP_DECODER_H__
#define __DVPP_DECODER_H__

extern "C" {
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
}

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "acl_cb_thread.h"

#include "util.h"

#include <functional>

class DvppDecoder {
public:
    DvppDecoder() = default;
    aclError Init(const pthread_t thread_id, int h, int w);
    aclError SendFrame(AVPacket* packet);
    void RegisterHandler(std::function<void(uint8_t*)> handler);
    const std::function<void(uint8_t*)>& GetHandler();
    void SetDeviceCtx(aclrtContext* ctx);
    aclrtContext* GetDeviceCtx();

    int GetHeight();
    int GetWidth();
    int GetOutputBufferSize();
private:
    int height;
    int width;
    int output_size;
    aclvdecChannelDesc* channel_desc;
    aclvdecFrameConfig* frame_config;
    std::function<void(uint8_t*)> buffer_handler;
    aclrtContext* dev_ctx;
};

#endif// __DVPP_DECODER_H__