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

class DvppDecoder {
public:
    DvppDecoder() = default;
    aclError Init(const pthread_t thread_id, int h, int w);
    aclError SendFrame(AVPacket* packet);
private:
    int height;
    int width;
    aclvdecChannelDesc* channel_desc;
    aclvdecFrameConfig* frame_config;
};

#endif// __DVPP_DECODER_H__