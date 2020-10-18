#ifndef __RTSP_INPUT_H__
#define __RTSP_INPUT_H__

extern "C" {
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libavutil/imgutils.h"
}

#include <string>
#include <iostream>

#include "dvpp_decoder.h"
#include "acl_cb_thread.h"

class RTSPInput {
public:
    RTSPInput() = default;
    int Init(const std::string& addr);
    void Pull();
private:
    AclCallBackThread cb_thread;
    DvppDecoder decoder;
    AVFormatContext* av_fc;
    AVCodecContext* av_cc;

    int video_stream{-1};
};

#endif//__RTSP_INPUT_H__