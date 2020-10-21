#ifndef __RTSP_INPUT_H__
#define __RTSP_INPUT_H__

extern "C" {
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libavutil/imgutils.h"
}

#include <string>
#include <iostream>
#include <functional>

class RTSPInput {
public:
    RTSPInput() = default;
    int Init(const std::string& addr);
    void RegisterHandler(std::function<void(AVPacket*)> handler);
    int GetHeight();
    int GetWidth();
    void Run();
private:
    bool ReceiveSinglePacket();
    AVFormatContext* av_fc{nullptr};
    AVCodecContext* av_cc{nullptr};
    std::function<void(AVPacket*)> packet_handler;

    int video_stream{-1};
};

#endif//__RTSP_INPUT_H__