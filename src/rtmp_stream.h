
#ifndef __FFMPEG_STREAM_H__
#define __FFMPEG_STREAM_H__

extern "C" {
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libavutil/imgutils.h"
}

class RtmpContext {
public:
    RtmpContext() = default;

    int Init();
    bool IsValid();

    void SendFrame(uint8_t* pdata);

private:
    AVFormatContext* encoder_avfc;
    AVCodec* video_avc;
    AVCodecContext* video_avcc;
    AVStream* avs;
    AVDictionary *codec_options;
    AVFrame* video_frame;

    bool valid{false};
};

void ffmpeg_test(uint8_t* pdata);

#endif//__FFMPEG_STREAM_H__

