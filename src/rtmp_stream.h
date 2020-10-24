
#ifndef __RTMP_STREAM_H__
#define __RTMP_STREAM_H__

extern "C" {
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libavutil/imgutils.h"
}

#include <string>

class RtmpContext {
public:
    RtmpContext() = default;

    int Init(std::string name, int img_h, int img_w, int frame_rate=20, int pic_fmt=AV_PIX_FMT_NV12);
    int Init(std::string name, int img_h, int img_w, AVRational frame_rate, int pic_fmt=AV_PIX_FMT_NV12);
    bool IsValid();

    void SendFrame(const uint8_t* pdata);
    void SendEncodedFrame(void* pdata, int size);

private:
    AVFormatContext* encoder_avfc;
    AVCodec* video_avc;
    AVCodecContext* video_avcc;
    AVStream* avs;
    AVDictionary *codec_options;
    AVFrame* video_frame;

    std::string stream_name;
    int h;
    int w;

    bool valid{false};
};


#endif//__RTMP_STREAM_H__

