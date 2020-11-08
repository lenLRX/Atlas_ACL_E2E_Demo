
#ifndef __FFMPEG_OUTPUT_H__
#define __FFMPEG_OUTPUT_H__

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
}

#include <string>

class FFMPEGOutput {
public:
  FFMPEGOutput() = default;

  int Init(std::string name, int img_h, int img_w, int frame_rate = 20,
           int pic_fmt = AV_PIX_FMT_NV12);
  int Init(std::string name, int img_h, int img_w, AVRational frame_rate,
           int pic_fmt = AV_PIX_FMT_NV12);
  bool IsValid();

  void SendFrame(const uint8_t *pdata);
  void SendEncodedFrame(void *pdata, int size);
  void Close();

private:
  AVFormatContext *encoder_avfc{nullptr};
  AVCodec *video_avc{nullptr};
  AVCodecContext *video_avcc{nullptr};
  AVStream *avs{nullptr};
  AVDictionary *codec_options{nullptr};
  AVFrame *video_frame{nullptr};

  std::string stream_name;
  int h;
  int w;

  bool valid{false};
};

#endif //__FFMPEG_OUTPUT_H__
