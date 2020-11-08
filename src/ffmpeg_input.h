#ifndef __FFMPEG_INPUT_H__
#define __FFMPEG_INPUT_H__

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
}

#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#include <functional>
#include <iostream>
#include <string>

class FFMPEGInput {
public:
  FFMPEGInput() = default;
  int Init(const std::string &addr);
  void RegisterHandler(std::function<void(AVPacket *)> handler);
  int GetHeight();
  int GetWidth();
  acldvppStreamFormat GetProfile();
  AVRational GetFramerate();
  void Run();

private:
  bool ReceiveSinglePacket();
  bool ReceivePacketWithBSF();
  bool need_bsf{false};
  AVFormatContext *av_fc{nullptr};
  AVCodecContext *av_cc{nullptr};
  AVBSFContext *bsfc{nullptr};
  const AVBitStreamFilter *bsf_filter{nullptr};
  AVCodecContext *decoder_context;
  std::function<void(AVPacket *)> packet_handler;

  int video_stream{-1};
};

#endif //__FFMPEG_INPUT_H__