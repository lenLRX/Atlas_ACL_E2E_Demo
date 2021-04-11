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

#include "util.h"

class FFMPEGInput {
public:
  FFMPEGInput() = default;
  int Init(const std::string &addr);
  int GetHeight();
  int GetWidth();
  acldvppStreamFormat GetProfile();
  AVRational GetFramerate();
  void Run();
  void Process() { Run(); }
  void SetOutputQueue(ThreadSafeQueueWithCapacity<AVPacket>* queue);

private:
  bool ReceiveSinglePacket();
  bool ReceivePacketWithBSF();
  bool need_bsf{false};
  AVFormatContext *av_fc{nullptr};
  AVCodecContext *av_cc{nullptr};
  AVBSFContext *bsfc{nullptr};
  const AVBitStreamFilter *bsf_filter{nullptr};
  AVCodecContext *decoder_context;
  ThreadSafeQueueWithCapacity<AVPacket>* output_queue{nullptr};

  int video_stream{-1};
};

#endif //__FFMPEG_INPUT_H__