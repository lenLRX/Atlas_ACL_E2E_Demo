
#ifndef __FFMPEG_OUTPUT_H__
#define __FFMPEG_OUTPUT_H__

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
}

#include <chrono>
#include <string>
#include <tuple>
#include <fstream>

#include "acl_model.h"

using Duration = std::chrono::duration<double, std::ratio<1>>;
using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

class FFMPEGOutput {
public:
  FFMPEGOutput() = default;

  int Init(std::string name, int img_h, int img_w, int frame_rate = 20,
           int pic_fmt = AV_PIX_FMT_NV12);
  int Init(std::string name, int img_h, int img_w, AVRational frame_rate,
           int pic_fmt = AV_PIX_FMT_NV12);
  bool IsValid();

  void ShutDown() {}
  void Process(DeviceBufferPtr buffer);
  void Process(std::tuple<void *, uint32_t> buffer);

  void Wait4Stream();

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

  Duration interval;
  TimePoint last_sent_tp;

  bool output_is_file;

  bool valid{false};
  std::ofstream h264_of;
};

#endif //__FFMPEG_OUTPUT_H__
