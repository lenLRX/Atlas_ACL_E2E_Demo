#include "ffmpeg_output.h"
#include "app_profiler.h"
#include "util.h"

#include <fstream>
#include <iostream>
#include <thread>

#include <x264.h>

/*
  Parse format from name
  rtmp   -> "flv"
  rtsp   -> "rtsp"
  mp4    -> ""
  other  -> ""
*/
static std::string GuessFormatFromName(const std::string &name) {
  std::string format;
  if (name.find("rtmp:") == 0) {
    format = "flv";
  } else if (name.find("rtsp:") == 0) {
    format = "rtsp";
  }
  return format;
}

int FFMPEGOutput::Init(std::string name, int img_h, int img_w, int frame_rate,
                       int pic_fmt) {
  AVRational av_framerate;
  av_framerate.num = frame_rate;
  av_framerate.den = 1;
  return Init(name, img_h, img_w, av_framerate, pic_fmt);
}

int FFMPEGOutput::Init(std::string name, int img_h, int img_w,
                       AVRational frame_rate, int pic_fmt) {
  last_sent_tp = std::chrono::steady_clock::now();
  interval = Duration((double)frame_rate.den / frame_rate.num);
  std::cout << "FFMPEGOutput::Init frame send interval: " << interval.count()
            << std::endl;
  stream_name = name;
  const char *output = stream_name.c_str();
  const char *profile = "high444";

  h264_of = std::ofstream("test264.h264", std::ios::binary);

  std::string format = GuessFormatFromName(name);

  av_register_all();
  avformat_network_init();
  // av_log_set_level(AV_LOG_TRACE);
  int ret = 0;

  encoder_avfc = NULL;
  ret = avformat_alloc_output_context2(
      &encoder_avfc, NULL, format.empty() ? NULL : format.c_str(), output);
  if (ret < 0) {
    std::cerr << "[FFMPEGOutput::Init] avformat_alloc_output_context2 failed"
              << ret << std::endl;
    return ret;
  }

  // encoder_avfc->flags |= AVFMT_FLAG_NOBUFFER;
  encoder_avfc->flags |= AVFMT_FLAG_FLUSH_PACKETS;

  if (!(encoder_avfc->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open2(&encoder_avfc->pb, output, AVIO_FLAG_WRITE, NULL, NULL);
    if (ret < 0) {
      char err_buf[AV_ERROR_MAX_STRING_SIZE] = {0};
      std::cerr << "[FFMPEGOutput::Init] avio_open2 failed err code: " << ret
                << " Reason: "
                << av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, ret)
                << std::endl;
      return ret;
    }
  }

  video_avc = avcodec_find_encoder(AV_CODEC_ID_H264);

  encoder_avfc->video_codec = video_avc;
  encoder_avfc->video_codec_id = AV_CODEC_ID_H264;

  video_avcc = avcodec_alloc_context3(video_avc);

  video_avcc->codec_tag = 0;
  video_avcc->codec_id = AV_CODEC_ID_H264;
  video_avcc->codec_type = AVMEDIA_TYPE_VIDEO;
  video_avcc->gop_size = 12;
  video_avcc->height = img_h;
  video_avcc->width = img_w;
  video_avcc->pix_fmt =
      (AVPixelFormat)pic_fmt; // AV_PIX_FMT_NV12;// NV12 IS YUV420
  // control rate
  video_avcc->bit_rate = 0;
  video_avcc->rc_buffer_size = 0;
  video_avcc->rc_max_rate = 0;
  video_avcc->rc_min_rate = 0;
  video_avcc->time_base.num = frame_rate.den;
  video_avcc->time_base.den = frame_rate.num;
  // video_avcc->gop_size = 0;

  if (encoder_avfc->oformat->flags & AVFMT_GLOBALHEADER) {
    video_avcc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  avs = avformat_new_stream(encoder_avfc, video_avc);

  ret = avcodec_parameters_from_context(avs->codecpar, video_avcc);
  avs->avg_frame_rate = frame_rate;

  if (ret < 0) {
    std::cerr << "[FFMPEGOutput::Init] avcodec_parameters_from_context failed"
              << std::endl;
    return ret;
  }

  codec_options = NULL;
  av_dict_set(&codec_options, "profile", profile, 0);
  av_dict_set(&codec_options, "preset", "superfast", 0);
  av_dict_set(&codec_options, "tune", "zerolatency", 0);

  ret = avcodec_open2(video_avcc, video_avc, &codec_options);
  if (ret < 0) {
    std::cerr << "[FFMPEGOutput::Init] avformat_new_stream failed" << std::endl;
    char err_buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    std::cerr << "[FFMPEGInput::Init] err string: "
              << av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, ret)
              << std::endl;
    return ret;
  }

  avs->codecpar->extradata = video_avcc->extradata;
  avs->codecpar->extradata_size = video_avcc->extradata_size;

  av_dump_format(encoder_avfc, 0, output, 1);

  AVDictionary* mux_options = NULL;
  
  av_dict_set_int(&mux_options, "no_metadata", 1, 0);
  av_dict_set_int(&mux_options, "add_keyframe_index", 1, 0);
  av_dict_set_int(&mux_options, "no_duration_filesize", 1, 0);

  ret = avformat_write_header(encoder_avfc, &mux_options);
  if (ret < 0) {
    std::cerr << "[FFMPEGOutput::init] avformat_write_header failed"
              << std::endl;
    return ret;
  }

  video_frame = av_frame_alloc();
  int frame_buf_size = av_image_get_buffer_size(
      video_avcc->pix_fmt, video_avcc->width, video_avcc->height, 1);
  std::cout << "[FFMPEGOutput::Init] expected frame size " << frame_buf_size
            << std::endl;

  video_frame->width = video_avcc->width;
  video_frame->height = video_avcc->height;
  video_frame->format = video_avcc->pix_fmt;
  video_frame->pts = 0;

  valid = true;
  std::ifstream test_f(name.c_str());
  output_is_file = test_f.good();
  return 0;
}

bool FFMPEGOutput::IsValid() { return valid; }

void FFMPEGOutput::Process(DeviceBufferPtr buffer) {
  SendFrame((const uint8_t *)buffer->GetHostPtr());
}

void FFMPEGOutput::Wait4Stream() {
  if (!output_is_file) {
    // if output is not file, e.g. RTSP or RTMP
    // we must not send too fast
    auto now = std::chrono::steady_clock::now();
    auto dt = now - last_sent_tp;
    auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(dt);
    auto dt_sec = std::chrono::duration_cast<Duration>(dt_us);
    auto time_to_sleep = interval - dt_sec;
    if (time_to_sleep.count() > 0) {
      std::this_thread::sleep_for(time_to_sleep);
    }
    last_sent_tp = std::chrono::steady_clock::now();
  }
}

void FFMPEGOutput::SendFrame(const uint8_t *pdata) {
  Wait4Stream();
  APP_PROFILE(FFMPEGOutput::SendFrame);
  int ret = 0;
  av_image_fill_arrays(video_frame->data, video_frame->linesize, pdata,
                       video_avcc->pix_fmt, video_avcc->width,
                       video_avcc->height, 1);

  ret = avcodec_send_frame(video_avcc, video_frame);

  if (ret < 0) {
    std::cerr << "[FFMPEGOutput::SendFrame] avcodec_send_frame failed"
              << std::endl;
    return;
  }

  while (true) {
    AVPacket pkt = {0};
    av_init_packet(&pkt);
    ret = avcodec_receive_packet(video_avcc, &pkt);

    if (ret < 0) {
      // std::cerr << "[FFMPEGOutput::SendFrame] avcodec_receive_packet failed"
      //          << std::endl;
      av_packet_unref(&pkt);
      return;
    }

    ret = av_interleaved_write_frame(encoder_avfc, &pkt);

    if (ret < 0) {
      std::cerr << "[FFMPEGOutput::SendFrame] av_interleaved_write_frame failed"
                << std::endl;
      return;
    }

    av_packet_unref(&pkt);

    video_frame->pts += av_rescale_q(1, video_avcc->time_base, avs->time_base);
  }
  // std::cerr << "[FFMPEGOutput::SendFrame] End" << std::endl;
}

static void dontfree(void *opaque, uint8_t *data) {
  // tell ffmpeg dont free data
}

static void custom_free(void *opaque, uint8_t *data) { free(data); }

void FFMPEGOutput::Process(std::tuple<void *, uint32_t> buffer) {
  SendEncodedFrame(std::get<0>(buffer), std::get<1>(buffer));
}

enum NALUType {
  UNSPECIFIED = 0,
  SLICE_NON_IDR = 1,
  SLICE_DPA = 2,
  SLICE_DPB = 3,
  SLICE_DPC = 4,
  SLICE_IDR = 5,
  SLICE_SEI = 6,
  SPS = 7,
  PPS = 8
};



static AVPacket* process_nal(void* pdata, int size) {
  AVPacket* packet = av_packet_alloc();
  std::vector<int> start_offsets;
  start_offsets.reserve(4);
  std::vector<NALUType> nalu_types;
  nalu_types.reserve(4);
  std::vector<int> end_offsets;
  end_offsets.reserve(4);
  std::vector<int> sizes;
  sizes.reserve(4);

  uint8_t* u8_ptr = (uint8_t*)pdata;
  int prev_offset = 0;
  for (int offset = 0;offset < size - 4; ++offset) {
    uint8_t* off_ptr = u8_ptr + offset;
    if (off_ptr[0] == 0 && off_ptr[1] == 0 && off_ptr[2] == 0 && off_ptr[3] == 1) {
      NALUType ty = (NALUType)(off_ptr[4]&0x1f);
      nalu_types.push_back(ty);
      start_offsets.push_back(offset);
      std::cout << "found start code @" << offset << " NAL type: " << (off_ptr[4]&0x1f) << std::endl;
    }
  }

  int nal_size = start_offsets.size();
  for (int i = 1;i < nal_size; ++i) {
    end_offsets.push_back(start_offsets[i]);
  }
  end_offsets.push_back(size);

  for (int i = 0;i < nal_size; ++i) {
    sizes.push_back(end_offsets[i] - start_offsets[i]);
  }

  int frame_no = 0;
  if (nalu_types[0] == SPS && nalu_types[1] == PPS) {
    std::cout << "write SPS and PPS" << std::endl;
    frame_no = 2;
    int extra_size = sizes[0] + sizes[1];
    uint8_t* new_extra_data = (uint8_t*)av_malloc(extra_size);
    memcpy(new_extra_data, u8_ptr, extra_size);

    av_packet_add_side_data(packet, AV_PKT_DATA_NEW_EXTRADATA, new_extra_data, extra_size);
  }

  int pkt_data_size = sizes[frame_no];
  uint8_t* u8_pkt_data = (uint8_t*)av_malloc(pkt_data_size + AV_INPUT_BUFFER_PADDING_SIZE);
  memcpy(u8_pkt_data, u8_ptr + start_offsets[frame_no], sizes[frame_no]);
  av_packet_from_data(packet, u8_pkt_data, pkt_data_size);

  if (nalu_types[frame_no] == SLICE_IDR) {
    packet->flags |= AV_PKT_FLAG_KEY;
  }

  free(pdata);

  return packet;

}

void FFMPEGOutput::SendEncodedFrame(void *pdata, int size) {
  //std::cout << "h264 writting at " << h264_of.tellp() << " size " << size << std::endl;
  //h264_of.write((const char*)pdata, size);
  Wait4Stream();
  APP_PROFILE(FFMPEGOutput::SendEncodedFrame);
  int ret = 0;
  AVPacket* pkt = process_nal(pdata, size);

  pkt->pts = video_frame->pts;
  pkt->dts = pkt->pts;

  ret = av_write_frame(encoder_avfc, pkt);

  if (ret < 0) {
    std::cerr << "[FFMPEGOutput::SendFrame] av_interleaved_write_frame failed"
              << std::endl;
    return;
  }

  av_packet_free(&pkt);

  video_frame->pts += av_rescale_q(1, video_avcc->time_base, avs->time_base);
  std::cout << "curr pts: " << video_frame->pts << std::endl;
}

void FFMPEGOutput::Close() {
  if (encoder_avfc) {
    av_write_trailer(encoder_avfc);
    if (!(encoder_avfc->flags & AVFMT_NOFILE)) {
      avio_closep(&encoder_avfc->pb);
    }
  }
}
