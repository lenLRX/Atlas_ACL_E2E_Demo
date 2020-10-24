#include "rtmp_stream.h"
#include "util.h"
#include <iostream>

int RtmpContext::Init(std::string name, int img_h, int img_w, int frame_rate, int pic_fmt) {
  AVRational av_framerate;
  av_framerate.num = frame_rate;
  av_framerate.den = 1;
  return Init(name, img_h, img_w, av_framerate, pic_fmt);
}

int RtmpContext::Init(std::string name, int img_h, int img_w, AVRational frame_rate, int pic_fmt) {
  //std::string base_addr = "rtmp://127.0.0.1:1935/myapp/";
  stream_name = name;
  const char* output = stream_name.c_str();
  const char* profile = "high444";
  
  av_register_all();
  avformat_network_init();
  //av_log_set_level(AV_LOG_TRACE);
  int ret=0;

  encoder_avfc = NULL;
  ret = avformat_alloc_output_context2(&encoder_avfc, NULL, "flv", output);
  if (ret < 0) {
      std::cerr << "[RtmpContext::Init] avformat_alloc_output_context2 failed"<< ret << std::endl;
      return ret;
  }

  //encoder_avfc->flags |= AVFMT_FLAG_NOBUFFER;
  encoder_avfc->flags |= AVFMT_FLAG_FLUSH_PACKETS;

  if (!(encoder_avfc->oformat->flags & AVFMT_NOFILE)) {
      ret = avio_open2(&encoder_avfc->pb, output, AVIO_FLAG_WRITE, NULL, NULL);
      if (ret < 0) {
          std::cerr << "[RtmpContext::Init] avio_open2 failed" << std::endl;
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
  video_avcc->pix_fmt = (AVPixelFormat)pic_fmt;// AV_PIX_FMT_NV12;// NV12 IS YUV420
  // control rate
  video_avcc->bit_rate = 2 * 1000 * 1000;
  video_avcc->rc_buffer_size = 4 * 1000 * 1000;
  video_avcc->rc_max_rate = 2 * 1000 * 1000;
  video_avcc->rc_min_rate = 2.5 * 1000 * 1000;
  video_avcc->time_base.num = frame_rate.den;
  video_avcc->time_base.den = frame_rate.num;
  //video_avcc->gop_size = 0;

  if (encoder_avfc->oformat->flags & AVFMT_GLOBALHEADER) {
    video_avcc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  avs = avformat_new_stream(encoder_avfc, video_avc);

  ret = avcodec_parameters_from_context(avs->codecpar, video_avcc);

  if (ret < 0) {
    std::cerr << "[RtmpContext::Init] avcodec_parameters_from_context failed" << std::endl;
    return ret;
  }

  codec_options = NULL;
  av_dict_set(&codec_options, "profile", profile, 0);
  av_dict_set(&codec_options, "preset", "superfast", 0);
  av_dict_set(&codec_options, "tune", "zerolatency", 0);

  ret = avcodec_open2(video_avcc, video_avc, &codec_options);
  if (ret < 0) {
    std::cerr << "[RtmpContext::Init] avformat_new_stream failed" << std::endl;
    return ret;
  }

  avs->codecpar->extradata = video_avcc->extradata;
  avs->codecpar->extradata_size = video_avcc->extradata_size;

  av_dump_format(encoder_avfc, 0, output, 1);

  ret = avformat_write_header(encoder_avfc, NULL);
  if (ret < 0) {
    std::cerr << "[RtmpContext::init] avformat_write_header failed" << std::endl;
    return ret;
  }

  video_frame = av_frame_alloc();
  int frame_buf_size = av_image_get_buffer_size(video_avcc->pix_fmt, video_avcc->width, video_avcc->height, 1);
  std::cout << "[RtmpContext::Init] expected frame size " << frame_buf_size << std::endl;

  video_frame->width = video_avcc->width;
  video_frame->height = video_avcc->height;
  video_frame->format = video_avcc->pix_fmt;
  video_frame->pts = 1;

  valid = true;
  return 0;
}

bool RtmpContext::IsValid() {
    return valid;
}

void RtmpContext::SendFrame(const uint8_t* pdata) {
  PERF_TIMER();
  int ret = 0;
  av_image_fill_arrays(video_frame->data, video_frame->linesize,
    pdata,video_avcc->pix_fmt, video_avcc->width, video_avcc->height, 1);

  AVPacket pkt = {0};
    av_init_packet(&pkt);

    ret = avcodec_send_frame(video_avcc, video_frame);

    if (ret < 0) {
      std::cerr << "[RtmpContext::SendFrame] avcodec_send_frame failed" << std::endl;
      return;
    }

    ret = avcodec_receive_packet(video_avcc, &pkt);

    if (ret < 0) {
      std::cerr << "[RtmpContext::SendFrame] avcodec_receive_packet failed" << std::endl;
      return;
    }

    ret = av_interleaved_write_frame(encoder_avfc, &pkt);

    if (ret < 0) {
      std::cerr << "[RtmpContext::SendFrame] av_interleaved_write_frame failed" << std::endl;
      return;
    }

    av_packet_unref(&pkt);

  video_frame->pts += av_rescale_q(1, video_avcc->time_base, avs->time_base);
}

static void dontfree(void *opaque, uint8_t *data) {
  // tell ffmpeg dont free data
}

void RtmpContext::SendEncodedFrame(void* pdata, int size) {
  int ret = 0;
    AVPacket pkt = {0};
    av_init_packet(&pkt);

    pkt.pts = video_frame->pts;
    pkt.dts = pkt.pts;
    pkt.flags = AV_PKT_FLAG_KEY;
    //av_packet_from_data(&pkt, (uint8_t*)pdata, size);

    pkt.buf = av_buffer_create((uint8_t*)pdata, size + AV_INPUT_BUFFER_PADDING_SIZE,
                                dontfree, NULL, 0);
  
    pkt.data = (uint8_t*)pdata;
    pkt.size = size;

    ret = av_write_frame(encoder_avfc, &pkt);
    
    if (ret < 0) {
      std::cerr << "[RtmpContext::SendFrame] av_interleaved_write_frame failed" << std::endl;
      return;
    }

    //pkt.buf = nullptr;

    av_packet_unref(&pkt);

    video_frame->pts += av_rescale_q(1, video_avcc->time_base, avs->time_base);
  std::cerr << "[RtmpContext::SendFrame] Sent Done" << std::endl;
}

