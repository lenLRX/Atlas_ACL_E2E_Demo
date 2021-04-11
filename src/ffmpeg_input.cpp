#include "ffmpeg_input.h"
#include "util.h"
#include "app_profiler.h"

#include <chrono>
#include <thread>

using namespace std::chrono_literals;

int FFMPEGInput::Init(const std::string &addr) {
  av_register_all();
  avformat_network_init();
  // av_log_set_level(AV_LOG_TRACE);

  av_fc = avformat_alloc_context();

  AVDictionary *avdic = nullptr;

  // av_dict_set(&avdic, "rtsp_transport", "tcp", 0);
  // av_dict_set(&avdic, "max_delay", "100", 0);

  int ret = avformat_open_input(&av_fc, addr.c_str(), nullptr, &avdic);

  if (ret != 0) {
    std::cerr << "can't open input: " << addr << " err code: " << ret
              << std::endl;
    return -1;
  }

  ret = avformat_find_stream_info(av_fc, nullptr);

  if (ret < 0) {
    std::cerr << "can't find stream info, err code:" << ret << std::endl;
    return -1;
  }

  video_stream = -1;

  for (unsigned int i = 0; i < av_fc->nb_streams; ++i) {
    if (av_fc->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream = i;
      break;
    }
  }

  if (video_stream < 0) {
    std::cerr << "failed to find a video stream" << std::endl;
    return -1;
  }

  av_cc = av_fc->streams[video_stream]->codec;

  // av_read_play(av_fc);

  std::cout << "[FFMPEGInput::Init] " << addr
            << " codec name:" << avcodec_get_name(av_cc->codec_id) << std::endl;
  std::cout << "avcc profile: " << av_cc->profile << std::endl;
  std::cout << "frame h: " << av_cc->height << " frame w: " << av_cc->width
            << std::endl;
  std::cout << "ticks_per_frame: " << av_cc->ticks_per_frame << std::endl;
  std::cout << "framerate.num: " << av_cc->framerate.num << std::endl;
  std::cout << "framerate.den: " << av_cc->framerate.den << std::endl;
  std::cout << "ref frame num: " << av_cc->refs << std::endl;
  std::cout << "has B frame: " << av_cc->has_b_frames << std::endl;
  std::cout << "pix format: " << av_cc->pix_fmt << std::endl;
  std::cout << "codec_tag " << av_cc->codec_tag << std::endl;
  std::cout << "extra_data size: " << av_cc->extradata_size << std::endl;

  if (true || av_cc->codec_tag == AVC1_TAG) {
    bsf_filter = av_bsf_get_by_name("h264_mp4toannexb");
    ret = av_bsf_alloc(bsf_filter, &bsfc);
    if (ret < 0) {
      std::cout << "failed to init bsfc" << std::endl;
      return -1;
    }
    avcodec_parameters_copy(bsfc->par_in,
                            av_fc->streams[video_stream]->codecpar);
    av_bsf_init(bsfc);
    need_bsf = true;
  }

  auto decoder_codec = avcodec_find_decoder(av_cc->codec_id);
  decoder_context = avcodec_alloc_context3(decoder_codec);
  ret = avcodec_parameters_to_context(decoder_context,
                                      av_fc->streams[video_stream]->codecpar);
  if (ret < 0) {
    std::cout << "failed to avcodec_parameters_to_context" << std::endl;
  }
  if (avcodec_open2(decoder_context, decoder_codec, nullptr) < 0) {
    std::cout << "failed to open" << std::endl;
  }

  return 0;
}

int FFMPEGInput::GetHeight() {
  if (av_cc == nullptr) {
    throw std::runtime_error("FFMPEGInput Stream is not Inited!");
  }
  return av_cc->height;
}

int FFMPEGInput::GetWidth() {
  if (av_cc == nullptr) {
    throw std::runtime_error("FFMPEGInput Stream is not Inited!");
  }
  return av_cc->width;
}

acldvppStreamFormat FFMPEGInput::GetProfile() {
  if (av_cc == nullptr) {
    throw std::runtime_error("FFMPEGInput Stream is not Inited!");
  }
  return h264_ffmpeg_profile_to_acl_stream_fromat(av_cc->profile);
}

AVRational FFMPEGInput::GetFramerate() {
  if (av_cc == nullptr) {
    throw std::runtime_error("FFMPEGInput Stream is not Inited!");
  }
  return av_cc->framerate;
}

void FFMPEGInput::Run() {
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = av_cc->extradata;
  packet.size = av_cc->extradata_size;
  output_queue->push(packet);
  if (need_bsf) {
    while (ReceivePacketWithBSF())
      ;
  } else {
    while (ReceiveSinglePacket())
      ;
  }
}

bool FFMPEGInput::ReceiveSinglePacket() {
  APP_PROFILE(FFMPEGInput::ReceiveSinglePacket);
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = nullptr;
  packet.size = 0;
  int ret = av_read_frame(av_fc, &packet);
  // std::cout << "ret: " << ret << "packet stream: "
  //    << packet.stream_index << " video stream: " << video_stream <<
  //    std::endl;
  if (ret < 0) {
    char err_buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    std::cerr << "[FFMPEGInput::ReceiveSinglePacket] err string: "
              << av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, ret)
              << std::endl;
    av_packet_unref(&packet);
    return false;
  }
  if (packet.stream_index == video_stream) {
    output_queue->push(packet);
  }
  return true;
}

bool FFMPEGInput::ReceivePacketWithBSF() {
  auto perf_obj = AppProfileGuard("FFMPEGInput::ReceivePacketWithBSF" , __FILE__, __LINE__, false);
  perf_obj.AddBeginRecord();
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = nullptr;
  packet.size = 0;
  int ret = av_read_frame(av_fc, &packet);
  if (ret < 0) {
    char err_buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    std::cerr << "[FFMPEGInput::ReceivePacketWithBSF] err string: "
              << av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, ret)
              << std::endl;
    av_packet_unref(&packet);
    perf_obj.AddEndRecord();
    return false;
  }
  if (packet.stream_index == video_stream) {
    ret = av_bsf_send_packet(bsfc, &packet);
    if (ret < 0) {
      std::cout << "av_bsf_send_packet failed" << std::endl;
      perf_obj.AddEndRecord();
      return false;
    }
    AVPacket filtered_packet;
    av_init_packet(&filtered_packet);
    filtered_packet.data = nullptr;
    filtered_packet.size = 0;
    while (av_bsf_receive_packet(bsfc, &filtered_packet) == 0) {
      perf_obj.AddEndRecord();
      output_queue->push(filtered_packet);
      perf_obj.AddBeginRecord();
    }
  }
  av_packet_unref(&packet);
  perf_obj.AddEndRecord();
  return true;
}

void FFMPEGInput::SetOutputQueue(ThreadSafeQueueWithCapacity<AVPacket>* queue) {
  output_queue = queue;
}
