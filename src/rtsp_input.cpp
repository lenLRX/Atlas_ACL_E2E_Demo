#include "rtsp_input.h"

int RTSPInput::Init(const std::string& addr) {
    av_register_all();
    avformat_network_init();
    av_log_set_level(AV_LOG_TRACE);

    av_fc = avformat_alloc_context();

    AVDictionary* avdic = nullptr;
    
    av_dict_set(&avdic, "rtsp_transport", "udp", 0);
    av_dict_set(&avdic, "max_delay", "100", 0);

    int ret = avformat_open_input(&av_fc, addr.c_str(), nullptr, &avdic);

    if (ret != 0) {
        std::cerr << "can't open input: " << addr << " err code: " << ret << std::endl;
        return -1;
    }

    ret = avformat_find_stream_info(av_fc, nullptr);

    if (ret < 0) {
        std::cerr << "can't find stream info, err code:" << ret << std::endl;
        return -1;
    }

    video_stream = -1;

    for (unsigned int i = 0;i < av_fc->nb_streams; ++i) {
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

    av_read_play(av_fc);

    std::cout << "[RTSPInput::Init] " << addr << " codec name:" << avcodec_get_name(av_cc->codec_id) << std::endl;
    std::cout << "avcc profile: " << av_cc->profile << std::endl;
    std::cout << "ref frame num: " << av_cc->refs << std::endl;
    std::cout << "has B frame: " << av_cc->has_b_frames << std::endl;
    
    return 0;
}

void RTSPInput::RegisterHandler(std::function<void(AVPacket*)> handler) {
    packet_handler = handler;
}

int RTSPInput::GetHeight() {
    if (av_cc == nullptr) {
        throw std::runtime_error("RTSP Stream is not Inited!");
    }
    return av_cc->height;
}

int RTSPInput::GetWidth() {
    if (av_cc == nullptr) {
        throw std::runtime_error("RTSP Stream is not Inited!");
    }
    return av_cc->width;
}


void RTSPInput::Run() {
    while (ReceiveSinglePacket());
}

bool RTSPInput::ReceiveSinglePacket() {
    AVPacket packet;
    av_init_packet(&packet);
    int ret = av_read_frame(av_fc, &packet);
    //std::cout << "ret: " << ret << "packet stream: " 
    //    << packet.stream_index << " video stream: " << video_stream << std::endl;
    if (ret < 0) {
        char err_buf[AV_ERROR_MAX_STRING_SIZE] = {0};
        std::cerr << "[RTSPInput::ReceiveSinglePacket] err string: " << av_make_error_string(err_buf, AV_ERROR_MAX_STRING_SIZE, ret) << std::endl;
        av_packet_unref(&packet);
        return false;
    }
    if (packet.stream_index == video_stream) {
        packet_handler(&packet);
    }
    av_packet_unref(&packet);
    return true;
}
