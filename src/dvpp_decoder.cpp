#include "dvpp_decoder.h"

#include <atomic>

class DecoderContext {
public:
    DecoderContext(DvppDecoder* decoder, AVPacket* pkt):
        decoder(decoder), pkt(pkt) {}
    DvppDecoder* decoder;
    AVPacket* pkt;
};

static void DvppDecCallback(acldvppStreamDesc* input, acldvppPicDesc* output, void* user_data) {
    DecoderContext* ctx = (DecoderContext*)user_data;
    aclrtSetCurrentContext(*ctx->decoder->GetDeviceCtx());
    uint8_t* output_buffer = (uint8_t*)acldvppGetPicDescData(output);
    //std::cout << "DvppDecCallback Enter" << std::endl;
    ctx->decoder->GetHandler()(output_buffer);
    //std::cout << "DvppDecCallback Exit" << std::endl;
    delete[] output_buffer;
    av_packet_unref((AVPacket*)ctx->pkt);
    delete ctx->pkt;
    delete ctx;
    acldvppDestroyPicDesc(output);
    acldvppDestroyStreamDesc(input);
}

aclError DvppDecoder::Init(const pthread_t thread_id, int h, int w, acldvppStreamFormat profile) {
    height = h;
    width = w;
    // YUV420SP
    output_size = (width * height * 3) / 2;
    std::cout << "[DvppDecoder::Init] h: " << h
        << " w:" << w << "output_size: " << output_size << std::endl;
    channel_desc = aclvdecCreateChannelDesc();
    CHECK_ACL(aclvdecSetChannelDescChannelId(channel_desc, GetChannelId()));
    CHECK_ACL(aclvdecSetChannelDescThreadId(channel_desc, thread_id));
    CHECK_ACL(aclvdecSetChannelDescCallback(channel_desc, &DvppDecCallback));
    CHECK_ACL(aclvdecSetChannelDescEnType(channel_desc, profile));
    CHECK_ACL(aclvdecSetChannelDescOutPicFormat(channel_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(aclvdecSetChannelDescOutPicWidth(channel_desc, width));
    CHECK_ACL(aclvdecSetChannelDescOutPicHeight(channel_desc, height));
    CHECK_ACL(aclvdecSetChannelDescRefFrameNum(channel_desc, 1));
    CHECK_ACL(aclvdecSetChannelDescOutMode(channel_desc, 0));
    aclvdecCreateChannel(channel_desc);
}

int DvppDecoder::GetChannelId() {
    static std::atomic_int channel_id(0);
    return channel_id++;
}

DvppDecoder::~DvppDecoder() {
    
}


void DvppDecoder::Destory() {
    std::cout << "DvppDecoder::~DvppDecoder Start" << std::endl;

    aclvdecDestroyChannel(channel_desc);
    std::cout << "DvppDecoder::~DvppDecoder DestroyChannel Done" << std::endl;
    //aclvdecDestroyChannelDesc(channel_desc);
    //aclvdecDestroyFrameConfig(frame_config);
    std::cout << "DvppDecoder::~DvppDecoder End" << std::endl;
}

aclError DvppDecoder::SendFrame(AVPacket* packet) {
    //std::cout << "DvppDecoder::SendFrame Enter" << std::endl;
    AVPacket* frame_packet = new AVPacket();
    av_packet_ref(frame_packet, packet);

    acldvppStreamDesc* stream_desc = acldvppCreateStreamDesc();
    CHECK_ACL(acldvppSetStreamDescData(stream_desc, frame_packet->data));
    CHECK_ACL(acldvppSetStreamDescSize(stream_desc, frame_packet->size));
    
    acldvppPicDesc* output = acldvppCreatePicDesc();

    uint8_t* output_buffer = new uint8_t[output_size];

    acldvppSetPicDescData(output, output_buffer);
    acldvppSetPicDescSize(output, output_size);
    acldvppSetPicDescFormat(output, PIXEL_FORMAT_YUV_SEMIPLANAR_420);

    DecoderContext* ctx = new DecoderContext(this, frame_packet);
    CHECK_ACL(aclvdecSendFrame(channel_desc, stream_desc, output, nullptr, ctx));

    //std::cout << "DvppDecoder::SendFrame Exit" << std::endl;
    return ACL_ERROR_NONE;
}

void DvppDecoder::RegisterHandler(std::function<void(uint8_t*)> handler) {
    buffer_handler = handler;
}

const std::function<void(uint8_t*)>& DvppDecoder::GetHandler() {
    return buffer_handler;
}

void DvppDecoder::SetDeviceCtx(aclrtContext* ctx) {
    dev_ctx = ctx;
}

aclrtContext* DvppDecoder::GetDeviceCtx() {
    return dev_ctx;
}

int DvppDecoder::GetHeight() {
    return height;
}

int DvppDecoder::GetWidth() {
    return width;
}

int DvppDecoder::GetOutputBufferSize() {
    return output_size;
}