#include "dvpp_decoder.h"

static void DvppDecCallback(acldvppStreamDesc* input, acldvppPicDesc* output, void* user_data) {
    std::cout << "DvppDecCallback Enter" << std::endl;
    delete[] acldvppGetPicDescData(output);
    av_packet_unref((AVPacket*)user_data);
    delete user_data;
    acldvppDestroyPicDesc(output);
    acldvppDestroyStreamDesc(input);
}

aclError DvppDecoder::Init(const pthread_t thread_id, int h, int w) {
    height = h;
    width = w;
    channel_desc = aclvdecCreateChannelDesc();
    CHECK_ACL(aclvdecSetChannelDescChannelId(channel_desc, 0));
    CHECK_ACL(aclvdecSetChannelDescThreadId(channel_desc, thread_id));
    CHECK_ACL(aclvdecSetChannelDescCallback(channel_desc, &DvppDecCallback));
    CHECK_ACL(aclvdecSetChannelDescEnType(channel_desc, H264_HIGH_LEVEL));
    CHECK_ACL(aclvdecSetChannelDescOutPicFormat(channel_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(aclvdecSetChannelDescOutPicWidth(channel_desc, width));
    CHECK_ACL(aclvdecSetChannelDescOutPicHeight(channel_desc, height));
    CHECK_ACL(aclvdecSetChannelDescRefFrameNum(channel_desc, 1));
    CHECK_ACL(aclvdecSetChannelDescOutMode(channel_desc, 0));
    aclvdecCreateChannel(channel_desc);
    
}

aclError DvppDecoder::SendFrame(AVPacket* packet) {
    std::cout << "DvppDecoder::SendFrame Enter" << std::endl;
    AVPacket* frame_packet = new AVPacket();
    av_packet_ref(frame_packet, packet);
    std::cout << "av_packet_ref done" << std::endl;

    acldvppStreamDesc* stream_desc = acldvppCreateStreamDesc();
    CHECK_ACL(acldvppSetStreamDescData(stream_desc, frame_packet->data));
    CHECK_ACL(acldvppSetStreamDescSize(stream_desc, frame_packet->size));
    
    acldvppPicDesc* output = acldvppCreatePicDesc();

    size_t output_size = (width * height * 3) / 2;

    uint8_t* output_buffer = new uint8_t[output_size];

    acldvppSetPicDescData(output, output_buffer);
    acldvppSetPicDescSize(output, output_size);
    acldvppSetPicDescFormat(output, PIXEL_FORMAT_YUV_SEMIPLANAR_420);

    CHECK_ACL(aclvdecSendFrame(channel_desc, stream_desc, output, nullptr, frame_packet));

    std::cout << "DvppDecoder::SendFrame Exit" << std::endl;
    return ACL_ERROR_NONE;
}

