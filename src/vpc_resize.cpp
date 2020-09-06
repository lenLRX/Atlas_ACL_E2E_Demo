#include "vpc_resize.h"
#include "util.h"
#include <string.h>

int align_up(int size, int align) {
    return (size + (align - 1)) / align * align;
}

int yuv420sp_size(int h, int w) {
    return (h * w * 3) / 2;
}

VPCResizeEngine::VPCResizeEngine(aclrtStream stream):stream(stream) {
    channel_desc  = acldvppCreateChannelDesc();
    input_desc = acldvppCreatePicDesc();
    output_desc = acldvppCreatePicDesc();
    resize_config = acldvppCreateResizeConfig();
    acldvppSetResizeConfigInterpolation(resize_config, 0);
}

aclError VPCResizeEngine::Init(int src_h, int src_w, int dst_h, int dst_w) {
    CHECK_ACL(acldvppCreateChannel(channel_desc));
    input_buffer_size = yuv420sp_size(align_up(src_h, 16), align_up(src_w, 128));
    CHECK_ACL(acldvppMalloc(&dvpp_input_mem, input_buffer_size));
    output_buffer_size = yuv420sp_size(dst_h, dst_w);
    CHECK_ACL(acldvppMalloc(&dvpp_output_mem, output_buffer_size));
    
    host_output_mem = new uint8_t[output_buffer_size];

    CHECK_ACL(acldvppSetPicDescData(input_desc, dvpp_input_mem));
    CHECK_ACL(acldvppSetPicDescFormat(input_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(acldvppSetPicDescWidth(input_desc, src_w));
    CHECK_ACL(acldvppSetPicDescHeight(input_desc, src_h));
    CHECK_ACL(acldvppSetPicDescWidthStride(input_desc, align_up(src_w, 128)));
    CHECK_ACL(acldvppSetPicDescHeightStride(input_desc, align_up(src_h, 16)));
    CHECK_ACL(acldvppSetPicDescSize(input_desc, input_buffer_size));

    CHECK_ACL(acldvppSetPicDescData(output_desc, dvpp_output_mem));
    CHECK_ACL(acldvppSetPicDescFormat(output_desc, PIXEL_FORMAT_YUV_SEMIPLANAR_420));
    CHECK_ACL(acldvppSetPicDescWidth(output_desc, dst_w));
    CHECK_ACL(acldvppSetPicDescHeight(output_desc, dst_h));
    CHECK_ACL(acldvppSetPicDescWidthStride(output_desc, dst_w));
    CHECK_ACL(acldvppSetPicDescHeightStride(output_desc, dst_h));
    CHECK_ACL(acldvppSetPicDescSize(output_desc, output_buffer_size));

}

aclError VPCResizeEngine::Resize(const uint8_t* pdata, int size) {
    //std::cerr << "input_buffer_size: " << input_buffer_size << std::endl;
    memcpy(dvpp_input_mem, pdata, input_buffer_size);
    CHECK_ACL(acldvppVpcResizeAsync(channel_desc, input_desc,
        output_desc, resize_config, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    return ACL_ERROR_NONE;
}

uint8_t* VPCResizeEngine::GetOutputBuffer() {
    return (uint8_t*)dvpp_output_mem;
}