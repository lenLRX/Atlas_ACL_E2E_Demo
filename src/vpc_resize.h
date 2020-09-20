#ifndef __VPC_RESIZE_H__
#define __VPC_RESIZE_H__
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"

#include "util.h"


class VPCResizeEngine {
public:
    VPCResizeEngine(aclrtStream stream);
    //~VPCResizeEngine();

    aclError Init(int src_h, int src_w, int dst_h, int dst_w);

    aclError Resize(const uint8_t* pdata, int size);
    uint8_t* GetOutputBuffer();
private:
    acldvppChannelDesc* channel_desc;
    acldvppPicDesc* input_desc;
    acldvppPicDesc* output_desc;
    acldvppResizeConfig* resize_config;
    aclrtStream stream;

    void* dvpp_input_mem;
    void* dvpp_output_mem;
    uint8_t* host_output_mem;
    int output_buffer_size;
    int input_buffer_size;
};

#endif//__VPC_RESIZE_H__