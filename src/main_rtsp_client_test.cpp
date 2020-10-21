#include "rtsp_input.h"
#include "dvpp_decoder.h"
#include "vpc_resize.h"
#include "rtmp_stream.h"

#include "acl_cb_thread.h"

const static int yolov3_model_size = 416;

int main(int argc, char** argv) {
    CHECK_ACL(aclrtSetDevice(0));
    AclCallBackThread cb_thread;

    aclrtContext ctx;
    CHECK_ACL(aclrtCreateContext(&ctx, 0));
    CHECK_ACL(aclrtSetCurrentContext(ctx));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    CHECK_ACL(aclrtSubscribeReport(cb_thread.GetPid(), stream));

    RTSPInput rtsp_input;
    rtsp_input.Init("rtsp://192.168.1.9:8554/tt.mp4");

    RtmpContext resized_ctx;
    resized_ctx.Init("resize", yolov3_model_size, yolov3_model_size);

    DvppDecoder decoder;
    decoder.Init(cb_thread.GetPid(), rtsp_input.GetHeight(), rtsp_input.GetWidth());
    decoder.SetDeviceCtx(&ctx);

    VPCResizeEngine resize_engine(stream);
    resize_engine.Init(rtsp_input.GetHeight(), rtsp_input.GetWidth(), yolov3_model_size, yolov3_model_size);

    resize_engine.RegisterHandler([&](uint8_t* buffer){resized_ctx.SendFrame(buffer);});

    decoder.RegisterHandler([&](uint8_t* buffer){resize_engine.Resize(buffer);});

    rtsp_input.RegisterHandler([&](AVPacket* packet){decoder.SendFrame(packet);});
    rtsp_input.Run();
    return 0;
}
