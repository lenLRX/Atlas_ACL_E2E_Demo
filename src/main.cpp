extern "C" {
#include "peripheral_api.h"
}

#include <unistd.h>
#include <iostream>

#include "acl/acl.h"

#include "rtmp_stream.h"
#include "vpc_resize.h"
#include "acl_model.h"
#include "util.h"
#include "yolov3_post.h"

const static int yolov3_model_size = 416;

class CameraCtx {
public:
    VPCResizeEngine* resize;
    RtmpContext* rtmp;
    RtmpContext* resize_rtmp;
    ACLModel* model;
    aclrtContext* dev_ctx;
};


int CameraCallBack(const void* pdata, int size, void* param) {
    //std::cerr << "CameraCallBack size: " << size << std::endl;
    CameraCtx* ctx = (CameraCtx*)param;
    CHECK_ACL(aclrtSetCurrentContext(*(ctx->dev_ctx)));
    {
        PERF_TIMER();
        ctx->rtmp->SendFrame((const uint8_t*)pdata);
    }
    {
        PERF_TIMER();
        CHECK_ACL(ctx->resize->Resize((const uint8_t*)pdata));
    }
    
    const uint8_t* resized_buffer = (const uint8_t*)ctx->resize->GetOutputBuffer();
    {
        PERF_TIMER();
        ctx->resize_rtmp->SendFrame(resized_buffer);
    }
    
    const auto& input_buffers = ctx->model->GetInputBuffer();
    memcpy(input_buffers[0], resized_buffer, ctx->model->GetInputBufferSizes()[0]);
    {
        PERF_TIMER();
        ctx->model->Infer();
    }
    
    {
        PERF_TIMER();
        yolov3_post(0.45, ctx->model->GetOutputBuffer(), ctx->model->GetOutputBufferSizes(),
            yolov3_model_size, yolov3_model_size);
    }
    
    return 1;
}

int main(int argc, char** argv) {
    CHECK_ACL(aclInit(nullptr));
    int ret;
    ret = MediaLibInit();
    if (ret != LIBMEDIA_STATUS_OK) {
        std::cerr << "MediaLibInit failed " << ret << std::endl;
        return -1;
    }

    ret = IsChipAlive(NULL);

    if (ret != LIBMEDIA_STATUS_OK) {
        std::cerr << "IsChipAlive failed" << std::endl;
        return -1;
    }

    ret = OpenCamera(0);

    if (ret != LIBMEDIA_STATUS_OK) {
        std::cerr << "OpenCamera 0 failed" << std::endl;
        return -1;
    }

    struct CameraResolution supported_resolution[HIAI_MAX_CAMERARESOLUTION_COUNT];

    ret = GetCameraProperty(0, CAMERA_PROP_SUPPORTED_RESOLUTION, supported_resolution);
    if (ret != LIBMEDIA_STATUS_OK) {
        std::cerr << "GetCameraProperty Resolution Failed " << ret << std::endl;
        return -1;
    }

    int i = 0;
    do {
        std::cout << "Camera suppported width=" << supported_resolution[i].width
            << " height=" << supported_resolution[i].height << std::endl;
        ++i;
    }while(supported_resolution[i].width != -1 && i < HIAI_MAX_CAMERARESOLUTION_COUNT);

    
    int fps = 20;
    ret = SetCameraProperty(0, CAMERA_PROP_FPS, &fps);
    if(ret != LIBMEDIA_STATUS_OK) {
        std::cerr << "SetCameraProperty 0 failed" << ret << std::endl;;
        return -1;
    }

    struct CameraResolution resolution;
    
    resolution.height = 720;
    resolution.width = 1280;
    ret = SetCameraProperty(0, CAMERA_PROP_RESOLUTION, &resolution);
    if(ret != LIBMEDIA_STATUS_OK) {
        std::cerr << "SetCameraProperty 0 failed" << ret << std::endl;
        return -1;
    }

    RtmpContext rtmp_ctx;
    ret = rtmp_ctx.Init("mystream", 720, 1280);

    if (ret != 0) {
        std::cerr << "InitRtmp Ctx failed ret: " << ret << std::endl;
        return -1;
    }

    CHECK_ACL(aclrtSetDevice(0));

    aclrtContext ctx;
    CHECK_ACL(aclrtCreateContext(&ctx, 0));
    CHECK_ACL(aclrtSetCurrentContext(ctx));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclrtRunMode mode;
    CHECK_ACL(aclrtGetRunMode(&mode));
    std::cerr << "run mode:" << std::string(mode==ACL_DEVICE?"device":"host") << std::endl;

    VPCResizeEngine resize_engine(stream);

    resize_engine.Init(720, 1280, yolov3_model_size, 512);

    RtmpContext resized_ctx;
    ret = resized_ctx.Init("resize", yolov3_model_size, 512);

    ACLModel model(stream);
    model.Init("./model/sample-yolov3.om");

    std::cout << "Model Info:" << std::endl;
    std::cout << model.ToString();

    CameraCtx camera_ctx;
    camera_ctx.resize = &resize_engine;
    camera_ctx.rtmp = &rtmp_ctx;
    camera_ctx.resize_rtmp = &resized_ctx;
    camera_ctx.dev_ctx = &ctx;
    camera_ctx.model = &model;

    
    ret = CapCamera(0, CameraCallBack, &camera_ctx);
    sleep(50000000);
    
    /*
    int cam_buffer_size = yuv420sp_size(720, 1280);

    uint8_t* cam_buffer = (uint8_t*)malloc(cam_buffer_size);

    CameraCapMode cap_mode = CAMERA_CAP_ACTIVE;
    

    ret = SetCameraProperty(0, CAMERA_PROP_CAP_MODE, &mode);

    if(ret != LIBMEDIA_STATUS_OK) {
        std::cerr << "SetCameraProperty 0 failed " << ret << std::endl;;
        return -1;
    }

    while (true) {
        ReadFrameFromCamera(0, cam_buffer, &cam_buffer_size);
        rtmp_ctx.SendFrame((const uint8_t*)cam_buffer);
        resize_engine.Resize((const uint8_t*)cam_buffer, cam_buffer_size);
        resized_ctx.SendFrame((const uint8_t*)resize_engine.GetOutputBuffer());
    }
    */

    CHECK_ACL(aclFinalize());

    return 0;
}