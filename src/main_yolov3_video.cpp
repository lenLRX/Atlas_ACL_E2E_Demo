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
#include "opencv2/opencv.hpp"

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
    /*
    {
        //PERF_TIMER();
        ctx->rtmp->SendFrame((const uint8_t*)pdata);
    }
    */
    {
        //PERF_TIMER();
        CHECK_ACL(ctx->resize->Resize((const uint8_t*)pdata, size));
    }
    
    const uint8_t* resized_buffer = (const uint8_t*)ctx->resize->GetOutputBuffer();
    {
        //PERF_TIMER();
        ctx->resize_rtmp->SendFrame(resized_buffer);
    }
    
    const auto& input_buffers = ctx->model->GetInputBuffer();
    memcpy(input_buffers[0], resized_buffer, ctx->model->GetInputBufferSizes()[0]);
    float* img_info = (float*)input_buffers[1];
    img_info[0] = yolov3_model_size;
    img_info[1] = yolov3_model_size;
    img_info[2] = 720;// scale H
    img_info[3] = 1280;// scale W
    {
        //PERF_TIMER();
        ctx->model->Infer();
    }

    int post_nms_num = 1024;
    const auto& output_buffers = ctx->model->GetOutputBuffer();
    float* box_info = (float*)output_buffers[0];
    int32_t box_out_num = ((int32_t*)output_buffers[1])[0];

    cv::Mat mYUV(720*1.5, 1280, CV_8UC1, (void*) pdata);
    cv::Mat mRGB(720, 1280, CV_8UC3);
    cv::Mat mYUV420P(720*1.5, 1280, CV_8UC1);
    {
        PERF_TIMER();
        cv::cvtColor(mYUV, mRGB, CV_YUV2RGB_NV12, 3);
    }
    
    
    std::cout << "result box num:" << box_out_num << std::endl;
    
    for (int i = 0;i < box_out_num; ++i) {
        float x1 = box_info[box_out_num * 0 + i];
        float y1 = box_info[box_out_num * 1 + i];
        float x2 = box_info[box_out_num * 2 + i];
        float y2 = box_info[box_out_num * 3 + i];
        float score = box_info[box_out_num * 4 + i];
        float label = box_info[box_out_num * 5 + i];
        std::cout << "box info: x1: " << x1
          << " y2: " << y1
          << " x2: " << x2
          << " y2: " << y2
          << " score: " << score
          << " label: " << yolov3_label[int(label)] << std::endl;
          //<< " label: " << label << std::endl;
        cv::rectangle(mRGB, {x1, y1}, {x2, y2}, cv::Scalar(237, 149, 100));
    }

    {
        PERF_TIMER();
        cv::cvtColor(mRGB, mYUV420P, CV_RGB2YUV_I420, 1);
    }

    ctx->rtmp->SendFrame((const uint8_t*)(mYUV420P.ptr()));

    return 1;
}


void hwc_to_chw(cv::InputArray src, cv::OutputArray dst) {
  const int src_h = src.rows();
  const int src_w = src.cols();
  const int src_c = src.channels();

  cv::Mat hw_c = src.getMat().reshape(1, src_h * src_w);

  const std::array<int,3> dims = {src_c, src_h, src_w};                         
  dst.create(3, &dims[0], CV_MAKETYPE(src.depth(), 1));                         
  int new_shapes[] = {src_c, src_h, src_w};
  cv::Mat dst_1d = dst.getMat().reshape(1, 3, new_shapes);              

  cv::transpose(hw_c, dst_1d);                                                  
}    

int main(int argc, char** argv) {
    CHECK_ACL(aclInit(nullptr));
    int ret;

    cv::VideoCapture cap("test.mp4");

    if(!cap.isOpened()){
        std::cerr << "failed to open video file" << std::endl;
        return -1;
    }


    RtmpContext rtmp_ctx;
    ret = rtmp_ctx.Init("mystream", 720, 1280, AV_PIX_FMT_YUV420P);

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

    ACLModel model(stream);
    model.Init("./model/sample-yolov3_pp_no_aipp.om");

    std::cout << "Model Info:" << std::endl;
    std::cout << model.ToString();

    while (true) {
        cv::Mat raw_frame;
        cap >> raw_frame;
        if (raw_frame.empty()) {
            break;
        }
        //std::cout << "frame size " << raw_frame.size() << " depth " << raw_frame.depth() << std::endl;

        cv::Mat raw_frame_fp32;
        raw_frame.convertTo(raw_frame_fp32, CV_32FC3);

        cv::Mat frame_720p(720, 1280, CV_8UC3);
        cv::resize(raw_frame, frame_720p, frame_720p.size(), 0, 0);
        
        //std::cout << "resized 720p" << std::endl;
        cv::Mat frame_416(416, 416, CV_32FC3);
        cv::resize(raw_frame_fp32, frame_416, frame_416.size(), 0, 0);

        frame_416 = frame_416 / 255;

        //std::cout << "resized 416" << std::endl;

        const auto& input_buffers = model.GetInputBuffer();

        float* imgbuf = (float*)input_buffers[0];
        float* cv_buf = (float*)frame_416.ptr();

        const int src_h = frame_416.rows;
        const int src_w = frame_416.cols;
        const int src_c = frame_416.channels();

        for (int c = 0;c < src_c; ++c)
            for (int h = 0;h < src_h; ++h)
                for (int w = 0;w < src_w; ++w)
                    imgbuf[c*src_h*src_w + h*src_w + w] = cv_buf[h * src_w * src_c + w * src_c + c];

        //memcpy(input_buffers[0], frame_416_chw.ptr(), model.GetInputBufferSizes()[0]);
        //std::cout << "memcpy done " << std::endl;
        float* img_info = (float*)input_buffers[1];
        img_info[0] = yolov3_model_size;
        img_info[1] = yolov3_model_size;
        img_info[2] = 720;// scale H
        img_info[3] = 1280;// scale W
        {
            PERF_TIMER();
            model.Infer();
        }

        int post_nms_num = 1024;
        const auto& output_buffers = model.GetOutputBuffer();
        float* box_info = (float*)output_buffers[0];
        int32_t box_out_num = ((int32_t*)output_buffers[1])[0];

        std::cout << "result box num:" << box_out_num << std::endl;

        cv::Mat mYUV420P(720*1.5, 1280, CV_8UC1);

        for (int i = 0;i < box_out_num; ++i) {
            float x1 = box_info[box_out_num * 0 + i];
            float y1 = box_info[box_out_num * 1 + i];
            float x2 = box_info[box_out_num * 2 + i];
            float y2 = box_info[box_out_num * 3 + i];
            float score = box_info[box_out_num * 4 + i];
            float label = box_info[box_out_num * 5 + i];
            std::cout << "box info: x1: " << x1
            << " y2: " << y1
            << " x2: " << x2
            << " y2: " << y2
            << " score: " << score
            << " label: " << yolov3_label[int(label)+ 1] << std::endl;
            //<< " label: " << label << std::endl;
            cv::rectangle(frame_720p, {x1, y1}, {x2, y2}, cv::Scalar(237, 149, 100));
        }

        {
            PERF_TIMER();
            cv::cvtColor(frame_720p, mYUV420P, CV_BGR2YUV_I420, 1);
        }

        rtmp_ctx.SendFrame((const uint8_t*)(mYUV420P.ptr()));
    }

    CHECK_ACL(aclFinalize());

    return 0;
}