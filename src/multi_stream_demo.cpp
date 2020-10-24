#include "rtsp_input.h"
#include "dvpp_decoder.h"
#include "dvpp_encoder.h"
#include "vpc_resize.h"
#include "rtmp_stream.h"
#include "acl_model.h"
#include "util.h"
#include "drawing.h"

#include "acl_cb_thread.h"

#include <thread>
#include <chrono>
#include <string>
#include <iostream>

using namespace std::chrono_literals;

const static int yolov3_model_size = 416;

// RTSP input >> DVPP decode >> DVPP resize
// >> AICORE yolov3 >> draw box >> DVPP encode >> RTMP output

void DetectAndDraw(ACLModel* model, uint8_t* buffer) {
    const auto& input_buffers = model->GetInputBuffer();
    memcpy(input_buffers[0], buffer, model->GetInputBufferSizes()[0]);
    float* img_info = (float*)input_buffers[1];
    img_info[0] = yolov3_model_size;
    img_info[1] = yolov3_model_size;
    img_info[2] = yolov3_model_size;// scale H
    img_info[3] = yolov3_model_size;// scale W

    model->Infer();

    int post_nms_num = 1024;
    const auto& output_buffers = model->GetOutputBuffer();
    float* box_info = (float*)output_buffers[0];
    int32_t box_out_num = ((int32_t*)output_buffers[1])[0];
    
    std::cout << "result box num:" << box_out_num << std::endl;
    PERF_TIMER();

    YUV420SPImage img(buffer, yolov3_model_size, yolov3_model_size);
    YUVColor box_color(0, 0, 0xff);// Red?
    
    for (int i = 0;i < box_out_num; ++i) {
        float x1 = box_info[box_out_num * 0 + i];
        float y1 = box_info[box_out_num * 1 + i];
        float x2 = box_info[box_out_num * 2 + i];
        float y2 = box_info[box_out_num * 3 + i];
        float score = box_info[box_out_num * 4 + i];
        float label = box_info[box_out_num * 5 + i];
        /*
        std::cout << "box info: x1: " << x1
          << " y2: " << y1
          << " x2: " << x2
          << " y2: " << y2
          << " score: " << score
          << " label: " << yolov3_label[int(label) + 1] << std::endl;
        */
        img.DrawRect(x1, y1, x2, y2, box_color, 3);
    }
}

void StreamThread(std::string input_addr, std::string output_addr) {
    CHECK_ACL(aclrtSetDevice(0));
    AclCallBackThread* cb_thread_p = new AclCallBackThread();
    AclCallBackThread& cb_thread = *cb_thread_p;

    aclrtContext ctx;
    CHECK_ACL(aclrtCreateContext(&ctx, 0));
    CHECK_ACL(aclrtSetCurrentContext(ctx));
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    CHECK_ACL(aclrtSubscribeReport(cb_thread.GetPid(), stream));

    ACLModel model(stream);
    model.Init("./model/sample-yolov3_pp_416.om");

    std::cout << "Model Info:" << std::endl;
    std::cout << model.ToString();

    RTSPInput rtsp_input;
    rtsp_input.Init(input_addr.c_str());

    RtmpContext rtmp_output;
    rtmp_output.Init(output_addr, yolov3_model_size, yolov3_model_size, rtsp_input.GetFramerate());

    DvppDecoder decoder;
    decoder.Init(cb_thread.GetPid(), rtsp_input.GetHeight(), rtsp_input.GetWidth());
    decoder.SetDeviceCtx(&ctx);

    //DvppEncoder encoder;
    //encoder.Init(cb_thread.GetPid(), yolov3_model_size, yolov3_model_size, &rtmp_output);

    VPCResizeEngine resize_engine(stream);
    resize_engine.Init(rtsp_input.GetHeight(), rtsp_input.GetWidth(), yolov3_model_size, yolov3_model_size);

    resize_engine.RegisterHandler([&](uint8_t* buffer){
            DetectAndDraw(&model, buffer);
            //encoder.SendFrame(buffer);
            rtmp_output.SendFrame(buffer);
        });

    decoder.RegisterHandler([&](uint8_t* buffer){
        PERF_TIMER();
        resize_engine.Resize(buffer);
        });

    rtsp_input.RegisterHandler([&](AVPacket* packet){decoder.SendFrame(packet);});
    rtsp_input.Run();

    decoder.Destory();

    std::this_thread::sleep_for(1s);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    resize_engine.Destory();
    //encoder.Destory();

    //CHECK_ACL(aclrtUnSubscribeReport(cb_thread.GetPid(), stream));
    //cb_thread.Join();
    std::cout << "End of stream" << std::endl;
}

int main(int argc, char** argv) {
    CHECK_ACL(aclInit(nullptr));

    std::string input_promot = "--input";
    std::string output_promot = "--output";

    std::vector<std::thread> streams;
    if ((argc - 1) % 4 != 0) {
        std::cerr << "Invalid cmd line option" << std::endl;
        return -1;
    }

    int stream_num = (argc - 1) / 4;
    int arg_i = 1;

    for (int i = 0;i < stream_num; ++i) {
        std::string i_promot(argv[arg_i]);
        if (i_promot != input_promot) {
            std::cerr << "invalid option: " << i_promot << std::endl;
            return -1;
        }
        ++arg_i;

        std::string input_addr(argv[arg_i]);

        ++arg_i;
        std::string o_promot(argv[arg_i]);
        if (o_promot != output_promot) {
            std::cerr << "invalid option: " << i_promot << std::endl;
            return -1;
        }

        ++arg_i;

        std::string output_addr(argv[arg_i]);
        ++arg_i;
        std::cout << "Add stream --input " << input_addr
            << " --output " << output_addr << std::endl;
        streams.emplace_back(StreamThread, input_addr, output_addr);
    }

    for (auto& t: streams) {
        t.join();
    }
    
    return 0;
}
