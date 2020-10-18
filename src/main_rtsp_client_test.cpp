#include "rtsp_input.h"


int main(int argc, char** argv) {
    RTSPInput rtsp_input;
    rtsp_input.Init("rtsp://192.168.1.9:8554/tt.mp4");
    while (true) {
        rtsp_input.Pull();
    }
    return 0;
}
