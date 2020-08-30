extern "C" {
#include "peripheral_api.h"
}

#include <iostream>

int main(int argc, char** argv) {
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
    
    resolution.height = 1080;
    resolution.width = 1920;
    ret = SetCameraProperty(0, CAMERA_PROP_RESOLUTION, &resolution);
    if(ret != LIBMEDIA_STATUS_OK) {
        std::cerr << "SetCameraProperty 0 failed" << ret << std::endl;
        return -1;
    }

    

    return 0;
}