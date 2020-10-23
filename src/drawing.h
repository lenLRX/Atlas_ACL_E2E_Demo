#ifndef __DRAWING__H__
#define __DRAWING__H__

#include <stdint.h>
#include <algorithm>
#include <iostream>

class YUVColor {
public:
    YUVColor(uint8_t y, uint8_t u, uint8_t v)
        :y(y), u(u), v(v) {}
    uint8_t y;
    uint8_t u;
    uint8_t v;
};

// nv12
class YUV420SPImage {
public:
    YUV420SPImage(uint8_t* img, int h, int w) {
        y_addr = img;
        uv_addr = img + h * w;
        img_h = h;
        img_w = w;
    }

    void DrawRect(int x1, int y1, int x2, int y2, const YUVColor& color, int width) {
        if (x1 > x2) {
            std::swap(x1, x2);
        }

        if (y1 > y2) {
            std::swap(y1, y2);
        }

        int i, j;

        for (j = y1;j < y1 + width; ++j) {
            for (i = x1; i <= x2; ++i) {
                SetPixel(j, i, color);
            }
        }

        for (j = y2 - width + 1;j <= y2; ++j) {
            for (i = x1; i <= x2; ++i) {
                SetPixel(j, i, color);
            }
        }

        for (i = x1;i < x1 + width; ++i) {
            for (j = y1; j <= y2; ++j) {
                SetPixel(j, i, color);
            }
        }

        for (i = x2 - width + 1;i <= x2; ++i) {
            for (j = y1; j <= y2; ++j) {
                SetPixel(j, i, color);
            }
        }
    }

    inline void SetPixel(int h, int w, const YUVColor& color) {
        *(y_addr + h * img_w) = color.y;
        uint8_t* uv_offset =  uv_addr + (h / 2) * img_w + w / 2 * 2;
        uv_offset[0] = color.u;
        uv_offset[1] = color.v;
    }
private:
    uint8_t* y_addr;
    uint8_t* uv_addr;
    int img_h;
    int img_w;
};

#endif//__DRAWING__H__