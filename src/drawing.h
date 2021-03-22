#ifndef __DRAWING__H__
#define __DRAWING__H__

#include <algorithm>
#include <iostream>
#include <string>
#include <stdint.h>
#include <cmath>

#include "freetype_helper.h"

class YUVColor {
public:
  YUVColor(uint8_t y, uint8_t u, uint8_t v) : y(y), u(u), v(v) {}
  uint8_t y;
  uint8_t u;
  uint8_t v;
};

// nv12
class YUV420SPImage {
public:
  YUV420SPImage(uint8_t *img, int h, int w) {
    y_addr = img;
    uv_addr = img + h * w;
    img_h = h;
    img_w = w;
  }

  void DrawText(int x, int y, const std::string& text, const YUVColor &color) {
    RenderText(x, y, text, &color, this);
  }

  void DrawRect(int x1, int y1, int x2, int y2, const YUVColor &color,
                int width) {
    if (x1 > x2) {
      std::swap(x1, x2);
    }

    if (y1 > y2) {
      std::swap(y1, y2);
    }

    int i, j;
    int i_bound, j_bound;
    int i_start, j_start;

    j_bound = std::min(img_h, y1 + width);
    i_bound = std::min(img_w - 1, x2);

    for (j = y1; j < j_bound; ++j) {
      for (i = x1; i <= i_bound; ++i) {
        SetPixel(j, i, color);
      }
    }

    j_start = std::max(0, y2 - width + 1);
    j_bound = std::min(img_h - 1, y2);

    i_start = std::max(0, x1);
    i_bound = std::min(img_w - 1, x2);

    for (j = j_start; j <= j_bound; ++j) {
      for (i = i_start; i <= i_bound; ++i) {
        SetPixel(j, i, color);
      }
    }

    i_bound = std::min(img_w, x1 + width);
    j_bound = std::min(img_h - 1, y2);

    for (i = x1; i < i_bound; ++i) {
      for (j = y1; j <= j_bound; ++j) {
        SetPixel(j, i, color);
      }
    }

    i_start = std::max(0, x2 - width + 1);
    i_bound = std::min(img_w - 1, x2);

    j_start = std::max(0, y1);
    j_bound = std::min(img_h - 1, y2);

    for (i = i_start; i <= i_bound; ++i) {
      for (j = j_start; j <= j_bound; ++j) {
        SetPixel(j, i, color);
      }
    }
  }

  inline void SetPixel(int h, int w, const YUVColor &color) {
    *(y_addr + h * img_w) = color.y;
    uint8_t *uv_offset = uv_addr + (h / 2) * img_w + w / 2 * 2;
    uv_offset[0] = color.u;
    uv_offset[1] = color.v;
  }

  inline int GetHeight() const {
    return img_h;
  }

  inline int GetWidth() const {
    return img_w;
  }

private:
  uint8_t *y_addr;
  uint8_t *uv_addr;
  int img_h;
  int img_w;
};

#endif //__DRAWING__H__