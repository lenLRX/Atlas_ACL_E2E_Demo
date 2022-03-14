#ifndef __FOCUS_OP_H__
#define __FOCUS_OP_H__

#include "app_profiler.h"

template <typename T>
void FocusTransform(int height, int width, T *dst, T *src) {
  APP_PROFILE(FocusTransform);
  // (h, w, 3) to (3*4, h/2, w/2)
  int dst_plane_size = height * width / 4;
  int src_row_size = width * 3;
  int dst_row_size = width / 2;
  int half_row = width / 2;
  int half_height = height / 2;
  for (int i = 0; i < height; ++i) {
    int row_off = i % 2;
    int dst_even_offset = row_off * 3 * dst_plane_size + dst_row_size * (i / 2);
    T *dst_even = dst + dst_even_offset;
    int dst_odd_offset =
        (row_off + 2) * 3 * dst_plane_size + dst_row_size * (i / 2);
    T *dst_odd = dst + dst_odd_offset;
    T *src_row = src + i * src_row_size;
    for (int j = 0; j < half_row; ++j) {
      dst_even[j] = *src_row;
      ++src_row;
      dst_even[j + dst_plane_size] = *src_row;
      ++src_row;
      dst_even[j + 2 * dst_plane_size] = *src_row;
      ++src_row;

      dst_odd[j] = *src_row;
      ++src_row;
      dst_odd[j + dst_plane_size] = *src_row;
      ++src_row;
      dst_odd[j + 2 * dst_plane_size] = *src_row;
      ++src_row;
    }
  }
  // std::ofstream ofs("focus.bin", std::ios::binary | std::ios::out);
  // ofs.write((const char*)dst, height * width * 3 * sizeof(T));
}

void FocusTransformNEON(int height, int width, uint8_t *dst, uint8_t *src);

void FocusTransformNEONFuse(int height, int width, float *dst, uint8_t *src);

void CvtFocusNEONFuse(int height, int width, float *dst, uint8_t *src,
                      float coeff = 0.00392156862745098);

void YUV420SP2RGBNEON(int height, int width, uint8_t *dst, uint8_t *src);

#endif //__FOCUS_OP_H__