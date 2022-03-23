#include "focus_op.h"

#ifdef __ARM_NEON

#include <arm_neon.h>

// https://en.wikipedia.org/wiki/YUV#Y%E2%80%B2UV420sp_(NV21)_to_RGB_conversion_(Android)
void YUV420SP2RGBNEON(int height, int width, uint8_t *dst, uint8_t *src) {
  APP_PROFILE(YUV420SP2RGBNEON);
  uint8_t *Y_even_row_start = src;
  uint8_t *Y_odd_row_start = src + width;
  uint8_t *UV_row_start = src + height * width;
  uint8_t *dst_even_row_start = dst;
  uint8_t *dst_odd_row_start = dst + width * 3;

  const int VL = 16;
  const int half_VL = VL / 2;
  int row_iter = height / 2; // process 2 row in a time
  int col_iter = width / VL; // process 16 pixel in a time

  int16x8_t minus128 = vdupq_n_s16(128);
  int16x4_t coeff351_s16 = vdup_n_s16(351);
  int16x4_t coeff179_s16 = vdup_n_s16(179);
  int16x4_t coeff86_s16 = vdup_n_s16(86);
  int16x4_t coeff443_s16 = vdup_n_s16(443);

  for (int i = 0; i < row_iter; ++i) {
    uint8_t *Y_even = Y_even_row_start;
    uint8_t *Y_odd = Y_odd_row_start;
    uint8_t *UV = UV_row_start;
    uint8_t *dst_even_ptr = dst_even_row_start;
    uint8_t *dst_odd_ptr = dst_odd_row_start;
    for (int j = 0; j < col_iter; ++j) {
      uint8x8x2_t y_even_pixels = vld2_u8(Y_even);
      Y_even += VL;
      uint8x8x2_t y_odd_pixels = vld2_u8(Y_odd);
      Y_odd += VL;

      uint8x8x2_t uv_pixels = vld2_u8(UV);
      UV += VL;

      // convert uv to int16 and minus 128
      uint8x8_t u_pixels = uv_pixels.val[0];
      uint8x8_t v_pixels = uv_pixels.val[1];

      uint16x8_t u_pixels_u16 = vmovl_u8(u_pixels);
      uint16x8_t v_pixels_u16 = vmovl_u8(v_pixels);

      int16x8_t u_pixels_s16 =
          vcombine_s16(vreinterpret_s16_u16(vget_low_u16(u_pixels_u16)),
                       vreinterpret_s16_u16(vget_high_u16(u_pixels_u16)));

      int16x8_t v_pixels_s16 =
          vcombine_s16(vreinterpret_s16_u16(vget_low_u16(v_pixels_u16)),
                       vreinterpret_s16_u16(vget_high_u16(v_pixels_u16)));

      int16x8_t u_pixels_m128_s16 = vsubq_s16(u_pixels_s16, minus128);
      int16x8_t v_pixels_m128_s16 = vsubq_s16(v_pixels_s16, minus128);

      // int16_t: (351*(vValue-128))>>8;
      // max value: (351*(255-128)) >> 8  --> 174
      // min value: (351*(0-128)) >> 8  --> -176
      // value will not overflow
      int16x4_t rTmp_s16_low = vmovn_s32(vshrq_n_s32(
          vmull_s16(coeff351_s16, vget_low_s16(v_pixels_m128_s16)), 8));
      int16x4_t rTmp_s16_high = vmovn_s32(vshrq_n_s32(
          vmull_s16(coeff351_s16, vget_high_s16(v_pixels_m128_s16)), 8));
      int16x8_t rTmp_s16 = vcombine_s16(rTmp_s16_low, rTmp_s16_high);

      // int16_t: (179*(vValue-128) + 86*(uValue-128))>>8
      int16x4_t gTmp_s16_low = vmovn_s32(vshrq_n_s32(
          vaddq_s32(vmull_s16(coeff179_s16, vget_low_s16(v_pixels_m128_s16)),
                    vmull_s16(coeff86_s16, vget_low_s16(u_pixels_m128_s16))),
          8));

      int16x4_t gTmp_s16_high = vmovn_s32(vshrq_n_s32(
          vaddq_s32(vmull_s16(coeff179_s16, vget_high_s16(v_pixels_m128_s16)),
                    vmull_s16(coeff86_s16, vget_high_s16(u_pixels_m128_s16))),
          8));
      int16x8_t gTmp_s16 = vcombine_s16(gTmp_s16_low, gTmp_s16_high);

      // int16_t: (443*(uValue-128))>>8
      int16x4_t bTmp_s16_low = vmovn_s32(vshrq_n_s32(
          vmull_s16(coeff443_s16, vget_low_s16(u_pixels_m128_s16)), 8));
      int16x4_t bTmp_s16_high = vmovn_s32(vshrq_n_s32(
          vmull_s16(coeff443_s16, vget_high_s16(u_pixels_m128_s16)), 8));
      int16x8_t bTmp_s16 = vcombine_s16(bTmp_s16_low, bTmp_s16_high);

      uint8x8x3_t result_even[2];
      uint8x8x3_t result_odd[2];

      uint8x8x3_t result_even_zip[2];
      uint8x8x3_t result_odd_zip[2];

      for (int k = 0; k < 2; ++k) {
        // convert y to int16
        uint8x8_t y_even_pixels_u8 = y_even_pixels.val[k];
        uint8x8_t y_odd_pixels_u8 = y_odd_pixels.val[k];

        uint16x8_t y_even_pixels_u16 = vmovl_u8(y_even_pixels_u8);
        uint16x8_t y_odd_pixels_u16 = vmovl_u8(y_odd_pixels_u8);

        int16x8_t y_even_pixels_s16 = vcombine_s16(
            vreinterpret_s16_u16(vget_low_u16(y_even_pixels_u16)),
            vreinterpret_s16_u16(vget_high_u16(y_even_pixels_u16)));

        int16x8_t y_odd_pixels_s16 =
            vcombine_s16(vreinterpret_s16_u16(vget_low_u16(y_odd_pixels_u16)),
                         vreinterpret_s16_u16(vget_high_u16(y_odd_pixels_u16)));

        // rTmp = yValue + (351*(vValue-128))>>8;
        int16x8_t rTmp_even = vaddq_s16(y_even_pixels_s16, rTmp_s16);
        result_even[k].val[0] = vmovn_u16(
            vcombine_u16(vreinterpret_u16_s16(vget_low_s16(rTmp_even)),
                         vreinterpret_u16_s16(vget_high_s16(rTmp_even))));

        int16x8_t rTmp_odd = vaddq_s16(y_odd_pixels_s16, rTmp_s16);
        result_odd[k].val[0] = vmovn_u16(
            vcombine_u16(vreinterpret_u16_s16(vget_low_s16(rTmp_odd)),
                         vreinterpret_u16_s16(vget_high_s16(rTmp_odd))));

        int16x8_t gTmp_even = vsubq_s16(y_even_pixels_s16, gTmp_s16);
        result_even[k].val[1] = vmovn_u16(
            vcombine_u16(vreinterpret_u16_s16(vget_low_s16(gTmp_even)),
                         vreinterpret_u16_s16(vget_high_s16(gTmp_even))));

        int16x8_t gTmp_odd = vsubq_s16(y_odd_pixels_s16, gTmp_s16);
        result_odd[k].val[1] = vmovn_u16(
            vcombine_u16(vreinterpret_u16_s16(vget_low_s16(gTmp_odd)),
                         vreinterpret_u16_s16(vget_high_s16(gTmp_odd))));

        int16x8_t bTmp_even = vaddq_s16(y_even_pixels_s16, bTmp_s16);
        result_even[k].val[2] = vmovn_u16(
            vcombine_u16(vreinterpret_u16_s16(vget_low_s16(bTmp_even)),
                         vreinterpret_u16_s16(vget_high_s16(bTmp_even))));

        int16x8_t bTmp_odd = vaddq_s16(y_odd_pixels_s16, bTmp_s16);
        result_odd[k].val[2] = vmovn_u16(
            vcombine_u16(vreinterpret_u16_s16(vget_low_s16(bTmp_odd)),
                         vreinterpret_u16_s16(vget_high_s16(bTmp_odd))));
      }

      for (int c = 0; c < 3; ++c) {
        result_even_zip[0].val[c] =
            vzip1_u8(result_even[0].val[c], result_even[1].val[c]);
        result_even_zip[1].val[c] =
            vzip2_u8(result_even[0].val[c], result_even[1].val[c]);

        result_odd_zip[0].val[c] =
            vzip1_u8(result_odd[0].val[c], result_odd[1].val[c]);
        result_odd_zip[1].val[c] =
            vzip2_u8(result_odd[0].val[c], result_odd[1].val[c]);
      }

      vst3_u8(dst_even_ptr, result_even_zip[0]);
      dst_even_ptr += half_VL * 3;
      vst3_u8(dst_even_ptr, result_even_zip[1]);
      dst_even_ptr += half_VL * 3;

      vst3_u8(dst_odd_ptr, result_odd_zip[0]);
      dst_odd_ptr += half_VL * 3;
      vst3_u8(dst_odd_ptr, result_odd_zip[1]);
      dst_odd_ptr += half_VL * 3;
    }

    Y_even_row_start += width * 2;
    Y_odd_row_start += width * 2;
    UV_row_start += width;
    dst_even_row_start += width * 2 * 3;
    dst_odd_row_start += width * 2 * 3;
  }
}

void FocusTransformNEON(int height, int width, uint8_t *dst, uint8_t *src) {
  APP_PROFILE(FocusTransformNEON);
  // (h, w, 3) to (3*4, h/2, w/2)
  int dst_plane_size = height * width / 4;
  int src_row_size = width * 3;
  int dst_row_size = width / 2;
  int half_row = width / 2;
  int half_height = height / 2;
  for (int i = 0; i < height; ++i) {
    int row_off = i % 2;
    int dst_even_offset = row_off * 3 * dst_plane_size + dst_row_size * (i / 2);
    uint8_t *dst_even = dst + dst_even_offset;
    int dst_odd_offset =
        (row_off + 2) * 3 * dst_plane_size + dst_row_size * (i / 2);
    uint8_t *dst_odd = dst + dst_odd_offset;
    uint8_t *src_row = src + i * src_row_size;
    const int row_repeat = half_row / 16;
    const int load_size = 16 * 3;
    for (int j = 0; j < row_repeat; ++j) {
      uint8x16x3_t vec_dst_low = vld3q_u8(src_row);
      src_row += load_size;

      uint8x16x3_t vec_dst_high = vld3q_u8(src_row);
      src_row += load_size;

      uint8_t *curr_even = dst_even + j * 16;
      uint8_t *curr_odd = dst_odd + j * 16;

      for (int c = 0; c < 3; ++c) {
        uint8x16_t vec_even =
            vuzp1q_u8(vec_dst_low.val[c], vec_dst_high.val[c]);
        uint8x16_t vec_odd = vuzp2q_u8(vec_dst_low.val[c], vec_dst_high.val[c]);
        vst1q_u8(curr_even, vec_even);
        curr_even += dst_plane_size;
        vst1q_u8(curr_odd, vec_odd);
        curr_odd += dst_plane_size;
      }
    }
  }

  // std::ofstream ofs("focus_neon.bin", std::ios::binary | std::ios::out);
  // ofs.write((const char*)dst, height * width * 3 * sizeof(float));
}

void FocusTransformNEONFuse(int height, int width, float *dst, uint8_t *src) {
  APP_PROFILE(FocusTransformNEONFuse);
  // (h, w, 3) to (3*4, h/2, w/2)
  int dst_plane_size = height * width / 4;
  int src_row_size = width * 3;
  int dst_row_size = width / 2;
  int half_row = width / 2;
  int half_height = height / 2;

  float32x4_t coeff = vdupq_n_f32(0.00392156862745098);

  for (int i = 0; i < height; ++i) {
    int row_off = i % 2;
    int dst_even_offset = row_off * 3 * dst_plane_size + dst_row_size * (i / 2);
    float *dst_even = dst + dst_even_offset;
    int dst_odd_offset =
        (row_off + 2) * 3 * dst_plane_size + dst_row_size * (i / 2);
    float *dst_odd = dst + dst_odd_offset;
    uint8_t *src_row = src + i * src_row_size;
    const int row_repeat = half_row / 16;
    const int load_size = 16 * 3;
    for (int j = 0; j < row_repeat; ++j) {
      uint8x16x3_t vec_dst_low = vld3q_u8(src_row);
      src_row += load_size;

      float *curr_even = dst_even + j * 16;
      uint8x16x3_t vec_dst_high = vld3q_u8(src_row);
      src_row += load_size;

      float *curr_odd = dst_odd + j * 16;

      for (int c = 0; c < 3; ++c) {
        uint8x16_t vec_even =
            vuzp1q_u8(vec_dst_low.val[c], vec_dst_high.val[c]);
        uint8x16_t vec_odd = vuzp2q_u8(vec_dst_low.val[c], vec_dst_high.val[c]);

        uint16x8_t vec_even_low_u16 = vmovl_u8(vget_low_u8(vec_even));
        uint16x8_t vec_even_high_u16 = vmovl_u8(vget_high_u8(vec_even));

        float32x4x4_t vec_even_q_f32;

        vec_even_q_f32.val[0] = vmulq_f32(
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vec_even_low_u16))), coeff);
        vec_even_q_f32.val[1] = vmulq_f32(
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vec_even_low_u16))), coeff);
        vec_even_q_f32.val[2] = vmulq_f32(
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vec_even_high_u16))), coeff);
        vec_even_q_f32.val[3] = vmulq_f32(
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vec_even_high_u16))), coeff);

        for (int q = 0; q < 4; ++q) {
          vst1q_f32(curr_even + q * 4, vec_even_q_f32.val[q]);
        }
        curr_even += dst_plane_size;

        uint16x8_t vec_odd_low_u16 = vmovl_u8(vget_low_u8(vec_odd));
        uint16x8_t vec_odd_high_u16 = vmovl_u8(vget_high_u8(vec_odd));

        float32x4x4_t vec_odd_q_f32;

        vec_odd_q_f32.val[0] = vmulq_f32(
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vec_odd_low_u16))), coeff);
        vec_odd_q_f32.val[1] = vmulq_f32(
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vec_odd_low_u16))), coeff);
        vec_odd_q_f32.val[2] = vmulq_f32(
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(vec_odd_high_u16))), coeff);
        vec_odd_q_f32.val[3] = vmulq_f32(
            vcvtq_f32_u32(vmovl_u16(vget_high_u16(vec_odd_high_u16))), coeff);

        for (int q = 0; q < 4; ++q) {
          vst1q_f32(curr_odd + q * 4, vec_odd_q_f32.val[q]);
        }
        curr_odd += dst_plane_size;
      }
    }
  }

  // std::ofstream ofs("focus.bin", std::ios::binary | std::ios::out);
  // ofs.write((const char*)dst, height * width * 3 * sizeof(float));
}

void CvtFocusNEONFuse(int height, int width, float *dst, uint8_t *src,
                      float coeff) {
  APP_PROFILE(CvtFocusNEONFuse);
  uint8_t *Y_even_row_start = src;
  uint8_t *Y_odd_row_start = src + width;
  uint8_t *UV_row_start = src + height * width;

  const int VL = 16;
  const int half_VL = VL / 2;
  int row_iter = height / 2; // process 2 row in a time
  int col_iter = width / VL; // process 16 pixel in a time

  int16x8_t minus128 = vdupq_n_s16(128);
  int16x4_t coeff351_s16 = vdup_n_s16(351);
  int16x4_t coeff179_s16 = vdup_n_s16(179);
  int16x4_t coeff86_s16 = vdup_n_s16(86);
  int16x4_t coeff443_s16 = vdup_n_s16(443);

  // (h, w, 3) to (3*4, h/2, w/2)
  int dst_plane_size = height * width / 4;
  int dst_row_size = width / 2;

  float32x4_t scale_coeff = vdupq_n_f32(coeff);

  for (int i = 0; i < row_iter; ++i) {
    uint8_t *Y_even = Y_even_row_start;
    uint8_t *Y_odd = Y_odd_row_start;
    uint8_t *UV = UV_row_start;

    int dst_even_row_even_col_offset = dst_row_size * i;
    int dst_even_row_odd_col_offset = 2 * 3 * dst_plane_size + dst_row_size * i;

    int dst_odd_row_even_col_offset = 3 * dst_plane_size + dst_row_size * i;
    int dst_odd_row_odd_col_offset = 3 * 3 * dst_plane_size + dst_row_size * i;

    float *dst_even_row[2];
    float *dst_odd_row[2];

    dst_even_row[0] = dst + dst_even_row_even_col_offset;
    dst_even_row[1] = dst + dst_even_row_odd_col_offset;
    dst_odd_row[0] = dst + dst_odd_row_even_col_offset;
    dst_odd_row[1] = dst + dst_odd_row_odd_col_offset;

    for (int j = 0; j < col_iter; ++j) {
      uint8x8x2_t y_even_pixels = vld2_u8(Y_even);
      Y_even += VL;
      uint8x8x2_t y_odd_pixels = vld2_u8(Y_odd);
      Y_odd += VL;

      uint8x8x2_t uv_pixels = vld2_u8(UV);
      UV += VL;

      // convert uv to int16 and minus 128
      uint8x8_t u_pixels = uv_pixels.val[0];
      uint8x8_t v_pixels = uv_pixels.val[1];

      uint16x8_t u_pixels_u16 = vmovl_u8(u_pixels);
      uint16x8_t v_pixels_u16 = vmovl_u8(v_pixels);

      int16x8_t u_pixels_s16 =
          vcombine_s16(vreinterpret_s16_u16(vget_low_u16(u_pixels_u16)),
                       vreinterpret_s16_u16(vget_high_u16(u_pixels_u16)));

      int16x8_t v_pixels_s16 =
          vcombine_s16(vreinterpret_s16_u16(vget_low_u16(v_pixels_u16)),
                       vreinterpret_s16_u16(vget_high_u16(v_pixels_u16)));

      int16x8_t u_pixels_m128_s16 = vsubq_s16(u_pixels_s16, minus128);
      int16x8_t v_pixels_m128_s16 = vsubq_s16(v_pixels_s16, minus128);

      // int16_t: (351*(vValue-128))>>8;
      // max value: (351*(255-128)) >> 8  --> 174
      // min value: (351*(0-128)) >> 8  --> -176
      // value will not overflow
      int16x4_t rTmp_s16_low = vmovn_s32(vshrq_n_s32(
          vmull_s16(coeff351_s16, vget_low_s16(v_pixels_m128_s16)), 8));
      int16x4_t rTmp_s16_high = vmovn_s32(vshrq_n_s32(
          vmull_s16(coeff351_s16, vget_high_s16(v_pixels_m128_s16)), 8));
      int16x8_t rTmp_s16 = vcombine_s16(rTmp_s16_low, rTmp_s16_high);

      // int16_t: (179*(vValue-128) + 86*(uValue-128))>>8
      int16x4_t gTmp_s16_low = vmovn_s32(vshrq_n_s32(
          vaddq_s32(vmull_s16(coeff179_s16, vget_low_s16(v_pixels_m128_s16)),
                    vmull_s16(coeff86_s16, vget_low_s16(u_pixels_m128_s16))),
          8));

      int16x4_t gTmp_s16_high = vmovn_s32(vshrq_n_s32(
          vaddq_s32(vmull_s16(coeff179_s16, vget_high_s16(v_pixels_m128_s16)),
                    vmull_s16(coeff86_s16, vget_high_s16(u_pixels_m128_s16))),
          8));
      int16x8_t gTmp_s16 = vcombine_s16(gTmp_s16_low, gTmp_s16_high);

      // int16_t: (443*(uValue-128))>>8
      int16x4_t bTmp_s16_low = vmovn_s32(vshrq_n_s32(
          vmull_s16(coeff443_s16, vget_low_s16(u_pixels_m128_s16)), 8));
      int16x4_t bTmp_s16_high = vmovn_s32(vshrq_n_s32(
          vmull_s16(coeff443_s16, vget_high_s16(u_pixels_m128_s16)), 8));
      int16x8_t bTmp_s16 = vcombine_s16(bTmp_s16_low, bTmp_s16_high);

      uint8x8x3_t result_even[2];
      uint8x8x3_t result_odd[2];

      uint8x8x3_t result_even_zip[2];
      uint8x8x3_t result_odd_zip[2];

      for (int k = 0; k < 2; ++k) {
        float *dst_even_base = dst_even_row[k];
        dst_even_row[k] += 8;
        float *dst_odd_base = dst_odd_row[k];
        dst_odd_row[k] += 8;

        // convert y to int16
        uint8x8_t y_even_pixels_u8 = y_even_pixels.val[k];
        uint8x8_t y_odd_pixels_u8 = y_odd_pixels.val[k];

        uint16x8_t y_even_pixels_u16 = vmovl_u8(y_even_pixels_u8);
        uint16x8_t y_odd_pixels_u16 = vmovl_u8(y_odd_pixels_u8);

        int16x8_t y_even_pixels_s16 = vcombine_s16(
            vreinterpret_s16_u16(vget_low_u16(y_even_pixels_u16)),
            vreinterpret_s16_u16(vget_high_u16(y_even_pixels_u16)));

        int16x8_t y_odd_pixels_s16 =
            vcombine_s16(vreinterpret_s16_u16(vget_low_u16(y_odd_pixels_u16)),
                         vreinterpret_s16_u16(vget_high_u16(y_odd_pixels_u16)));

        // rTmp = yValue + (351*(vValue-128))>>8;
        int16x8_t rTmp_even = vaddq_s16(y_even_pixels_s16, rTmp_s16);
        float32x4_t rTmp_even_f32_qlow = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(rTmp_even))), scale_coeff);
        float32x4_t rTmp_even_f32_qhigh = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(rTmp_even))), scale_coeff);
        vst1q_f32(dst_even_base, rTmp_even_f32_qlow);
        vst1q_f32(dst_even_base + 4, rTmp_even_f32_qhigh);
        dst_even_base += dst_plane_size;

        int16x8_t rTmp_odd = vaddq_s16(y_odd_pixels_s16, rTmp_s16);
        float32x4_t rTmp_odd_f32_qlow = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(rTmp_odd))), scale_coeff);
        float32x4_t rTmp_odd_f32_qhigh = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(rTmp_odd))), scale_coeff);
        vst1q_f32(dst_odd_base, rTmp_odd_f32_qlow);
        vst1q_f32(dst_odd_base + 4, rTmp_odd_f32_qhigh);
        dst_odd_base += dst_plane_size;

        int16x8_t gTmp_even = vsubq_s16(y_even_pixels_s16, gTmp_s16);
        float32x4_t gTmp_even_f32_qlow = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(gTmp_even))), scale_coeff);
        float32x4_t gTmp_even_f32_qhigh = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(gTmp_even))), scale_coeff);
        vst1q_f32(dst_even_base, gTmp_even_f32_qlow);
        vst1q_f32(dst_even_base + 4, gTmp_even_f32_qhigh);
        dst_even_base += dst_plane_size;

        int16x8_t gTmp_odd = vsubq_s16(y_odd_pixels_s16, gTmp_s16);
        float32x4_t gTmp_odd_f32_qlow = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(gTmp_odd))), scale_coeff);
        float32x4_t gTmp_odd_f32_qhigh = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(gTmp_odd))), scale_coeff);
        vst1q_f32(dst_odd_base, gTmp_odd_f32_qlow);
        vst1q_f32(dst_odd_base + 4, gTmp_odd_f32_qhigh);
        dst_odd_base += dst_plane_size;

        int16x8_t bTmp_even = vaddq_s16(y_even_pixels_s16, bTmp_s16);
        float32x4_t bTmp_even_f32_qlow = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(bTmp_even))), scale_coeff);
        float32x4_t bTmp_even_f32_qhigh = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(bTmp_even))), scale_coeff);
        vst1q_f32(dst_even_base, bTmp_even_f32_qlow);
        vst1q_f32(dst_even_base + 4, bTmp_even_f32_qhigh);
        dst_even_base += dst_plane_size;

        int16x8_t bTmp_odd = vaddq_s16(y_odd_pixels_s16, bTmp_s16);
        float32x4_t bTmp_odd_f32_qlow = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(bTmp_odd))), scale_coeff);
        float32x4_t bTmp_odd_f32_qhigh = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(bTmp_odd))), scale_coeff);
        vst1q_f32(dst_odd_base, bTmp_odd_f32_qlow);
        vst1q_f32(dst_odd_base + 4, bTmp_odd_f32_qhigh);
        dst_odd_base += dst_plane_size;
      }
    }

    Y_even_row_start += width * 2;
    Y_odd_row_start += width * 2;
    UV_row_start += width;
  }

  // std::ofstream ofs("focus.bin", std::ios::binary | std::ios::out);
  // ofs.write((const char*)dst, height * width * 3 * sizeof(float));
}
#else

void FocusTransformNEON(int height, int width, uint8_t *dst, uint8_t *src) {}

void FocusTransformNEONFuse(int height, int width, float *dst, uint8_t *src) {}

void CvtFocusNEONFuse(int height, int width, float *dst, uint8_t *src, float coeff) {}

void YUV420SP2RGBNEON(int height, int width, uint8_t *dst, uint8_t *src) {}

#endif //__ARM_NEON
