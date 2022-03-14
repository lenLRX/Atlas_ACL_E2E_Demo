#pragma once

#include "BYTETracker.h"

struct GridAndStride {
  int grid0;
  int grid1;
  int stride;
};

std::vector<Object>
detect_yolox(int model_h, int model_w, int num_grid, int num_class,
             float nms_thr, float score_thr, const float *feature,
             const std::vector<GridAndStride> &grid_strides);

std::vector<GridAndStride> generate_grids_and_stride(const int target_w,
                                                     const int target_h,
                                                     std::vector<int> &strides);
