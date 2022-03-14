
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "BYTETracker.h"
#include "bytetrack.h"
#include <chrono>
#include <float.h>
#include <stdio.h>
#include <vector>

static inline float intersection_area(const Object &a, const Object &b) {
  cv::Rect_<float> inter = a.rect & b.rect;
  return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left,
                                  int right) {
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].prob;

  while (i <= j) {
    while (faceobjects[i].prob > p)
      i++;

    while (faceobjects[j].prob < p)
      j--;

    if (i <= j) {
      // swap
      std::swap(faceobjects[i], faceobjects[j]);

      i++;
      j--;
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (left < j)
        qsort_descent_inplace(faceobjects, left, j);
    }
#pragma omp section
    {
      if (i < right)
        qsort_descent_inplace(faceobjects, i, right);
    }
  }
}

static void qsort_descent_inplace(std::vector<Object> &objects) {
  if (objects.empty())
    return;

  qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects,
                              std::vector<int> &picked, float nms_threshold) {
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].rect.area();
  }

  for (int i = 0; i < n; i++) {
    const Object &a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object &b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = intersection_area(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) {
        keep = 0;
        break;
      }
    }

    if (keep)
      picked.push_back(i);
  }
}

std::vector<GridAndStride>
generate_grids_and_stride(const int target_w, const int target_h,
                          std::vector<int> &strides) {
  std::vector<GridAndStride> grid_strides;
  for (int i = 0; i < (int)strides.size(); i++) {
    int stride = strides[i];
    int num_grid_w = target_w / stride;
    int num_grid_h = target_h / stride;
    for (int g1 = 0; g1 < num_grid_h; g1++) {
      for (int g0 = 0; g0 < num_grid_w; g0++) {
        GridAndStride gs;
        gs.grid0 = g0;
        gs.grid1 = g1;
        gs.stride = stride;
        grid_strides.push_back(gs);
      }
    }
  }
  return grid_strides;
}

static void generate_yolox_proposals(int num_grid, int num_class,
                                     std::vector<GridAndStride> grid_strides,
                                     const float *feat_ptr,
                                     float prob_threshold,
                                     std::vector<Object> &objects) {
  const int feature_w = num_class + 5;
  const int num_anchors = grid_strides.size();

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    // yolox/models/yolo_head.py decode logic
    //  outputs[..., :2] = (outputs[..., :2] + grids) * strides
    //  outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    float x_center = (feat_ptr[0] + grid0) * stride;
    float y_center = (feat_ptr[1] + grid1) * stride;
    float w = exp(feat_ptr[2]) * stride;
    float h = exp(feat_ptr[3]) * stride;
    float x0 = x_center - w * 0.5f;
    float y0 = y_center - h * 0.5f;

    float box_objectness = feat_ptr[4];
    for (int class_idx = 0; class_idx < num_class; class_idx++) {
      float box_cls_score = feat_ptr[5 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > prob_threshold) {
        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = w;
        obj.rect.height = h;
        obj.label = class_idx;
        obj.prob = box_prob;

        objects.push_back(obj);
      }

    } // class loop
    feat_ptr += feature_w;

  } // point anchor loop
}

std::vector<Object>
detect_yolox(int model_h, int model_w, int num_grid, int num_class,
             float nms_thr, float score_thr, const float *feature,
             const std::vector<GridAndStride> &grid_strides) {
  std::vector<Object> objects;

  std::vector<Object> proposals;

  generate_yolox_proposals(num_grid, num_class, grid_strides, feature,
                           score_thr, proposals);
  // sort all proposals by score from highest to lowest
  qsort_descent_inplace(proposals);

  // apply nms with nms_threshold
  std::vector<int> picked;
  nms_sorted_bboxes(proposals, picked, nms_thr);

  int count = picked.size();

  objects.resize(count);
  for (int i = 0; i < count; i++) {
    objects[i] = proposals[picked[i]];
  }

  return objects;
}

/*
int main(int argc, char** argv)
{

        //detect_yolox(img, objects);
        detect_yolox(in_pad, objects, ex, scale);
        vector<STrack> output_stracks = tracker.update(objects);
        for (int i = 0; i < output_stracks.size(); i++)
                {
                        vector<float> tlwh = output_stracks[i].tlwh;
                        bool vertical = tlwh[2] / tlwh[3] > 1.6;
                        if (tlwh[2] * tlwh[3] > 20 && !vertical)
                        {
                                Scalar s =
tracker.get_color(output_stracks[i].track_id);
                        }
                }
}
*/
