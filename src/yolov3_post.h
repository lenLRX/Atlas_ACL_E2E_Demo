/**
 * ============================================================================
 *
 * Copyright (C) 2018, Hisilicon Technologies Co., Ltd. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#ifndef __YOLOV3_POST_H__
#define __YOLOV3_POST_H__

#include "acl/acl.h"
#include <stdint.h>
#include <vector>

#define sigmoid(x) (1 / (1 + exp(-1 * x)))

aclError yolov3_post(const float nmsThresh, const std::vector<void *> &inputs,
                     const std::vector<size_t> &sizes, int ImgH, int ImgW);

struct BoundingBox {
  uint32_t lt_x;
  uint32_t lt_y;
  uint32_t rb_x;
  uint32_t rb_y;
  uint32_t attribute;
  float score;
};

#if 0

template <class T>
class Tensor {
  public:
    Tensor() : data_(nullptr) {}
    ~Tensor() { Clear(); }

    uint32_t Size() const {
      uint32_t size(1);
      for(auto& dim : dims_){
        size *= dim;
      }
      return size;
    }

    bool FromArray(const T* pdata, const std::vector<uint32_t>& shape){
      if(pdata == nullptr){
        return false;
      }
      Clear();
      uint32_t size(1);
      for(auto dim:shape){
        if(dim>0){
          size *= dim;
        }else{
          return false;
        }
      }
      data_ = new T[size];
      int ret = memcpy_s(data_, size * sizeof(T), pdata, size * sizeof(T));
      if(ret !=0){
        return false;
      }
      dims_ = shape;
      return true;
    }

    T& operator()(uint32_t i, ...) { 
      va_list arg_ptr;
      va_start(arg_ptr, i);
      uint32_t index = i;
      for (uint32_t idx = 1; idx < dims_.size();++idx){
        index *= dims_[idx];
        index += va_arg(arg_ptr, uint32_t);
      }
      va_end(arg_ptr);
      return data_[index];
    }

    T& operator[](unsigned int index) { return data_[index]; }
    const T& operator[](unsigned int index) const { return data_[index]; }

  private:
    void Clear(){
      if(data_ != nullptr){
        delete[] data_;
        data_ = nullptr;
      }
      dims_.clear();
    }
    std::vector<uint32_t> dims_; // tensor shape
    T* data_; //tensor data
};

/**
 * @brief: inference engine class
 */
class GeneralPost : public hiai::Engine {
public:
  /**
   * @brief: engine initialize
   * @param [in]: engine's parameters which configured in graph.config
   * @param [in]: model description
   * @return: HIAI_StatusT
   */
  HIAI_StatusT Init(const hiai::AIConfig& config,
                    const std::vector<hiai::AIModelDescription>& model_desc);

#define sigmoid(x) (1 / (1 + exp(-1 * x)))

  /**
   * @brief: engine processor which override HIAI engine
   *         get every image, and then send data to inference engine
   * @param [in]: input size
   * @param [in]: output size
   */
  HIAI_DEFINE_PROCESS(INPUT_SIZE, OUTPUT_SIZE);

private:
  /**
   * @brief: send result
   * @return: true: success; false: failed
   */
  bool SendSentinel();

  /**
   * @brief: mark the oject based on detection result
   * @param [in]: result: engine transform image
   * @return: HIAI_StatusT
   */
  HIAI_StatusT PostProcess(
      const std::shared_ptr<EngineTrans> &result);

  std::vector<BoundingBox>  decodeTensor(
      const std::shared_ptr<EngineTrans> &result, uint ImgW,uint Imgh);
      
std::vector<BoundingBox> nonMaximumSuppression(const float nmsThresh, std::vector<BoundingBox> binfo);
  std::vector<BoundingBox> nmsAllClasses(const float nmsThresh, std::vector<BoundingBox>& binfo,
                                    const uint numClasses);
};

#endif

#endif //__YOLOV3_POST_H__
