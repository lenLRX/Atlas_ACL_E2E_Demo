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

#include "yolov3_post.h"

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

#include "opencv2/opencv.hpp"


using namespace std;


// num tensor shape
const static std::vector<uint32_t> kDimBBoxCnt = {32};
const static std::vector<string> label = {"background", "person", "bicycle", "car", "motorbike",
 "aeroplane","bus", "train", "truck", "boat", 
 "traffic light", "fire hydrant", "stop sign", "parking meter", 
 "bench", "bird", "cat", "dog", "horse", 
 "sheep", "cow", "elephant", "bear", "zebra", 
 "giraffe", "backpack", "umbrella", "handbag","tie", 
 "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", 
		"fork", "knife", "spoon", "bowl", "banana", 
		"apple", "sandwich", "orange", "broccoli", "carrot", 
		"hot dog", "pizza", "donut", "cake", "chair", 
		"sofa", "potted plant", "bed", "dining table", "toilet", 
		"TV monitor", "laptop", "mouse", "remote", "keyboard", 
		"cell phone", "microwave", "oven", "toaster", "sink", 
		"refrigerator", "book", "clock", "vase","scissors", 
		"teddy bear", "hair drier", "toothbrush"};

const double kFountScale = 0.5;
const cv::Scalar kFontColor(0, 0, 255);
const uint32_t kLabelOffset = 11;

const uint numClasses = 80; 
const uint BoxTensorLabel  = 85; 

const uint numBBoxes = 3;
const uint  BoxTensorLength = (BoxTensorLabel * numBBoxes);
const float nmsThresh = 0.45;
const float MaxBoxClassThresh = 0.1;
const float MaxClassThresh = 0.1;

const static std::vector<uint32_t>  anchors = {116,90,156,198,373,326,30,61,62,45,59,119,10,13,16,30,33,23};
const static std::vector<uint32_t>  kGridSize = {13,26,52};
const uint32_t kOutputTensorSize = 3;

std::vector<BoundingBox> nmsAllClasses(const float nmsThresh, std::vector<BoundingBox>& binfo, const uint numClasses);
std::vector<BoundingBox> nonMaximumSuppression(const float nmsThresh, std::vector<BoundingBox> binfo);
std::vector<BoundingBox> decodeTensor(const std::vector<void*>& inputs, const std::vector<size_t>& sizes, int ImgH, int ImgW);


aclError yolov3_post(const float nmsThresh,
 const std::vector<void*>& inputs,
 const std::vector<size_t>& sizes,
 int ImgH, int ImgW) {
    auto bbox1 = decodeTensor(inputs, sizes, ImgH, ImgW);
    std::cout << "bbox1_size " << bbox1.size() << std::endl;
    auto bbox2 = nmsAllClasses(nmsThresh, bbox1, numClasses);
    std::cout << "yolov3_post box size " << bbox2.size() << std::endl;
}

std::vector<BoundingBox> decodeTensor(const std::vector<void*>& inputs,
const std::vector<size_t>& sizes, int ImgH, int ImgW) {
  std::vector<BoundingBox> binfo;
    for(uint32_t ImgIndex=0; ImgIndex < kOutputTensorSize;ImgIndex++) 
    {
      uint gridSize = kGridSize[ImgIndex];
      float* out = (float*)inputs[ImgIndex];
      int32_t size = sizes[ImgIndex] / sizeof(float); 
    
      float *res = new (nothrow) float[size];
    
      for(int i = 0 ;i < BoxTensorLength;i++)
      {
        for(int j = 0;j < gridSize;j++)
        {
          for(int k = 0; k < gridSize; k++)
          {
            if ((((k * gridSize +j) * BoxTensorLength + i )>= size) ||(((i*gridSize+j)*gridSize +k)>=size) )
            {
              std::cerr << "invalid binfo" <<  std::endl;
                return binfo;
            }

            res[(k * gridSize +j) * BoxTensorLength + i] = out[(i*gridSize+j)*gridSize +k];
          }
        }
      }

      int count =1;         
      for ( uint cx = 0; cx < gridSize; cx++)
      {  
        for(uint cy = 0; cy < gridSize; cy++)
        {   
            float MaxBoxProb = 0;            
            float bx = 0;
            float by = 0;
            float bw = 0;
            float bh = 0;
            float tx ;
            float ty ;
            float tw ;
            float th;
            float cf;
            
                        
            for (uint i = 0; i  < numBBoxes; ++i)
            {   
                const int bbindex = BoxTensorLength*(cx * gridSize + cy);

                tx =  res[bbindex+i * BoxTensorLabel + 0];
                ty =  res[bbindex+i * BoxTensorLabel + 1];
                tw =  res[bbindex+i * BoxTensorLabel + 2];
                th =  res[bbindex+i * BoxTensorLabel + 3];
                cf =  res[bbindex+i * BoxTensorLabel + 4];
                cf = sigmoid(cf);
                
                float MaxClass =0.0f;
                uint32_t MaxClass_Loc = 0;
                for (int j = 5;j< BoxTensorLabel; j++)
                {
                    float class_prob = sigmoid(res[bbindex+ (i * BoxTensorLabel + j)]);
                    if(MaxClass < class_prob)
                    {
                      MaxClass = class_prob;
                      MaxClass_Loc = j - 5;
                    }
                }
                bx = (sigmoid(tx)+cx)/(gridSize);
                by = (sigmoid(ty)+cy)/(gridSize);
                bw = anchors[i*2+ ImgIndex*6] *exp(tw)/ImgW;
                bh = anchors[i*2+1+ImgIndex*6] *exp(th)/ImgH;

                //std::cout << "class_prob " << MaxClass << std::endl;
              
                /*count++;
                if(cf* MaxClass >MaxBoxProb)
                {
                  MaxBoxProb = cf* MaxClass;
                } */               
                
                if( ( cf * MaxClass > MaxBoxClassThresh)&&( MaxClass > MaxClassThresh ))
                {
                  uint32_t x1 = ((bx-bw/2))*ImgW;
                  uint32_t y1 = ((by-bh/2))*ImgH;
                  uint32_t x2 = ((bx+bw/2))*ImgW;
                  uint32_t y2 = ((by+bh/2))*ImgH; 
                  std::cout << "add box class: " << label[MaxClass_Loc] << " cf " <<
                    cf << " cf x MaxClass" << cf*MaxClass << " confidence: " << MaxClass << std::endl;
                  binfo.push_back({x1, y1,x2, y2, MaxClass_Loc, MaxClass});
                }

            }
          }
      }    
      delete[] res;
    }
    return binfo;
}

std::vector<BoundingBox> nonMaximumSuppression(const float nmsThresh, std::vector<BoundingBox> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BoundingBox& bbox1, BoundingBox& bbox2) -> float {
        float overlapX = overlap1D(bbox1.lt_x, bbox1.rb_x, bbox2.lt_x, bbox2.rb_x);
        float overlapY = overlap1D(bbox1.lt_y, bbox1.rb_y, bbox2.lt_y, bbox2.rb_y);
        float area1 = (bbox1.rb_x - bbox1.lt_x) * (bbox1.rb_y - bbox1.lt_y);
        float area2 = (bbox2.rb_x - bbox2.lt_x) * (bbox2.rb_y - bbox2.lt_y);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const BoundingBox& b1, const BoundingBox& b2) { return b1.score > b2.score; });
    std::vector<BoundingBox> out;
    for (auto& i : binfo)
    {
        bool keep = true;
        for (auto& j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}


std::vector<BoundingBox> nmsAllClasses(const float nmsThresh, std::vector<BoundingBox>& binfo, const uint numClasses)
{
    std::vector<BoundingBox> result;
    std::vector<std::vector<BoundingBox>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.attribute).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }

    return result;
}

/*
namespace {
// callback port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;

// size of output tensor vector should be 2.
const uint32_t kOutputTensorSize = 3;
const uint32_t kOutputNumIndex = 0;
const uint32_t kOutputTesnorIndex = 1;

const uint32_t kCategoryIndex = 2;
const uint32_t kScorePrecision = 3;

// bounding box line solid
const uint32_t kLineSolid = 2;

// output image prefix
const string kOutputFilePrefix = "out_";


// num tensor shape
const static std::vector<uint32_t> kDimBBoxCnt = {32};
const static std::vector<string> label = {"background", "person", "bicycle", "car", "motorbike",
 "aeroplane","bus", "train", "truck", "boat", 
 "traffic light", "fire hydrant", "stop sign", "parking meter", 
 "bench", "bird", "cat", "dog", "horse", 
 "sheep", "cow", "elephant", "bear", "zebra", 
 "giraffe", "backpack", "umbrella", "handbag","tie", 
 "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", 
		"fork", "knife", "spoon", "bowl", "banana", 
		"apple", "sandwich", "orange", "broccoli", "carrot", 
		"hot dog", "pizza", "donut", "cake", "chair", 
		"sofa", "potted plant", "bed", "dining table", "toilet", 
		"TV monitor", "laptop", "mouse", "remote", "keyboard", 
		"cell phone", "microwave", "oven", "toaster", "sink", 
		"refrigerator", "book", "clock", "vase","scissors", 
		"teddy bear", "hair drier", "toothbrush"};

// opencv draw label params.
const double kFountScale = 0.5;
const cv::Scalar kFontColor(0, 0, 255);
const uint32_t kLabelOffset = 11;

const uint numClasses = 80; 
const uint BoxTensorLabel  = 85; 

const uint numBBoxes = 3;
const uint  BoxTensorLength = (BoxTensorLabel * numBBoxes);
const float nmsThresh = 0.45;
const float MaxBoxClassThresh = 0.25;
const float MaxClassThresh = 0.6;


const string kFileSperator = "/";

// opencv color list for boundingbox
const vector<cv::Scalar> kColors {
  cv::Scalar(237, 149, 100), cv::Scalar(0, 215, 255), cv::Scalar(50, 205, 50),
  cv::Scalar(139, 85, 26)};
// output tensor index
enum BBoxIndex {kTopLeftX, kTopLeftY, kLowerRigltX, kLowerRightY, kScore};
//const static std::vector<uint32_t>  

const static std::vector<uint32_t>  anchors = {116,90,156,198,373,326,30,61,62,45,59,119,10,13,16,30,33,23};
const static std::vector<uint32_t>  kGridSize = {13,26,52};

}
 // namespace



HIAI_StatusT GeneralPost::Init(
  const hiai::AIConfig &config,
  const vector<hiai::AIModelDescription> &model_desc) {
  // do noting
  return HIAI_OK;
}

bool GeneralPost::SendSentinel() {
  // can not discard when queue full
  HIAI_StatusT hiai_ret = HIAI_OK;
  shared_ptr<string> sentinel_msg(new (nothrow) string);
  do {
    hiai_ret = SendData(kSendDataPort, "string",
                        static_pointer_cast<void>(sentinel_msg));
    // when queue full, sleep
    if (hiai_ret == HIAI_QUEUE_FULL) {
      HIAI_ENGINE_LOG("queue full, sleep 200ms");
      usleep(kSleepInterval);
    }
  } while (hiai_ret == HIAI_QUEUE_FULL);

  // send failed
  if (hiai_ret != HIAI_OK) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "call SendData failed, err_code=%d", hiai_ret);
    return false;
  }
  return true;
}
std::vector<BoundingBox> GeneralPost::nonMaximumSuppression(const float nmsThresh, std::vector<BoundingBox> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BoundingBox& bbox1, BoundingBox& bbox2) -> float {
        float overlapX = overlap1D(bbox1.lt_x, bbox1.rb_x, bbox2.lt_x, bbox2.rb_x);
        float overlapY = overlap1D(bbox1.lt_y, bbox1.rb_y, bbox2.lt_y, bbox2.rb_y);
        float area1 = (bbox1.rb_x - bbox1.lt_x) * (bbox1.rb_y - bbox1.lt_y);
        float area2 = (bbox2.rb_x - bbox2.lt_x) * (bbox2.rb_y - bbox2.lt_y);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::stable_sort(binfo.begin(), binfo.end(),
                     [](const BoundingBox& b1, const BoundingBox& b2) { return b1.score > b2.score; });
    std::vector<BoundingBox> out;
    for (auto& i : binfo)
    {
        bool keep = true;
        for (auto& j : out)
        {
            if (keep)
            {
                float overlap = computeIoU(i, j);
                keep = overlap <= nmsThresh;
            }
            else
                break;
        }
        if (keep) out.push_back(i);
    }
    return out;
}
std::vector<BoundingBox> GeneralPost::nmsAllClasses(const float nmsThresh, std::vector<BoundingBox>& binfo, const uint numClasses)
{
    std::vector<BoundingBox> result;
    std::vector<std::vector<BoundingBox>> splitBoxes(numClasses);
    for (auto& box : binfo)
    {
        splitBoxes.at(box.attribute).push_back(box);
    }

    for (auto& boxes : splitBoxes)
    {
        boxes = nonMaximumSuppression(nmsThresh, boxes);
        result.insert(result.end(), boxes.begin(), boxes.end());
    }

    return result;
}
std::vector<BoundingBox>  GeneralPost::decodeTensor(const shared_ptr<EngineTrans> &result,uint ImgW, uint Imgh)
{
    std::vector<BoundingBox> binfo;  
	
    if(kOutputTensorSize >3)
    {		
        ERROR_LOG("kOutputTensorSize   does not match.");
        return binfo;
    }
	
    for(uint ImgIndex =0; ImgIndex < kOutputTensorSize ;ImgIndex ++) 
    {  
      uint gridSize = kGridSize[ImgIndex];
      Output out = result->inference_res[ImgIndex];
      int32_t  size= out.size / sizeof(float); 
      float *resold = new (nothrow) float[size];
    
      error_t mem_ret =  memcpy_s(resold, out.size , out.data.get(), out.size );
      float *res = new (nothrow) float[size];
    
      for(int i = 0 ;i < BoxTensorLength;i++)
      {
        for(int j = 0;j < gridSize;j++)
        {
          for(int k = 0; k < gridSize; k++)
          {
            if ((((k * gridSize +j) * BoxTensorLength + i )>= size) ||(((i*gridSize+j)*gridSize +k)>=size) )
            {
                ERROR_LOG("kOutputTensorSize   does not match.");
                return binfo;
            }

            res[(k * gridSize +j) * BoxTensorLength + i] = resold[(i*gridSize+j)*gridSize +k];
          }
        }
      }

      int count =1;         
      for ( uint cx = 0; cx < gridSize; cx++)
      {  
        for(uint cy = 0; cy < gridSize; cy++)
        {   
            float MaxBoxProb = 0;            
            float bx = 0;
            float by = 0;
            float bw = 0;
            float bh = 0;
            float tx ;
            float ty ;
            float tw ;
            float th;
            float cf;
            
                        
            for (uint i = 0; i  < numBBoxes; ++i)
            {   
                const int bbindex = BoxTensorLength*(cx * gridSize + cy);

                tx =  res[bbindex+i * BoxTensorLabel + 0];
                ty =  res[bbindex+i * BoxTensorLabel + 1];
                tw =  res[bbindex+i * BoxTensorLabel + 2];
                th =  res[bbindex+i * BoxTensorLabel + 3];
                cf =  res[bbindex+i * BoxTensorLabel + 4];
                cf = Round(sigmoid(cf), 3);
                
                float MaxClass =0.0f;
                uint32_t MaxClass_Loc = 0;
                for (int j = 5;j< BoxTensorLabel; j++)
                {
                    float class_prob =  Round(sigmoid(res[bbindex+ (i * BoxTensorLabel + j)]), 3);
                    if(MaxClass < class_prob)
                    {
                      MaxClass = class_prob;
                      MaxClass_Loc = j - 5;
                    }
                }
                bx = (sigmoid(tx)+cx)/(gridSize);
                by = (sigmoid(ty)+cy)/(gridSize);
                bw = anchors[i*2+ ImgIndex*6] *exp(tw)/ImgW;
                bh = anchors[i*2+1+ImgIndex*6] *exp(th)/Imgh;                  
                
                if( ( cf * MaxClass > MaxBoxClassThresh)&&( MaxClass > MaxClassThresh ))
                {
                  uint32_t x1 = ((bx-bw/2))*ImgW;
                  uint32_t y1 = ((by-bh/2))*Imgh;
                  uint32_t x2 = ((bx+bw/2))*ImgW;
                  uint32_t y2 = ((by+bh/2))*Imgh; 
                  binfo.push_back({x1, y1,x2, y2, MaxClass_Loc, MaxClass});
                  // printf("rect: %d %d %d %d %f, class: %d %f, Prob(box*class): %f \n",
                  //         x1, y1, x2, y2, cf, MaxClass_Loc, MaxClass, cf * MaxClass);
                }

            }
          }
      }    
  	  delete[] res;
  	  delete[] resold;
    }
    
    return binfo;
}

HIAI_StatusT GeneralPost::PostProcess(
  const shared_ptr<EngineTrans> &result) {
  vector<Output> outputs = result->inference_res;
  
  if (outputs.size() != kOutputTensorSize) {
    ERROR_LOG("Detection output size does not match.");
    return HIAI_ERROR;
  }
  
  cv::Mat mat = cv::imread(result->image_info.path, CV_LOAD_IMAGE_UNCHANGED);

  float scale_width = (float)mat.cols / result->image_info.width;
  float scale_height = (float)mat.rows / result->image_info.height;

  if (mat.empty()) {
    ERROR_LOG("Fialed to deal file=%s. Reason: read image failed.",
              result->image_info.path.c_str());
    return HIAI_ERROR;
  }
  printf(" %d %d \n",mat.cols,mat.rows);

  stringstream sstream;
  
  vector<BoundingBox> bboxesOld,bboxesNew;
  bboxesOld = decodeTensor(result,result->console_params.model_width,result->console_params.model_height);
  bboxesNew = nmsAllClasses(nmsThresh, bboxesOld,numClasses);

  for (int i = 0; i < bboxesNew.size(); ++i) {
    cv::Point p1, p2;
    p1.x =   scale_width  * bboxesNew[i].lt_x;
    p1.y =  scale_height * bboxesNew[i].lt_y;
    p2.x = scale_width  * bboxesNew[i].rb_x;
    p2.y = scale_height * bboxesNew[i].rb_y;
    cv::rectangle(mat, p1, p2, kColors[i % kColors.size()], kLineSolid);

    sstream.str("");
    sstream << label[1+bboxesNew[i].attribute] << " ";
    sstream.precision(kScorePrecision);
    sstream << 100 * bboxesNew[i].score << "%";
    string obj_str = sstream.str();    
    cv::putText(mat, obj_str, cv::Point(p1.x, p1.y + kLabelOffset),
                cv::FONT_HERSHEY_COMPLEX, kFountScale, kFontColor);

    printf("%s %f \n",label[1+bboxesNew[i].attribute].c_str(), 100 * bboxesNew[i].score);
                
  }

  int pos = result->image_info.path.find_last_of(kFileSperator);
  string file_name(result->image_info.path.substr(pos + 1));
  bool save_ret(true);
  sstream.str("");
  sstream << result->console_params.output_path << kFileSperator
          << kOutputFilePrefix << file_name;
  string output_path = sstream.str();
  save_ret = cv::imwrite(output_path, mat);
  if (!save_ret) {
    ERROR_LOG("Failed to deal file=%s. Reason: save image failed.",
              result->image_info.path.c_str());
    return HIAI_ERROR;
  }
  return HIAI_OK;
}

HIAI_IMPL_ENGINE_PROCESS("general_post", GeneralPost, INPUT_SIZE) {
HIAI_StatusT ret = HIAI_OK;

// check arg0
if (arg0 == nullptr) {
  ERROR_LOG("Failed to deal file=nothing. Reason: arg0 is empty.");
  return HIAI_ERROR;
}

// just send to callback function when finished
shared_ptr<EngineTrans> result = static_pointer_cast<EngineTrans>(arg0);
if (result->is_finished) {
  if (SendSentinel()) {
    return HIAI_OK;
  }
  ERROR_LOG("Failed to send finish data. Reason: SendData failed.");
  ERROR_LOG("Please stop this process manually.");
  return HIAI_ERROR;
}

// inference failed
if (result->err_msg.error) {
  ERROR_LOG("%s", result->err_msg.err_msg.c_str());
  return HIAI_ERROR;
}

// arrange result
  return PostProcess(result);
}

*/