#ifndef __ACL_MODEL_H__
#define __ACL_MODEL_H__

#include "acl/acl.h"

#include <vector>
#include <string>

class ACLModel {
public:
    ACLModel(aclrtStream stream);
    aclError Init(const char* model_path);
    aclError Infer();
    // TODO avoid memcpy
    const std::vector<void*>& GetInputBuffer();
    const std::vector<void*>& GetOutputBuffer();
    const std::vector<size_t>& GetInputBufferSizes();
    const std::vector<size_t>& GetOutputBufferSizes();
    std::string ToString();
    //~ACLModel(); //TODO
private:
    std::string path;
    uint32_t model_id;
    void *model_mem;
    void *model_weight;
    bool loaded;  // model load flag
    aclmdlDesc *model_desc;
    aclmdlDataset* input_dataset;
    aclmdlDataset* output_dataset;

    std::vector<void*> input_buffers;
    std::vector<size_t> input_buffer_sizes;
    std::vector<void*> output_buffers;
    std::vector<size_t> output_buffer_sizes;

    aclrtStream stream;
};

#endif//__ACL_MODEL_H__