#include "acl_model.h"
#include "app_profiler.h"
#include "util.h"

#include <sstream>

ACLModel::ACLModel(aclrtStream stream) : stream(stream) {}

aclError ACLModel::Init(const char *model_path) {
  path = model_path;
  size_t model_size;
  size_t weight_size;
  CHECK_ACL(aclmdlQuerySize(model_path, &model_size, &weight_size));
  CHECK_ACL(aclrtMalloc(&model_mem, model_size, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&model_weight, weight_size, ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK_ACL(aclmdlLoadFromFileWithMem(model_path, &model_id, model_mem,
                                      model_size, model_weight, weight_size));

  model_desc = aclmdlCreateDesc();
  aclmdlGetDesc(model_desc, model_id);

  size_t model_input_num = aclmdlGetNumInputs(model_desc);
  size_t model_output_num = aclmdlGetNumOutputs(model_desc);

  for (size_t i = 0; i < model_input_num; ++i) {
    size_t buffer_size = aclmdlGetInputSizeByIndex(model_desc, i);
    input_buffer_sizes.push_back(buffer_size);
  }

  for (size_t i = 0; i < model_output_num; ++i) {
    size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc, i);
    output_buffer_sizes.push_back(buffer_size);
  }

  loaded = true;
  return ACL_ERROR_NONE;
}

ACLModel::DevBufferVec ACLModel::Infer(const DevBufferVec &inputs) {
  DevBufferVec result;
  aclmdlDataset *input_dataset = aclmdlCreateDataset();
  aclmdlDataset *output_dataset = aclmdlCreateDataset();

  size_t model_input_num = aclmdlGetNumInputs(model_desc);
  size_t model_output_num = aclmdlGetNumOutputs(model_desc);

  for (size_t i = 0; i < model_input_num; ++i) {
    size_t buffer_size = aclmdlGetInputSizeByIndex(model_desc, i);
    aclDataBuffer *input_databuffer =
        aclCreateDataBuffer(inputs[i]->GetDevicePtr(), buffer_size);
    CHECK_ACL(aclmdlAddDatasetBuffer(input_dataset, input_databuffer));
  }

  for (size_t i = 0; i < model_output_num; ++i) {
    size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc, i);
    void *output_buffer;
    CHECK_ACL(
        aclrtMalloc(&output_buffer, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST));
    auto dev_buffer_ptr = std::make_shared<DeviceBuffer>(
        output_buffer, buffer_size, DeviceBuffer::DevMemDeleter());
    result.emplace_back(dev_buffer_ptr);
    aclDataBuffer *output_databuffer =
        aclCreateDataBuffer(output_buffer, buffer_size);
    CHECK_ACL(aclmdlAddDatasetBuffer(output_dataset, output_databuffer));
  }

  {
    APP_PROFILE(aclmdlExecuteAsync);
    CHECK_ACL(
        aclmdlExecuteAsync(model_id, input_dataset, output_dataset, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
  }

  for (size_t i = 0; i < model_input_num; ++i) {
    CHECK_ACL(aclDestroyDataBuffer(aclmdlGetDatasetBuffer(input_dataset, i)));
  }

  for (size_t i = 0; i < model_output_num; ++i) {
    CHECK_ACL(aclDestroyDataBuffer(aclmdlGetDatasetBuffer(output_dataset, i)));
  }

  CHECK_ACL(aclmdlDestroyDataset(input_dataset));
  CHECK_ACL(aclmdlDestroyDataset(output_dataset));

  return result;
}

void ACLModel::Infer(const DevBufferVec &inputs, const DevBufferVec &outputs) {
  aclmdlDataset *input_dataset = aclmdlCreateDataset();
  aclmdlDataset *output_dataset = aclmdlCreateDataset();

  size_t model_input_num = aclmdlGetNumInputs(model_desc);
  size_t model_output_num = aclmdlGetNumOutputs(model_desc);

  for (size_t i = 0; i < model_input_num; ++i) {
    size_t buffer_size = aclmdlGetInputSizeByIndex(model_desc, i);
    aclDataBuffer *input_databuffer =
        aclCreateDataBuffer(inputs[i]->GetDevicePtr(), buffer_size);
    CHECK_ACL(aclmdlAddDatasetBuffer(input_dataset, input_databuffer));
  }

  for (size_t i = 0; i < model_output_num; ++i) {
    size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc, i);
    aclDataBuffer *output_databuffer =
        aclCreateDataBuffer(outputs[i]->GetDevicePtr(), buffer_size);
    CHECK_ACL(aclmdlAddDatasetBuffer(output_dataset, output_databuffer));
  }

  {
    APP_PROFILE(aclmdlExecuteAsync);
    CHECK_ACL(
        aclmdlExecuteAsync(model_id, input_dataset, output_dataset, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
  }

  for (size_t i = 0; i < model_input_num; ++i) {
    CHECK_ACL(aclDestroyDataBuffer(aclmdlGetDatasetBuffer(input_dataset, i)));
  }

  for (size_t i = 0; i < model_output_num; ++i) {
    CHECK_ACL(aclDestroyDataBuffer(aclmdlGetDatasetBuffer(output_dataset, i)));
  }

  CHECK_ACL(aclmdlDestroyDataset(input_dataset));
  CHECK_ACL(aclmdlDestroyDataset(output_dataset));
}

const std::vector<size_t> &ACLModel::GetInputBufferSizes() {
  return input_buffer_sizes;
}

const std::vector<size_t> &ACLModel::GetOutputBufferSizes() {
  return output_buffer_sizes;
}

std::string ACLModel::ToString() {
  std::stringstream ss;
  ss << "ACLModel:" << path << "\n"
     << "Input Num:" << input_buffer_sizes.size() << "\n";
  ss << "Input shapes:";
  for (auto s : input_buffer_sizes) {
    ss << s << ", ";
  }
  ss << "\n";
  ss << "Output Num:" << output_buffer_sizes.size() << "\n";
  ss << "Output shapes:";
  for (auto s : output_buffer_sizes) {
    ss << s << ", ";
  }
  ss << "\n";
  return ss.str();
}
