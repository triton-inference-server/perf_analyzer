// Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "triton_c_api_backend.h"

#include "c_api_infer_results.h"
#include "json_utils.h"
#include "triton_loader.h"

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritoncapi {

//==============================================================================

Error
TritonCApiClientBackend::Create(
    const std::string& triton_server_path,
    const std::string& model_repository_path, const bool verbose,
    std::unique_ptr<ClientBackend>* client_backend)
{
  if (triton_server_path.empty()) {
    return Error(
        "--triton-server-path should not be empty when using "
        "service-kind=triton_c_api.");
  }

  if (model_repository_path.empty()) {
    return Error(
        "--model-repository should not be empty when using "
        "service-kind=triton_c_api.");
  }

  std::unique_ptr<TritonCApiClientBackend> triton_client_backend(
      new TritonCApiClientBackend());
  RETURN_IF_ERROR(
      TritonLoader::Create(triton_server_path, model_repository_path, verbose));
  *client_backend = std::move(triton_client_backend);
  return Error::Success;
}

Error
TritonCApiClientBackend::ServerExtensions(std::set<std::string>* extensions)
{
  rapidjson::Document server_metadata_json;
  RETURN_IF_ERROR(triton_loader_->ServerMetaData(&server_metadata_json));
  for (const auto& extension : server_metadata_json["extensions"].GetArray()) {
    extensions->insert(
        std::string(extension.GetString(), extension.GetStringLength()));
  }
  return Error::Success;
}

Error
TritonCApiClientBackend::ModelMetadata(
    rapidjson::Document* model_metadata, const std::string& model_name,
    const std::string& model_version)
{
  if (!triton_loader_->ModelIsLoaded()) {
    triton_loader_->LoadModel(model_name, model_version);
  }
  RETURN_IF_ERROR(triton_loader_->ModelMetadata(model_metadata));
  return Error::Success;
}

Error
TritonCApiClientBackend::ModelConfig(
    rapidjson::Document* model_config, const std::string& model_name,
    const std::string& model_version)
{
  if (!triton_loader_->ModelIsLoaded()) {
    triton_loader_->LoadModel(model_name, model_version);
  }
  RETURN_IF_ERROR(
      triton_loader_->ModelConfig(model_config, model_name, model_version));
  return Error::Success;
}

Error
TritonCApiClientBackend::Infer(
    cb::InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  std::vector<tc::InferInput*> triton_inputs;
  ParseInferInputToTriton(inputs, &triton_inputs);

  std::vector<const tc::InferRequestedOutput*> triton_outputs;
  ParseInferRequestedOutputToTriton(outputs, &triton_outputs);

  tc::InferOptions triton_options(options.model_name_);
  ParseInferOptionsToTriton(options, &triton_options);

  capi::InferResult* triton_result;
  RETURN_IF_ERROR(triton_loader_->Infer(
      triton_options, triton_inputs, triton_outputs, &triton_result));

  *result = new TritonCApiInferResult(triton_result);
  return Error::Success;
}

Error
TritonCApiClientBackend::AsyncInfer(
    OnCompleteFn callback, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  auto wrapped_callback = [callback](capi::InferResult* client_result) {
    cb::InferResult* result = new TritonCApiInferResult(client_result);
    callback(result);
  };

  std::vector<tc::InferInput*> triton_inputs;
  ParseInferInputToTriton(inputs, &triton_inputs);

  std::vector<const tc::InferRequestedOutput*> triton_outputs;
  ParseInferRequestedOutputToTriton(outputs, &triton_outputs);

  tc::InferOptions triton_options(options.model_name_);
  ParseInferOptionsToTriton(options, &triton_options);

  RETURN_IF_ERROR(triton_loader_->AsyncInfer(
      wrapped_callback, triton_options, triton_inputs, triton_outputs,
      enable_stats_));

  return Error::Success;
}

Error
TritonCApiClientBackend::StartStream(OnCompleteFn callback, bool enable_stats)
{
  stream_callback_ = callback;
  enable_stats_ = enable_stats;
  return Error::Success;
}

Error
TritonCApiClientBackend::AsyncStreamInfer(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  auto wrapped_callback = [this](capi::InferResult* client_result) {
    cb::InferResult* result = new TritonCApiInferResult(client_result);
    stream_callback_(result);
  };

  std::vector<tc::InferInput*> triton_inputs;
  ParseInferInputToTriton(inputs, &triton_inputs);

  std::vector<const tc::InferRequestedOutput*> triton_outputs;
  ParseInferRequestedOutputToTriton(outputs, &triton_outputs);

  tc::InferOptions triton_options(options.model_name_);
  ParseInferOptionsToTriton(options, &triton_options);

  RETURN_IF_ERROR(triton_loader_->AsyncInfer(
      wrapped_callback, triton_options, triton_inputs, triton_outputs,
      enable_stats_));

  return Error::Success;
}

Error
TritonCApiClientBackend::ClientInferStat(InferStat* infer_stat)
{
  tc::InferStat triton_infer_stat;

  triton_loader_->ClientInferStat(&triton_infer_stat);
  ParseInferStat(triton_infer_stat, infer_stat);
  return Error::Success;
}

Error
TritonCApiClientBackend::ModelInferenceStatistics(
    std::map<ModelIdentifier, ModelStatistics>* model_stats,
    const std::string& model_name, const std::string& model_version)
{
  rapidjson::Document infer_stat_json;
  RETURN_IF_ERROR(triton_loader_->ModelInferenceStatistics(
      model_name, model_version, &infer_stat_json));
  ParseStatistics(infer_stat_json, model_stats);

  return Error::Success;
}

Error
TritonCApiClientBackend::UnregisterAllSharedMemory()
{
  RETURN_IF_ERROR(triton_loader_->UnregisterAllSharedMemory());
  return Error::Success;
}

Error
TritonCApiClientBackend::RegisterSystemMemory(
    const std::string& name, void* ptr, const size_t byte_size)
{
  RETURN_IF_ERROR(triton_loader_->RegisterSystemMemory(name, ptr, byte_size));
  return Error::Success;
}

Error
TritonCApiClientBackend::RegisterCudaMemory(
    const std::string& name, void* handle, const size_t byte_size)
{
  RETURN_IF_ERROR(triton_loader_->RegisterCudaMemory(name, handle, byte_size));
  return Error::Success;
}

void
TritonCApiClientBackend::ParseInferInputToTriton(
    const std::vector<InferInput*>& inputs,
    std::vector<tc::InferInput*>* triton_inputs)
{
  for (const auto input : inputs) {
    triton_inputs->push_back(
        (dynamic_cast<TritonCApiInferInput*>(input))->Get());
  }
}

void
TritonCApiClientBackend::ParseInferRequestedOutputToTriton(
    const std::vector<const InferRequestedOutput*>& outputs,
    std::vector<const tc::InferRequestedOutput*>* triton_outputs)
{
  for (const auto output : outputs) {
    triton_outputs->push_back(
        (dynamic_cast<const TritonCApiInferRequestedOutput*>(output))->Get());
  }
}

void
TritonCApiClientBackend::ParseInferOptionsToTriton(
    const InferOptions& options, tc::InferOptions* triton_options)
{
  triton_options->model_version_ = options.model_version_;
  triton_options->request_id_ = options.request_id_;
  if ((options.sequence_id_ != 0) || (options.sequence_id_str_ != "")) {
    if (options.sequence_id_ != 0) {
      triton_options->sequence_id_ = options.sequence_id_;
    } else {
      triton_options->sequence_id_str_ = options.sequence_id_str_;
    }
    triton_options->sequence_start_ = options.sequence_start_;
    triton_options->sequence_end_ = options.sequence_end_;
  }
}

void
TritonCApiClientBackend::ParseStatistics(
    const rapidjson::Document& infer_stat,
    std::map<ModelIdentifier, ModelStatistics>* model_stats)
{
  model_stats->clear();
  for (const auto& this_stat : infer_stat["model_stats"].GetArray()) {
    auto it = model_stats
                  ->emplace(
                      std::make_pair(
                          this_stat["name"].GetString(),
                          this_stat["version"].GetString()),
                      ModelStatistics())
                  .first;
    it->second.inference_count_ = this_stat["inference_count"].GetUint64();
    it->second.execution_count_ = this_stat["execution_count"].GetUint64();
    it->second.success_count_ =
        this_stat["inference_stats"]["success"]["count"].GetUint64();
    it->second.queue_count_ =
        this_stat["inference_stats"]["queue"]["count"].GetUint64();
    it->second.compute_input_count_ =
        this_stat["inference_stats"]["compute_input"]["count"].GetUint64();
    it->second.compute_infer_count_ =
        this_stat["inference_stats"]["compute_infer"]["count"].GetUint64();
    it->second.compute_output_count_ =
        this_stat["inference_stats"]["compute_output"]["count"].GetUint64();
    it->second.cumm_time_ns_ =
        this_stat["inference_stats"]["success"]["ns"].GetUint64();
    it->second.queue_time_ns_ =
        this_stat["inference_stats"]["queue"]["ns"].GetUint64();
    it->second.compute_input_time_ns_ =
        this_stat["inference_stats"]["compute_input"]["ns"].GetUint64();
    it->second.compute_infer_time_ns_ =
        this_stat["inference_stats"]["compute_infer"]["ns"].GetUint64();
    it->second.compute_output_time_ns_ =
        this_stat["inference_stats"]["compute_output"]["ns"].GetUint64();
    it->second.cache_hit_count_ =
        this_stat["inference_stats"]["cache_hit"]["count"].GetUint64();
    it->second.cache_hit_time_ns_ =
        this_stat["inference_stats"]["cache_hit"]["ns"].GetUint64();
    it->second.cache_miss_count_ =
        this_stat["inference_stats"]["cache_miss"]["count"].GetUint64();
    it->second.cache_miss_time_ns_ =
        this_stat["inference_stats"]["cache_miss"]["ns"].GetUint64();
  }
}

void
TritonCApiClientBackend::ParseInferStat(
    const tc::InferStat& triton_infer_stat, InferStat* infer_stat)
{
  infer_stat->completed_request_count =
      triton_infer_stat.completed_request_count;
  infer_stat->cumulative_total_request_time_ns =
      triton_infer_stat.cumulative_total_request_time_ns;
  infer_stat->cumulative_send_time_ns =
      triton_infer_stat.cumulative_send_time_ns;
  infer_stat->cumulative_receive_time_ns =
      triton_infer_stat.cumulative_receive_time_ns;
}

//==============================================================================

Error
TritonCApiInferInput::Create(
    InferInput** infer_input, const std::string& name,
    const std::vector<int64_t>& dims, const std::string& datatype)
{
  TritonCApiInferInput* local_infer_input =
      new TritonCApiInferInput(name, datatype);

  tc::InferInput* triton_infer_input;
  RETURN_IF_TRITON_ERROR(
      tc::InferInput::Create(&triton_infer_input, name, dims, datatype));
  local_infer_input->input_.reset(triton_infer_input);

  *infer_input = local_infer_input;
  return Error::Success;
}

const std::vector<int64_t>&
TritonCApiInferInput::Shape() const
{
  return input_->Shape();
}

Error
TritonCApiInferInput::SetShape(const std::vector<int64_t>& shape)
{
  RETURN_IF_TRITON_ERROR(input_->SetShape(shape));
  return Error::Success;
}

Error
TritonCApiInferInput::Reset()
{
  RETURN_IF_TRITON_ERROR(input_->Reset());
  return Error::Success;
}

Error
TritonCApiInferInput::AppendRaw(const uint8_t* input, size_t input_byte_size)
{
  RETURN_IF_TRITON_ERROR(input_->AppendRaw(input, input_byte_size));
  return Error::Success;
}

Error
TritonCApiInferInput::SetSharedMemory(
    const std::string& name, size_t byte_size, size_t offset)
{
  RETURN_IF_TRITON_ERROR(input_->SetSharedMemory(name, byte_size, offset));
  return Error::Success;
}

Error
TritonCApiInferInput::RawData(const uint8_t** buf, size_t* byte_size)
{
  RETURN_IF_TRITON_ERROR(input_->RawData(buf, byte_size));
  return Error::Success;
}

TritonCApiInferInput::TritonCApiInferInput(
    const std::string& name, const std::string& datatype)
    : InferInput(BackendKind::TRITON_C_API, name, datatype)
{
}


//==============================================================================

Error
TritonCApiInferRequestedOutput::Create(
    InferRequestedOutput** infer_output, const std::string& name,
    const size_t class_count, const std::string& datatype)
{
  TritonCApiInferRequestedOutput* local_infer_output =
      new TritonCApiInferRequestedOutput(name, datatype);

  tc::InferRequestedOutput* triton_infer_output;
  RETURN_IF_TRITON_ERROR(tc::InferRequestedOutput::Create(
      &triton_infer_output, name, class_count, datatype));
  local_infer_output->output_.reset(triton_infer_output);

  *infer_output = local_infer_output;

  return Error::Success;
}

Error
TritonCApiInferRequestedOutput::SetSharedMemory(
    const std::string& name, size_t byte_size, size_t offset)
{
  RETURN_IF_TRITON_ERROR(output_->SetSharedMemory(name, byte_size, offset));
  return Error::Success;
}

TritonCApiInferRequestedOutput::TritonCApiInferRequestedOutput(
    const std::string& name, const std::string& datatype)
    : InferRequestedOutput(BackendKind::TRITON_C_API, name, datatype)
{
}

//==============================================================================

TritonCApiInferResult::TritonCApiInferResult(capi::InferResult* result)
{
  result_.reset(result);
}

Error
TritonCApiInferResult::Id(std::string* id) const
{
  RETURN_IF_TRITON_ERROR(result_->Id(id));
  return Error::Success;
}

Error
TritonCApiInferResult::RequestStatus() const
{
  RETURN_IF_TRITON_ERROR(result_->RequestStatus());
  return Error::Success;
}

Error
TritonCApiInferResult::RawData(
    const std::string& output_name, std::vector<uint8_t>& buf) const
{
  RETURN_IF_TRITON_ERROR(result_->RawData(output_name, buf));
  return Error::Success;
}

Error
TritonCApiInferResult::IsFinalResponse(bool* is_final_response) const
{
  RETURN_IF_TRITON_ERROR(result_->IsFinalResponse(is_final_response));
  return Error::Success;
}

Error
TritonCApiInferResult::IsNullResponse(bool* is_null_response) const
{
  RETURN_IF_TRITON_ERROR(result_->IsNullResponse(is_null_response));
  return Error::Success;
}

//==============================================================================

}}}}  // namespace triton::perfanalyzer::clientbackend::tritoncapi
