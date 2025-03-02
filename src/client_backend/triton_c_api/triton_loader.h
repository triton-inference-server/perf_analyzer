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
#pragma once

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "../client_backend.h"
#include "alloc_payload.h"
#include "common.h"
#include "response_output.h"
#include "shared_library.h"
#include "shared_memory_manager.h"
#include "triton/core/tritonserver.h"

// If TRITONSERVER error is non-OK, return the corresponding status.
#define RETURN_IF_TRITONSERVER_ERROR(E, MSG)                                \
  do {                                                                      \
    TRITONSERVER_Error* err__ = (E);                                        \
    if (err__ != nullptr) {                                                 \
      std::cout << "error: " << (MSG) << ": "                               \
                << GetSingleton()->error_code_to_string_fn_(err__) << " - " \
                << GetSingleton()->error_message_fn_(err__) << std::endl;   \
      Error newErr = Error(MSG);                                            \
      GetSingleton()->error_delete_fn_(err__);                              \
      return newErr;                                                        \
    }                                                                       \
  } while (false)

#define FAIL_IF_TRITONSERVER_ERROR(E, MSG)                                  \
  do {                                                                      \
    TRITONSERVER_Error* err__ = (E);                                        \
    if (err__ != nullptr) {                                                 \
      std::cerr << "error: " << (MSG) << ": "                               \
                << GetSingleton()->error_code_to_string_fn_(err__) << " - " \
                << GetSingleton()->error_message_fn_(err__) << std::endl;   \
      Error newErr = Error(MSG);                                            \
      GetSingleton()->error_delete_fn_(err__);                              \
      exit(1);                                                              \
    }                                                                       \
  } while (false)

#define REPORT_TRITONSERVER_ERROR(E, MSG)                                 \
  do {                                                                    \
    TRITONSERVER_Error* err__ = (E);                                      \
    if (err__ != nullptr) {                                               \
      std::cerr << "error: " << (MSG) << ": "                             \
                << GetSingleton()->error_message_fn_(err__) << std::endl; \
      GetSingleton()->error_delete_fn_(err__);                            \
    }                                                                     \
  } while (false)

namespace tc = triton::client;

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritoncapi {

class InferResult;

class TritonLoader : public tc::InferenceServerClient {
 public:
  using OnCompleteFn = std::function<void(InferResult*)>;

  struct AsyncRequestInfo {
    std::unique_ptr<AllocPayload> alloc_payload{};
    std::string request_id{};
    std::shared_ptr<tc::RequestTimers> timer{};
    OnCompleteFn callback{};
    bool enable_stats{true};
  };

  ~TritonLoader();

  static Error Create(
      const std::string& triton_server_path,
      const std::string& model_repository_path, bool verbose);

  Error Delete();
  Error StartTriton();

  Error LoadModel(
      const std::string& model_name, const std::string& model_version);

  Error ModelMetadata(rapidjson::Document* model_metadata);

  Error ModelConfig(
      rapidjson::Document* model_config, const std::string& model_name,
      const std::string& model_version);

  Error ServerMetaData(rapidjson::Document* server_metadata);

  Error Infer(
      const tc::InferOptions& options,
      const std::vector<tc::InferInput*>& inputs,
      const std::vector<const tc::InferRequestedOutput*>& outputs,
      InferResult** result);

  void InferResponseCompleteAsync(
      TRITONSERVER_InferenceResponse* response, const uint32_t flags,
      AsyncRequestInfo* async_request_info);

  Error AsyncInfer(
      OnCompleteFn callback, const tc::InferOptions& options,
      const std::vector<tc::InferInput*>& inputs,
      const std::vector<const tc::InferRequestedOutput*>& outputs,
      bool enable_stats);

  Error CleanUp(TRITONSERVER_InferenceResponse* completed_response);

  Error ModelInferenceStatistics(
      const std::string& model_name, const std::string& model_version,
      rapidjson::Document* infer_stat);

  Error ClientInferStat(tc::InferStat* infer_stat)
  {
    *infer_stat = infer_stat_;
    return Error::Success;
  }

  Error RegisterCudaMemory(
      const std::string& name, void* handle, const size_t byte_size);

  Error RegisterSystemMemory(
      const std::string& name, void* ptr, const size_t byte_size);

  Error UnregisterAllSharedMemory();

  TRITONSERVER_Error* ErrorNew(
      TRITONSERVER_Error_Code code, const char* message);

  bool ModelIsLoaded() { return model_is_loaded_; }
  bool ServerIsReady() { return server_is_ready_; }

  TRITONSERVER_Error* DeleteInferRequest(
      TRITONSERVER_InferenceRequest* irequest)
  {
    return request_delete_fn_(irequest);
  }
  static TritonLoader* GetSingleton();

  // TRITONSERVER_ApiVersion
  typedef TRITONSERVER_Error* (*TritonServerApiVersionFn_t)(
      uint32_t* major, uint32_t* minor);
  // TRITONSERVER_ServerOptionsNew
  typedef TRITONSERVER_Error* (*TritonServerOptionsNewFn_t)(
      TRITONSERVER_ServerOptions** options);
  // TRITONSERVER_ServerOptionsSetModelRepositoryPath
  typedef TRITONSERVER_Error* (*TritonServerOptionSetModelRepoPathFn_t)(
      TRITONSERVER_ServerOptions* options, const char* model_repository_path);
  // TRITONSERVER_ServerOptionsSetLogVerbose
  typedef TRITONSERVER_Error* (*TritonServerSetLogVerboseFn_t)(
      TRITONSERVER_ServerOptions* options, int level);

  // TRITONSERVER_ServerOptionsSetBackendDirectory
  typedef TRITONSERVER_Error* (*TritonServerSetBackendDirFn_t)(
      TRITONSERVER_ServerOptions* options, const char* backend_dir);

  // TRITONSERVER_ServerOptionsSetRepoAgentDirectory
  typedef TRITONSERVER_Error* (*TritonServerSetRepoAgentDirFn_t)(
      TRITONSERVER_ServerOptions* options, const char* repoagent_dir);

  // TRITONSERVER_ServerOptionsSetStrictModelConfig
  typedef TRITONSERVER_Error* (*TritonServerSetStrictModelConfigFn_t)(
      TRITONSERVER_ServerOptions* options, bool strict);

  // TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability
  typedef TRITONSERVER_Error* (
      *TritonServerSetMinSupportedComputeCapabilityFn_t)(
      TRITONSERVER_ServerOptions* options, double cc);

  // TRITONSERVER_ServerNew
  typedef TRITONSERVER_Error* (*TritonServerNewFn_t)(
      TRITONSERVER_Server** server, TRITONSERVER_ServerOptions* option);

  // TRITONSERVER_ServerOptionsDelete
  typedef TRITONSERVER_Error* (*TritonServerOptionsDeleteFn_t)(
      TRITONSERVER_ServerOptions* options);

  // TRITONSERVER_ServerDelete
  typedef TRITONSERVER_Error* (*TritonServerDeleteFn_t)(
      TRITONSERVER_Server* server);

  // TRITONSERVER_ServerIsLive
  typedef TRITONSERVER_Error* (*TritonServerIsLiveFn_t)(
      TRITONSERVER_Server* server, bool* live);

  // TRITONSERVER_ServerIsReady
  typedef TRITONSERVER_Error* (*TritonServerIsReadyFn_t)(
      TRITONSERVER_Server* server, bool* ready);

  // TRITONSERVER_ServerMetadata
  typedef TRITONSERVER_Error* (*TritonServerMetadataFn_t)(
      TRITONSERVER_Server* server, TRITONSERVER_Message** server_metadata);

  // TRITONSERVER_MessageSerializeToJson
  typedef TRITONSERVER_Error* (*TritonServerMessageSerializeToJsonFn_t)(
      TRITONSERVER_Message* message, const char** base, size_t* byte_size);

  // TRITONSERVER_MessageDelete
  typedef TRITONSERVER_Error* (*TritonServerMessageDeleteFn_t)(
      TRITONSERVER_Message* message);

  // TRITONSERVER_ServerModelIsReady
  typedef TRITONSERVER_Error* (*TritonServerModelIsReadyFn_t)(
      TRITONSERVER_Server* server, const char* model_name,
      const int64_t model_version, bool* ready);

  // TRITONSERVER_ServerModelMetadata
  typedef TRITONSERVER_Error* (*TritonServerModelMetadataFn_t)(
      TRITONSERVER_Server* server, const char* model_name,
      const int64_t model_version, TRITONSERVER_Message** model_metadata);

  // TRITONSERVER_ResponseAllocatorNew
  typedef TRITONSERVER_Error* (*TritonServerResponseAllocatorNewFn_t)(
      TRITONSERVER_ResponseAllocator** allocator,
      TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
      TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn,
      TRITONSERVER_ResponseAllocatorStartFn_t start_fn);

  // TRITONSERVER_InferenceRequestNew
  typedef TRITONSERVER_Error* (*TritonServerInferenceRequestNewFn_t)(
      TRITONSERVER_InferenceRequest** inference_request,
      TRITONSERVER_Server* server, const char* model_name,
      const int64_t model_version);

  // TRITONSERVER_InferenceRequestSetId
  typedef TRITONSERVER_Error* (*TritonServerInferenceRequestSetIdFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char* id);

  // TRITONSERVER_InferenceRequestSetReleaseCallback
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestSetReleaseCallbackFn_t)(
      TRITONSERVER_InferenceRequest* inference_request,
      TRITONSERVER_InferenceRequestReleaseFn_t request_release_fn,
      void* request_release_userp);

  // TRITONSERVER_InferenceRequestAddInput
  typedef TRITONSERVER_Error* (*TritonServerInferenceRequestAddInputFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char* name,
      const TRITONSERVER_DataType datatype, const int64_t* shape,
      uint64_t dim_count);

  // TRITONSERVER_InferenceRequestAddRequestedOutput
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestAddRequestedOutputFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char* name);

  // TRITONSERVER_InferenceRequestAppendInputData
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestAppendInputDataFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char* name,
      const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_i);

  // TRITONSERVER_InferenceRequestSetResponseCallback
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestSetResponseCallbackFn_t)(
      TRITONSERVER_InferenceRequest* inference_request,
      TRITONSERVER_ResponseAllocator* response_allocator,
      void* response_allocator_userp,
      TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
      void* response_userp);

  // TRITONSERVER_ServerInferAsync
  typedef TRITONSERVER_Error* (*TritonServerInferAsyncFn_t)(
      TRITONSERVER_Server* server,
      TRITONSERVER_InferenceRequest* inference_request,
      TRITONSERVER_InferenceTrace* trace);

  // TRITONSERVER_InferenceResponseError
  typedef TRITONSERVER_Error* (*TritonServerInferenceResponseErrorFn_t)(
      TRITONSERVER_InferenceResponse* inference_response);

  // TRITONSERVER_InferenceResponseDelete
  typedef TRITONSERVER_Error* (*TritonServerInferenceResponseDeleteFn_t)(
      TRITONSERVER_InferenceResponse* inference_response);

  // TRITONSERVER_InferenceRequestRemoveAllInputData
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestRemoveAllInputDataFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char* name);

  // TRITONSERVER_ResponseAllocatorDelete
  typedef TRITONSERVER_Error* (*TritonServerResponseAllocatorDeleteFn_t)(
      TRITONSERVER_ResponseAllocator* allocator);

  // TRITONSERVER_ErrorNew
  typedef TRITONSERVER_Error* (*TritonServerErrorNewFn_t)(
      TRITONSERVER_Error_Code code, const char* msg);

  // TRITONSERVER_MemoryTypeString
  typedef const char* (*TritonServerMemoryTypeStringFn_t)(
      TRITONSERVER_MemoryType memtype);

  // TRITONSERVER_InferenceResponseOutputCount
  typedef TRITONSERVER_Error* (*TritonServerInferenceResponseOutputCountFn_t)(
      TRITONSERVER_InferenceResponse* inference_response, uint32_t* count);

  // TRITONSERVER_DataTypeString
  typedef const char* (*TritonServerDataTypeStringFn_t)(
      TRITONSERVER_DataType datatype);

  // TRITONSERVER_ErrorMessage
  typedef const char* (*TritonServerErrorMessageFn_t)(
      TRITONSERVER_Error* error);

  // TRITONSERVER_ErrorDelete
  typedef void (*TritonServerErrorDeleteFn_t)(TRITONSERVER_Error* error);

  // TRITONSERVER_ErrorCodeString
  typedef const char* (*TritonServerErrorCodeToStringFn_t)(
      TRITONSERVER_Error* error);

  // TRITONSERVER_ServerModelConfig
  typedef TRITONSERVER_Error* (*TritonServerModelConfigFn_t)(
      TRITONSERVER_Server* server, const char* model_name,
      const int64_t model_version, const uint32_t config_version,
      TRITONSERVER_Message** model_config);

  // TRITONSERVER_InferenceRequestSetCorrelationId
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestSetCorrelationIdFn_t)(
      TRITONSERVER_InferenceRequest* inference_request,
      uint64_t correlation_id);

  // TRITONSERVER_InferenceRequestSetCorrelationId
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestSetStringCorrelationIdFn_t)(
      TRITONSERVER_InferenceRequest* inference_request,
      const char* correlation_id);

  // TRITONSERVER_InferenceRequestSetFlags
  typedef TRITONSERVER_Error* (*TritonServerInferenceRequestSetFlagsFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, uint32_t flags);

  // TRITONSERVER_InferenceRequestSetPriorityUInt64
  typedef TRITONSERVER_Error* (*TritonServerInferenceRequestSetPriorityFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, uint64_t priority);

  // TRITONSERVER_InferenceRequestSetTimeoutMicroseconds
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestSetTimeoutMicrosecondsFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, uint64_t timeout_us);

  // TRITONSERVER_StringToDataType
  typedef TRITONSERVER_DataType (*TritonServerStringToDatatypeFn_t)(
      const char* dtype);

  // TRITONSERVER_InferenceResponseOutput
  typedef TRITONSERVER_Error* (*TritonServerInferenceResponseOutputFn_t)(
      TRITONSERVER_InferenceResponse* inference_response, const uint32_t index,
      const char** name, TRITONSERVER_DataType* datatype, const int64_t** shape,
      uint64_t* dim_count, const void** base, size_t* byte_size,
      TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id,
      void** userp);

  // TRITONSERVER_InferenceRequestId
  typedef TRITONSERVER_Error* (*TritonServerRequestIdFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char** id);

  // TRITONSERVER_InferenceRequestDelete
  typedef TRITONSERVER_Error* (*TritonServerRequestDeleteFn_t)(
      TRITONSERVER_InferenceRequest* inference_request);

  // TRITONSERVER_ServerModelStatistics
  typedef TRITONSERVER_Error* (*TritonServerModelStatisticsFn_t)(
      TRITONSERVER_Server* server, const char* model_name,
      const int64_t model_version, TRITONSERVER_Message** model_stats);

  // TRITONSERVER_ServerUnloadModel
  typedef TRITONSERVER_Error* (*TritonSeverUnloadModelFn_t)(
      TRITONSERVER_Server* server, const char* model_name);

  // TRITONSERVER_ServerOptionsSetLogInfo
  typedef TRITONSERVER_Error* (*TritonSeverSetLogInfoFn_t)(
      TRITONSERVER_ServerOptions* options, bool log);

  // TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize
  typedef TRITONSERVER_Error* (*TritonServerSetCudaMemoryPoolByteSizeFn_t)(
      TRITONSERVER_ServerOptions* options, int gpu_device, uint64_t size);

 private:
  TritonLoader()
      : InferenceServerClient(
            false /* verbose flag that is set later during ::Create*/)
  {
    verbose_level_ = 0;
    enforce_memory_type_ = false;
    requested_memory_type_ = TRITONSERVER_MEMORY_CPU;
    model_is_loaded_ = false;
    server_is_ready_ = false;
    shm_manager_ = std::make_unique<SharedMemoryManager>();
  }

  Error PopulateInternals(
      const std::string& triton_server_path,
      const std::string& model_repository_path, bool verbose);

  /// Load all tritonserver.h functions onto triton_loader
  /// internal handles
  Error LoadServerLibrary();

  void ClearHandles();

  /// Check if file exists in the current directory
  /// \param filepath Path of library to check
  /// \return perfanalyzer::clientbackend::Error
  Error FileExists(std::string& filepath);

  Error InitializeRequest(
      const tc::InferOptions& options,
      const std::vector<const tc::InferRequestedOutput*>& outputs,
      TRITONSERVER_InferenceRequest** irequest);

  Error AddInputs(
      const std::vector<tc::InferInput*>& inputs,
      TRITONSERVER_InferenceRequest* irequest);

  Error AddOutputs(
      const std::vector<const tc::InferRequestedOutput*>& outputs,
      TRITONSERVER_InferenceRequest* irequest);

  Error GetOutputs(
      TRITONSERVER_InferenceResponse* response,
      std::unordered_map<std::string, ResponseOutput>& outputs);

  void* dlhandle_;
  TritonServerApiVersionFn_t api_version_fn_;
  TritonServerOptionsNewFn_t options_new_fn_;
  TritonServerOptionSetModelRepoPathFn_t options_set_model_repo_path_fn_;
  TritonServerSetLogVerboseFn_t set_log_verbose_fn_;

  TritonServerSetBackendDirFn_t set_backend_directory_fn_;
  TritonServerSetRepoAgentDirFn_t set_repo_agent_directory_fn_;
  TritonServerSetStrictModelConfigFn_t set_strict_model_config_fn_;
  TritonServerSetMinSupportedComputeCapabilityFn_t
      set_min_supported_compute_capability_fn_;

  TritonServerNewFn_t server_new_fn_;
  TritonServerOptionsDeleteFn_t server_options_delete_fn_;
  TritonServerDeleteFn_t server_delete_fn_;
  TritonServerIsLiveFn_t server_is_live_fn_;

  TritonServerIsReadyFn_t server_is_ready_fn_;
  TritonServerMetadataFn_t server_metadata_fn_;
  TritonServerMessageSerializeToJsonFn_t message_serialize_to_json_fn_;
  TritonServerMessageDeleteFn_t message_delete_fn_;

  TritonServerModelIsReadyFn_t model_is_ready_fn_;
  TritonServerModelMetadataFn_t model_metadata_fn_;
  TritonServerResponseAllocatorNewFn_t response_allocator_new_fn_;
  TritonServerInferenceRequestNewFn_t inference_request_new_fn_;

  TritonServerInferenceRequestSetIdFn_t inference_request_set_id_fn_;
  TritonServerInferenceRequestSetReleaseCallbackFn_t
      inference_request_set_release_callback_fn_;
  TritonServerInferenceRequestAddInputFn_t inference_request_add_input_fn_;
  TritonServerInferenceRequestAddRequestedOutputFn_t
      inference_request_add_requested_output_fn_;

  TritonServerInferenceRequestAppendInputDataFn_t
      inference_request_append_input_data_fn_;
  TritonServerInferenceRequestSetResponseCallbackFn_t
      inference_request_set_response_callback_fn_;
  TritonServerInferAsyncFn_t infer_async_fn_;
  TritonServerInferenceResponseErrorFn_t inference_response_error_fn_;

  TritonServerInferenceResponseDeleteFn_t inference_response_delete_fn_;
  TritonServerResponseAllocatorDeleteFn_t response_allocator_delete_fn_;
  TritonServerErrorNewFn_t error_new_fn_;

  TritonServerMemoryTypeStringFn_t memory_type_string_fn_;
  TritonServerInferenceResponseOutputCountFn_t
      inference_response_output_count_fn_;
  TritonServerDataTypeStringFn_t data_type_string_fn_;
  TritonServerErrorMessageFn_t error_message_fn_;

  TritonServerErrorDeleteFn_t error_delete_fn_;
  TritonServerErrorCodeToStringFn_t error_code_to_string_fn_;
  TritonServerModelConfigFn_t model_config_fn_;
  TritonServerInferenceRequestSetCorrelationIdFn_t set_correlation_id_fn_;
  TritonServerInferenceRequestSetStringCorrelationIdFn_t
      set_string_correlation_id_fn_;

  TritonServerInferenceRequestSetFlagsFn_t set_flags_fn_;
  TritonServerInferenceRequestSetPriorityFn_t set_priority_fn_;
  TritonServerInferenceRequestSetTimeoutMicrosecondsFn_t set_timeout_ms_fn_;
  TritonServerStringToDatatypeFn_t string_to_datatype_fn_;

  TritonServerInferenceResponseOutputFn_t inference_response_output_fn_;
  TritonServerRequestIdFn_t request_id_fn_;
  TritonServerRequestDeleteFn_t request_delete_fn_;
  TritonServerModelStatisticsFn_t model_statistics_fn_;

  TritonSeverUnloadModelFn_t unload_model_fn_;
  TritonSeverSetLogInfoFn_t set_log_info_fn_;
  TritonServerSetCudaMemoryPoolByteSizeFn_t set_cuda_memory_pool_byte_size_;

  std::shared_ptr<TRITONSERVER_Server> server_{nullptr};
  std::string triton_server_path_{};
  const std::string server_library_path_{"/lib/libtritonserver.so"};
  int verbose_level_{0};
  TRITONSERVER_MemoryType requested_memory_type_{TRITONSERVER_MEMORY_CPU};
  bool enforce_memory_type_{false};
  std::string model_repository_path_{""};
  std::string model_name_{""};
  int64_t model_version_{-1};
  bool model_is_loaded_{false};
  bool server_is_ready_{false};
  std::unique_ptr<SharedMemoryManager> shm_manager_{nullptr};
  TRITONSERVER_ResponseAllocator* allocator_{};
  std::mutex update_infer_stat_mutex_{};
};

}}}}  // namespace triton::perfanalyzer::clientbackend::tritoncapi
