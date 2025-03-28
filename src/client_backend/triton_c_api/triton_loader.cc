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

#define TRITON_INFERENCE_SERVER_CLIENT_CLASS \
  triton::perfanalyzer::clientbackend::tritoncapi::TritonLoader

#include "triton_loader.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <sys/stat.h>

#include <future>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#include "c_api_infer_results.h"
#include "scoped_defer.h"

#ifdef TRITON_ENABLE_GPU
#include "../../cuda_runtime_library_manager.h"
#endif  // TRITON_ENABLE_GPU


namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritoncapi {
namespace {

bool helper_verbose = false;
/// Helper function for allocating memory
TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // Initially attempt to make the actual memory type and id that we
  // allocate be the same as preferred memory type
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // This variable indicates whether the buffer should be freed or not.
  bool* should_free = new bool;
  *buffer_userp = should_free;
  *should_free = false;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    if (helper_verbose) {
      std::cout << "allocated " << byte_size << " bytes for result tensor "
                << tensor_name << std::endl;
    }
  } else {
    AllocPayload* alloc_payload = reinterpret_cast<AllocPayload*>(userp);
    auto output_map_it = alloc_payload->output_map_.find(tensor_name);
    if (output_map_it == alloc_payload->output_map_.end()) {
      void* allocated_ptr = nullptr;
      *actual_memory_type = TRITONSERVER_MEMORY_CPU;
      *actual_memory_type_id = 0;
      allocated_ptr = malloc(byte_size);
      *should_free = true;

      if (allocated_ptr != nullptr) {
        *buffer = allocated_ptr;
      }
    } else {
      // It is in shared memory
      AllocPayload::OutputInfo* output_info = output_map_it->second;
      if (byte_size > output_info->byte_size_) {
        return TritonLoader::GetSingleton()->ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "shared memory size specified with the request for output '" +
                std::string(tensor_name) + "' (" +
                std::to_string(output_info->byte_size_) +
                " bytes) should be at least " + std::to_string(byte_size) +
                " bytes to hold the results")
                .c_str());
      }
      *actual_memory_type = output_info->memory_type_;
      *actual_memory_type_id = output_info->device_id_;
      *buffer = output_info->base_;
    }
  }

  return nullptr;  // Success
}

/// Helper function for releasing memory
TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  bool* should_free = reinterpret_cast<bool*>(buffer_userp);
  switch (memory_type) {
    case TRITONSERVER_MEMORY_CPU:
      if (*should_free) {
        free(buffer);
      }
      break;
  }

  free(should_free);
  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  TritonLoader::GetSingleton()->DeleteInferRequest(request);
}


void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    // Send 'response' to the future.
    std::promise<TRITONSERVER_InferenceResponse*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
    delete p;
  }
}

Error
GetModelVersionFromString(const std::string& version_string, int64_t* version)
{
  if (version_string.empty()) {
    *version = 1;
    return Error::Success;
  }

  try {
    *version = std::stol(version_string);
  }
  catch (std::exception& e) {
    return Error(
        std::string(
            "Failed to get model version from specified version string '" +
            version_string + "' (details: " + e.what() +
            "), version should be an integral value > 0")
            .c_str());
  }

  if (*version < 0) {
    return Error(std::string(
                     "invalid model version specified '" + version_string +
                     "' , version should be an integral value > 0")
                     .c_str());
  }

  return Error::Success;
}

Error
FolderExists(const std::string& path)
{
  struct stat buffer;
  if (!stat(path.c_str(), &buffer)) {
    return Error::Success;
  } else {
    return Error("Unable to find filepath: " + path);
  }
}
}  // namespace

Error
TritonLoader::Create(
    const std::string& triton_server_path,
    const std::string& model_repository_path, bool verbose)
{
  if (!GetSingleton()->ServerIsReady()) {
    GetSingleton()->ClearHandles();
    RETURN_IF_ERROR(GetSingleton()->PopulateInternals(
        triton_server_path, model_repository_path, verbose));
    RETURN_IF_ERROR(GetSingleton()->LoadServerLibrary());
    RETURN_IF_ERROR(GetSingleton()->StartTriton());
  }

  return Error::Success;
}

Error
TritonLoader::Delete()
{
  if (server_ != nullptr) {
    server_is_ready_ = false;
    model_is_loaded_ = false;
    server_.reset();
  }
  return Error::Success;
}

Error
TritonLoader::PopulateInternals(
    const std::string& triton_server_path,
    const std::string& model_repository_path, bool verbose)
{
  RETURN_IF_ERROR(FolderExists(triton_server_path));
  RETURN_IF_ERROR(FolderExists(model_repository_path));

  triton_server_path_ = triton_server_path;
  model_repository_path_ = model_repository_path;
  verbose_ = verbose;
  verbose_level_ = verbose_ ? 1 : 0;
  return Error::Success;
}

Error
TritonLoader::StartTriton()
{
  // Check API version.
  uint32_t api_version_major, api_version_minor;
  REPORT_TRITONSERVER_ERROR(
      api_version_fn_(&api_version_major, &api_version_minor),
      "unable to get api version");

  auto createErrorMessage = [](const std::string& message, int expected,
                               int actual) {
    return message + "Expected version: " + std::to_string(expected) + "\n" +
           "Actual version: " + std::to_string(actual) + "\n";
  };

  if (TRITONSERVER_API_VERSION_MAJOR != api_version_major) {
    std::string errorMessage = createErrorMessage(
        "Error: Triton server API major version mismatch.\n",
        TRITONSERVER_API_VERSION_MAJOR, api_version_major);
    return Error(errorMessage);
  }

  if (TRITONSERVER_API_VERSION_MINOR != api_version_minor) {
    std::string warningMessage = createErrorMessage(
        "Warning: Triton server API minor version mismatch.\n",
        TRITONSERVER_API_VERSION_MINOR, api_version_minor);
    warningMessage +=
        "Attempting to proceed, but undefined behavior may occur.\n";
    std::cerr << warningMessage;
  }

  // Create the server...
  TRITONSERVER_ServerOptions* server_options = nullptr;
  RETURN_IF_TRITONSERVER_ERROR(
      options_new_fn_(&server_options), "creating server options");
  RETURN_IF_TRITONSERVER_ERROR(
      options_set_model_repo_path_fn_(
          server_options, model_repository_path_.c_str()),
      "setting model repository path");
  RETURN_IF_TRITONSERVER_ERROR(
      set_cuda_memory_pool_byte_size_(server_options, 0, 1073741824),
      "setting cuda memory pool byte size failed.");
  REPORT_TRITONSERVER_ERROR(
      set_log_verbose_fn_(server_options, verbose_level_),
      "setting verbose logging level");
  REPORT_TRITONSERVER_ERROR(
      set_log_info_fn_(server_options, verbose_),
      "setting if log verbose level is true");
  RETURN_IF_TRITONSERVER_ERROR(
      set_backend_directory_fn_(
          server_options, (triton_server_path_ + "/backends").c_str()),
      "setting backend directory");
  RETURN_IF_TRITONSERVER_ERROR(
      set_repo_agent_directory_fn_(
          server_options, (triton_server_path_ + "/repoagents").c_str()),
      "setting repository agent directory");
  RETURN_IF_TRITONSERVER_ERROR(
      set_strict_model_config_fn_(server_options, true),
      "setting strict model configuration");
  double min_compute_capability = 0;
  // FIXME: Do not have GPU support right now
  RETURN_IF_TRITONSERVER_ERROR(
      set_min_supported_compute_capability_fn_(
          server_options, min_compute_capability),
      "setting minimum supported CUDA compute capability");
  TRITONSERVER_Server* server_ptr = nullptr;
  RETURN_IF_TRITONSERVER_ERROR(
      server_new_fn_(&server_ptr, server_options), "creating server");
  RETURN_IF_TRITONSERVER_ERROR(
      server_options_delete_fn_(server_options), "deleting server options");
  std::shared_ptr<TRITONSERVER_Server> shared_server(
      server_ptr, server_delete_fn_);
  server_ = shared_server;

  // Wait until the server is both live and ready.
  size_t health_iters = 0;
  while (true) {
    bool live, ready;
    RETURN_IF_TRITONSERVER_ERROR(
        server_is_live_fn_(server_.get(), &live),
        "unable to get server liveness");
    RETURN_IF_TRITONSERVER_ERROR(
        server_is_ready_fn_(server_.get(), &ready),
        "unable to get server readiness");
    if (live && ready) {
      server_is_ready_ = true;
      break;
    }

    if (++health_iters >= 10) {
      return Error("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  // Print status of the server.
  if (verbose_) {
    TRITONSERVER_Message* server_metadata_message;
    RETURN_IF_TRITONSERVER_ERROR(
        server_metadata_fn_(server_.get(), &server_metadata_message),
        "unable to get server metadata message");
    const char* buffer;
    size_t byte_size;
    RETURN_IF_TRITONSERVER_ERROR(
        message_serialize_to_json_fn_(
            server_metadata_message, &buffer, &byte_size),
        "unable to serialize server metadata message");

    RETURN_IF_TRITONSERVER_ERROR(
        message_delete_fn_(server_metadata_message),
        "deleting status metadata");
  }

  // Create the allocator that will be used to allocate buffers for
  // the result tensors.
  RETURN_IF_TRITONSERVER_ERROR(
      GetSingleton()->response_allocator_new_fn_(
          &allocator_,
          reinterpret_cast<
              TRITONSERVER_Error* (*)(TRITONSERVER_ResponseAllocator* allocator,
                                      const char* tensor_name, size_t byte_size,
                                      TRITONSERVER_MemoryType memory_type,
                                      int64_t memory_type_id, void* userp,
                                      void** buffer, void** buffer_userp,
                                      TRITONSERVER_MemoryType*
                                          actual_memory_type,
                                      int64_t* actual_memory_type_id)>(
              ResponseAlloc),
          reinterpret_cast<
              TRITONSERVER_Error* (*)(TRITONSERVER_ResponseAllocator* allocator,
                                      void* buffer, void* buffer_userp,
                                      size_t byte_size,
                                      TRITONSERVER_MemoryType memory_type,
                                      int64_t memory_type_id)>(ResponseRelease),
          nullptr /* start_fn */),
      "creating response allocator");

  return Error::Success;
}

Error
TritonLoader::ServerMetaData(rapidjson::Document* server_metadata)
{
  if (!ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  TRITONSERVER_Message* server_metadata_message;
  RETURN_IF_TRITONSERVER_ERROR(
      server_metadata_fn_(server_.get(), &server_metadata_message),
      "unable to get server metadata message");
  const char* buffer;
  size_t byte_size;
  RETURN_IF_TRITONSERVER_ERROR(
      message_serialize_to_json_fn_(
          server_metadata_message, &buffer, &byte_size),
      "unable to serialize server metadata message");
  server_metadata->Parse(buffer, byte_size);
  if (server_metadata->HasParseError()) {
    return Error(
        "error: failed to parse server metadata from JSON: " +
        std::string(GetParseError_En(server_metadata->GetParseError())) +
        " at " + std::to_string(server_metadata->GetErrorOffset()));
  }
  RETURN_IF_TRITONSERVER_ERROR(
      message_delete_fn_(server_metadata_message), "deleting status metadata");
  return Error::Success;
}

Error
TritonLoader::LoadModel(
    const std::string& model_name, const std::string& model_version)
{
  if (!ServerIsReady()) {
    return Error("server is not ready, abort!");
  }
  model_name_ = model_name;

  RETURN_IF_ERROR(GetModelVersionFromString(model_version, &model_version_));
  // Wait for the model to become available.
  bool is_ready = false;
  size_t health_iters = 0;

  // some error handling
  if (model_repository_path_.empty()) {
    return Error("Need to specify model repository");
  }
  while (!is_ready) {
    RETURN_IF_TRITONSERVER_ERROR(
        model_is_ready_fn_(
            server_.get(), model_name_.c_str(), model_version_, &is_ready),
        "unable to get model readiness");
    if (!is_ready) {
      if (++health_iters >= 10) {
        return Error("model failed to be ready in 10 iterations");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }
  }
  // flag to confirm model is correct and loaded
  model_is_loaded_ = true;
  return Error::Success;
}

Error
TritonLoader::ModelMetadata(rapidjson::Document* model_metadata)
{
  if (!ModelIsLoaded() || !ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  TRITONSERVER_Message* model_metadata_message;

  // get model metadata
  RETURN_IF_TRITONSERVER_ERROR(
      model_metadata_fn_(
          server_.get(), model_name_.c_str(), model_version_,
          &model_metadata_message),
      "unable to get model metadata message");
  const char* buffer;
  size_t byte_size;
  RETURN_IF_TRITONSERVER_ERROR(
      message_serialize_to_json_fn_(
          model_metadata_message, &buffer, &byte_size),
      "unable to serialize model status protobuf");

  model_metadata->Parse(buffer, byte_size);
  if (model_metadata->HasParseError()) {
    return Error(
        "error: failed to parse model metadata from JSON: " +
        std::string(GetParseError_En(model_metadata->GetParseError())) +
        " at " + std::to_string(model_metadata->GetErrorOffset()));
  }

  RETURN_IF_TRITONSERVER_ERROR(
      message_delete_fn_(model_metadata_message), "deleting status protobuf");

  if (strcmp((*model_metadata)["name"].GetString(), model_name_.c_str())) {
    return Error("unable to find metadata for model");
  }

  bool found_version = false;
  if (model_metadata->HasMember("versions")) {
    for (const auto& version : (*model_metadata)["versions"].GetArray()) {
      if (strcmp(version.GetString(), std::to_string(model_version_).c_str()) ==
          0) {
        found_version = true;
        break;
      }
    }
  }
  if (!found_version) {
    std::string msg = "unable to find version " +
                      std::to_string(model_version_) + " status for model";
    return Error(msg);
  }
  return Error::Success;
}

Error
TritonLoader::ModelConfig(
    rapidjson::Document* model_config, const std::string& model_name,
    const std::string& model_version)
{
  if (!ModelIsLoaded() || !ServerIsReady()) {
    return Error("Model is not loaded and/or server is not ready");
  }
  TRITONSERVER_Message* model_config_message;
  uint32_t config_version = 1;
  RETURN_IF_TRITONSERVER_ERROR(
      model_config_fn_(
          (server_).get(), model_name.c_str(), model_version_, config_version,
          &model_config_message),
      "unable to get model config message");
  const char* buffer;
  size_t byte_size;
  RETURN_IF_TRITONSERVER_ERROR(
      message_serialize_to_json_fn_(model_config_message, &buffer, &byte_size),
      "unable to serialize model config status protobuf");

  model_config->Parse(buffer, byte_size);
  if (model_config->HasParseError()) {
    return Error(
        "error: failed to parse model config from JSON: " +
        std::string(GetParseError_En(model_config->GetParseError())) + " at " +
        std::to_string(model_config->GetErrorOffset()));
  }

  RETURN_IF_TRITONSERVER_ERROR(
      message_delete_fn_(model_config_message),
      "deleting server config status protobuf");

  return Error::Success;
}

Error
TritonLoader::LoadServerLibrary()
{
  std::string full_path = triton_server_path_ + server_library_path_;
  RETURN_IF_ERROR(FolderExists(full_path));
  RETURN_IF_ERROR(OpenLibraryHandle(full_path, &dlhandle_));

  TritonServerApiVersionFn_t apifn;
  TritonServerOptionsNewFn_t onfn;
  TritonServerOptionSetModelRepoPathFn_t rpfn;
  TritonServerSetLogVerboseFn_t slvfn;

  TritonServerSetBackendDirFn_t sbdfn;
  TritonServerSetRepoAgentDirFn_t srdfn;
  TritonServerSetStrictModelConfigFn_t ssmcfn;
  TritonServerSetMinSupportedComputeCapabilityFn_t smsccfn;

  TritonServerNewFn_t snfn;
  TritonServerOptionsDeleteFn_t odfn;
  TritonServerDeleteFn_t sdfn;
  TritonServerIsLiveFn_t ilfn;

  TritonServerIsReadyFn_t irfn;
  TritonServerMetadataFn_t smfn;
  TritonServerMessageSerializeToJsonFn_t stjfn;
  TritonServerMessageDeleteFn_t mdfn;

  TritonServerModelIsReadyFn_t mirfn;
  TritonServerModelMetadataFn_t mmfn;
  TritonServerResponseAllocatorNewFn_t ranfn;
  TritonServerInferenceRequestNewFn_t irnfn;

  TritonServerInferenceRequestSetIdFn_t irsifn;
  TritonServerInferenceRequestSetReleaseCallbackFn_t irsrcfn;
  TritonServerInferenceRequestAddInputFn_t iraifn;
  TritonServerInferenceRequestAddRequestedOutputFn_t irarofn;

  TritonServerInferenceRequestAppendInputDataFn_t iraidfn;
  TritonServerInferenceRequestSetResponseCallbackFn_t irsrescfn;
  TritonServerInferAsyncFn_t iafn;
  TritonServerInferenceResponseErrorFn_t irefn;

  TritonServerInferenceResponseDeleteFn_t irdfn;
  TritonServerResponseAllocatorDeleteFn_t radfn;
  TritonServerErrorNewFn_t enfn;

  TritonServerMemoryTypeStringFn_t mtsfn;
  TritonServerInferenceResponseOutputCountFn_t irocfn;
  TritonServerDataTypeStringFn_t dtsfn;

  TritonServerErrorDeleteFn_t edfn;
  TritonServerErrorCodeToStringFn_t ectsfn;
  TritonServerErrorMessageFn_t emfn;
  TritonServerModelConfigFn_t mcfn;
  TritonServerInferenceRequestSetCorrelationIdFn_t scidfn;
  TritonServerInferenceRequestSetStringCorrelationIdFn_t sscidfn;

  TritonServerInferenceRequestSetFlagsFn_t sffn;
  TritonServerInferenceRequestSetPriorityFn_t spfn;
  TritonServerInferenceRequestSetTimeoutMicrosecondsFn_t stmsfn;
  TritonServerStringToDatatypeFn_t stdtfn;

  TritonServerInferenceResponseOutputFn_t irofn;
  TritonServerRequestIdFn_t ridfn;
  TritonServerRequestDeleteFn_t rdfn;
  TritonServerModelStatisticsFn_t msfn;

  TritonSeverUnloadModelFn_t umfn;
  TritonSeverSetLogInfoFn_t slifn;
  TritonServerSetCudaMemoryPoolByteSizeFn_t scmpbsfn;

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ApiVersion", false /* optional */,
      reinterpret_cast<void**>(&apifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsNew", false /* optional */,
      reinterpret_cast<void**>(&onfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetModelRepositoryPath",
      false /* optional */, reinterpret_cast<void**>(&rpfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetLogVerbose",
      false /* optional */, reinterpret_cast<void**>(&slvfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetBackendDirectory",
      false /* optional */, reinterpret_cast<void**>(&sbdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetRepoAgentDirectory",
      false /* optional */, reinterpret_cast<void**>(&srdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetStrictModelConfig",
      false /* optional */, reinterpret_cast<void**>(&ssmcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability",
      false /* optional */, reinterpret_cast<void**>(&smsccfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize",
      false /* optional */, reinterpret_cast<void**>(&scmpbsfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerNew", false /* optional */,
      reinterpret_cast<void**>(&snfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsDelete", false /* optional */,
      reinterpret_cast<void**>(&odfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerDelete", false /* optional */,
      reinterpret_cast<void**>(&sdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerIsLive", false /* optional */,
      reinterpret_cast<void**>(&ilfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerIsReady", false /* optional */,
      reinterpret_cast<void**>(&irfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerMetadata", false /* optional */,
      reinterpret_cast<void**>(&smfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MessageSerializeToJson", false /* optional */,
      reinterpret_cast<void**>(&stjfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MessageDelete", false /* optional */,
      reinterpret_cast<void**>(&mdfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelIsReady", false /* optional */,
      reinterpret_cast<void**>(&mirfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelMetadata", false /* optional */,
      reinterpret_cast<void**>(&mmfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ResponseAllocatorNew", false /* optional */,
      reinterpret_cast<void**>(&ranfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestNew", false /* optional */,
      reinterpret_cast<void**>(&irnfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetId", false /* optional */,
      reinterpret_cast<void**>(&irsifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetReleaseCallback",
      false /* optional */, reinterpret_cast<void**>(&irsrcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAddInput", false /* optional */,
      reinterpret_cast<void**>(&iraifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAddRequestedOutput",
      false /* optional */, reinterpret_cast<void**>(&irarofn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestAppendInputData",
      false /* optional */, reinterpret_cast<void**>(&iraidfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetResponseCallback",
      false /* optional */, reinterpret_cast<void**>(&irsrescfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerInferAsync", false /* optional */,
      reinterpret_cast<void**>(&iafn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseError", false /* optional */,
      reinterpret_cast<void**>(&irefn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseDelete", false /* optional */,
      reinterpret_cast<void**>(&irdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ResponseAllocatorDelete", false /* optional */,
      reinterpret_cast<void**>(&radfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorNew", false /* optional */,
      reinterpret_cast<void**>(&enfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_MemoryTypeString", false /* optional */,
      reinterpret_cast<void**>(&mtsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseOutputCount",
      false /* optional */, reinterpret_cast<void**>(&irocfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_DataTypeString", false /* optional */,
      reinterpret_cast<void**>(&dtsfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorDelete", false /* optional */,
      reinterpret_cast<void**>(&edfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorCodeString", false /* optional */,
      reinterpret_cast<void**>(&ectsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ErrorMessage", false /* optional */,
      reinterpret_cast<void**>(&emfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelConfig", false /* optional */,
      reinterpret_cast<void**>(&mcfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetCorrelationId",
      false /* optional */, reinterpret_cast<void**>(&scidfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetCorrelationIdString",
      false /* optional */, reinterpret_cast<void**>(&sscidfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetFlags", false /* optional */,
      reinterpret_cast<void**>(&sffn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetPriorityUInt64",
      false /* optional */, reinterpret_cast<void**>(&spfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestSetTimeoutMicroseconds",
      false /* optional */, reinterpret_cast<void**>(&stmsfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_StringToDataType", false /* optional */,
      reinterpret_cast<void**>(&stdtfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceResponseOutput", false /* optional */,
      reinterpret_cast<void**>(&irofn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestId", false /* optional */,
      reinterpret_cast<void**>(&ridfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_InferenceRequestDelete", false /* optional */,
      reinterpret_cast<void**>(&rdfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerModelStatistics", false /* optional */,
      reinterpret_cast<void**>(&msfn)));

  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerUnloadModel", false /* optional */,
      reinterpret_cast<void**>(&umfn)));
  RETURN_IF_ERROR(GetEntrypoint(
      dlhandle_, "TRITONSERVER_ServerOptionsSetLogInfo", false /* optional */,
      reinterpret_cast<void**>(&slifn)));


  api_version_fn_ = apifn;
  options_new_fn_ = onfn;
  options_set_model_repo_path_fn_ = rpfn;
  set_log_verbose_fn_ = slvfn;

  set_backend_directory_fn_ = sbdfn;
  set_repo_agent_directory_fn_ = srdfn;
  set_strict_model_config_fn_ = ssmcfn;
  set_min_supported_compute_capability_fn_ = smsccfn;

  server_new_fn_ = snfn;
  server_options_delete_fn_ = odfn;
  server_delete_fn_ = sdfn;
  server_is_live_fn_ = ilfn;

  server_is_ready_fn_ = irfn;
  server_metadata_fn_ = smfn;
  message_serialize_to_json_fn_ = stjfn;
  message_delete_fn_ = mdfn;

  model_is_ready_fn_ = mirfn;
  model_metadata_fn_ = mmfn;
  response_allocator_new_fn_ = ranfn;
  inference_request_new_fn_ = irnfn;

  inference_request_set_id_fn_ = irsifn;
  inference_request_set_release_callback_fn_ = irsrcfn;
  inference_request_add_input_fn_ = iraifn;
  inference_request_add_requested_output_fn_ = irarofn;

  inference_request_append_input_data_fn_ = iraidfn;
  inference_request_set_response_callback_fn_ = irsrescfn;
  infer_async_fn_ = iafn;
  inference_response_error_fn_ = irefn;

  inference_response_delete_fn_ = irdfn;
  response_allocator_delete_fn_ = radfn;
  error_new_fn_ = enfn;

  memory_type_string_fn_ = mtsfn;
  inference_response_output_count_fn_ = irocfn;
  data_type_string_fn_ = dtsfn;

  error_delete_fn_ = edfn;
  error_code_to_string_fn_ = ectsfn;
  error_message_fn_ = emfn;
  model_config_fn_ = mcfn;
  set_correlation_id_fn_ = scidfn;
  set_string_correlation_id_fn_ = sscidfn;

  set_flags_fn_ = sffn;
  set_priority_fn_ = spfn;
  set_timeout_ms_fn_ = stmsfn;
  string_to_datatype_fn_ = stdtfn;

  inference_response_output_fn_ = irofn;
  request_id_fn_ = ridfn;
  request_delete_fn_ = rdfn;
  model_statistics_fn_ = msfn;

  unload_model_fn_ = umfn;
  set_log_info_fn_ = slifn;
  set_cuda_memory_pool_byte_size_ = scmpbsfn;

  return Error::Success;
}

void
TritonLoader::ClearHandles()
{
  dlhandle_ = nullptr;

  api_version_fn_ = nullptr;
  options_new_fn_ = nullptr;
  options_set_model_repo_path_fn_ = nullptr;
  set_log_verbose_fn_ = nullptr;

  set_backend_directory_fn_ = nullptr;
  set_repo_agent_directory_fn_ = nullptr;
  set_strict_model_config_fn_ = nullptr;
  set_min_supported_compute_capability_fn_ = nullptr;

  server_new_fn_ = nullptr;
  server_options_delete_fn_ = nullptr;
  server_delete_fn_ = nullptr;
  server_is_live_fn_ = nullptr;

  server_is_ready_fn_ = nullptr;
  server_metadata_fn_ = nullptr;
  message_serialize_to_json_fn_ = nullptr;
  message_delete_fn_ = nullptr;

  model_is_ready_fn_ = nullptr;
  model_metadata_fn_ = nullptr;
  response_allocator_new_fn_ = nullptr;
  inference_request_new_fn_ = nullptr;

  inference_request_set_id_fn_ = nullptr;
  inference_request_set_release_callback_fn_ = nullptr;
  inference_request_add_input_fn_ = nullptr;
  inference_request_add_requested_output_fn_ = nullptr;

  inference_request_append_input_data_fn_ = nullptr;
  inference_request_set_response_callback_fn_ = nullptr;
  infer_async_fn_ = nullptr;
  inference_response_error_fn_ = nullptr;

  inference_response_delete_fn_ = nullptr;
  response_allocator_delete_fn_ = nullptr;
  error_new_fn_ = nullptr;

  memory_type_string_fn_ = nullptr;
  inference_response_output_count_fn_ = nullptr;
  data_type_string_fn_ = nullptr;
  error_message_fn_ = nullptr;

  error_delete_fn_ = nullptr;
  error_code_to_string_fn_ = nullptr;
  model_config_fn_ = nullptr;
  set_correlation_id_fn_ = nullptr;
  set_string_correlation_id_fn_ = nullptr;

  set_flags_fn_ = nullptr;
  set_priority_fn_ = nullptr;
  set_timeout_ms_fn_ = nullptr;
  string_to_datatype_fn_ = nullptr;

  inference_response_output_fn_ = nullptr;
  request_id_fn_ = nullptr;
  request_delete_fn_ = nullptr;
  model_statistics_fn_ = nullptr;
  unload_model_fn_ = nullptr;
  set_log_info_fn_ = nullptr;
}

Error
TritonLoader::FileExists(std::string& filepath)
{
  std::ifstream ifile;
  ifile.open(filepath);
  if (!ifile) {
    return Error("unable to find local Triton library: " + filepath);
  } else {
    return Error::Success;
  }
}

Error
TritonLoader::Infer(
    const tc::InferOptions& options, const std::vector<tc::InferInput*>& inputs,
    const std::vector<const tc::InferRequestedOutput*>& outputs,
    InferResult** result)
{
  Error error = Error::Success;
  if (!ServerIsReady() || !ModelIsLoaded()) {
    return Error("Server is not ready and/or requested model is not loaded");
  }

  TRITONSERVER_InferenceRequest* irequest = nullptr;
  TRITONSERVER_InferenceResponse* completed_response = nullptr;
  tc::RequestTimers timer;
  timer.Reset();
  timer.CaptureTimestamp(tc::RequestTimers::Kind::REQUEST_START);

  RETURN_IF_ERROR(InitializeRequest(options, outputs, &irequest));
  ScopedDefer error_handler([&error, &completed_response, this] {
    error = CleanUp(completed_response);
  });
  RETURN_IF_ERROR(AddInputs(inputs, irequest));
  RETURN_IF_ERROR(AddOutputs(outputs, irequest));

  AllocPayload alloc_payload;
  for (auto& output : outputs) {
    if (output->IsSharedMemory()) {
      std::string shm_name;
      size_t shm_byte_size;
      size_t offset;
      // TODO: Error handling
      output->SharedMemoryInfo(&shm_name, &shm_byte_size, &offset);

      void* buf;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      RETURN_IF_ERROR(shm_manager_->GetMemoryInfo(
          shm_name, offset, shm_byte_size, &buf, &memory_type,
          &memory_type_id));

      alloc_payload.output_map_.emplace(
          std::piecewise_construct, std::forward_as_tuple(output->Name()),
          std::forward_as_tuple(new AllocPayload::OutputInfo(
              buf, shm_byte_size, memory_type, memory_type_id)));
    }
  }

  const char* cid = nullptr;
  RETURN_IF_TRITONSERVER_ERROR(
      request_id_fn_(irequest, &cid), "Failed to get request id");
  std::string id = cid;

  // Perform inference...
  timer.CaptureTimestamp(tc::RequestTimers::Kind::SEND_START);
  auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
  std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();
  RETURN_IF_TRITONSERVER_ERROR(
      inference_request_set_response_callback_fn_(
          irequest, allocator_, &alloc_payload /* response_allocator_userp */,
          InferResponseComplete, reinterpret_cast<void*>(p)),
      "setting response callback");
  RETURN_IF_TRITONSERVER_ERROR(
      infer_async_fn_((server_).get(), irequest, nullptr /* trace */),
      "running inference");
  timer.CaptureTimestamp(tc::RequestTimers::Kind::SEND_END);

  // Wait for the inference to complete.
  completed_response = completed.get();

  RETURN_IF_TRITONSERVER_ERROR(
      inference_response_error_fn_(completed_response),
      "inference response error");

  timer.CaptureTimestamp(tc::RequestTimers::Kind::RECV_START);
  timer.CaptureTimestamp(tc::RequestTimers::Kind::RECV_END);
  timer.CaptureTimestamp(tc::RequestTimers::Kind::REQUEST_END);

  tc::Error err = UpdateInferStat(timer);
  if (!err.IsOk()) {
    std::cerr << "Failed to update context stat: " << err << std::endl;
  }

  std::unordered_map<std::string, ResponseOutput> response_outputs{};
  GetOutputs(completed_response, response_outputs);

  // Synchronous mode requests only ever have one response, which is "final"
  bool is_final_response{true};
  bool is_null_response{completed_response == nullptr};

  InferResult::Create(
      result, err, id, std::move(response_outputs), is_final_response,
      is_null_response);

  // CleanUp the response
  error_handler.Complete();

  return error;
}

Error
TritonLoader::GetOutputs(
    TRITONSERVER_InferenceResponse* response,
    std::unordered_map<std::string, ResponseOutput>& outputs)
{
  uint32_t count{};
  RETURN_IF_TRITONSERVER_ERROR(
      inference_response_output_count_fn_(response, &count),
      "inference_response_output_count_fn_ error");

  for (uint32_t index{0}; index < count; index++) {
    const char* name{};
    TRITONSERVER_DataType datatype{};
    const int64_t* shape{};
    uint64_t dim_count{};
    const void* base{};
    size_t byte_size{};
    TRITONSERVER_MemoryType memory_type{};
    int64_t memory_type_id{};
    void* userp{};

    RETURN_IF_TRITONSERVER_ERROR(
        inference_response_output_fn_(
            response, index, &name, &datatype, &shape, &dim_count, &base,
            &byte_size, &memory_type, &memory_type_id, &userp),
        "inference_response_output_fn_ error");

    std::string data_type{datatype};
    std::vector<uint8_t> data_copy;

    if (memory_type == TRITONSERVER_MEMORY_CPU ||
        memory_type == TRITONSERVER_MEMORY_CPU_PINNED) {
      std::copy(
          static_cast<const uint8_t*>(base),
          static_cast<const uint8_t*>(base) + byte_size,
          std::back_inserter(data_copy));
    }
#ifdef TRITON_ENABLE_GPU
    else {
      CUDARuntimeLibraryManager cuda_manager;
      CUDARuntimeLibraryManager::cudaError_t cuda_err = cuda_manager.cudaMemcpy(
          data_copy.data(), base, byte_size,
          CUDARuntimeLibraryManager::cudaMemcpyKind::cudaMemcpyDeviceToHost);
      if (cuda_err != cudaSuccess) {
        return Error(
            "CUDA memory copy failed: " +
            std::string(cuda_manager.cudaGetErrorString(cuda_err)));
      }
    }
#endif  // TRITON_ENABLE_GPU

    outputs.emplace(
        name,
        ResponseOutput{
            std::string(name), datatype, shape, dim_count, std::move(data_copy),
            byte_size, memory_type, memory_type_id, userp});
  }

  return Error::Success;
}

void
InferResponseCompleteAsyncNonMember(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  TritonLoader::GetSingleton()->InferResponseCompleteAsync(
      response, flags,
      reinterpret_cast<TritonLoader::AsyncRequestInfo*>(userp));
}

void
TritonLoader::InferResponseCompleteAsync(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags,
    AsyncRequestInfo* async_request_info)
{
  REPORT_TRITONSERVER_ERROR(
      inference_response_error_fn_(response),
      "unable to get inference response error");

  if (async_request_info->enable_stats) {
    tc::RequestTimers timer{*async_request_info->timer};

    timer.CaptureTimestamp(tc::RequestTimers::Kind::RECV_START);
    timer.CaptureTimestamp(tc::RequestTimers::Kind::RECV_END);
    timer.CaptureTimestamp(tc::RequestTimers::Kind::REQUEST_END);

    {
      std::lock_guard<std::mutex> lock(update_infer_stat_mutex_);
      tc::Error err{UpdateInferStat(timer)};
      if (!err.IsOk()) {
        std::cerr << "Failed to update context stat: " << err << std::endl;
      }
    }
  }

  std::unordered_map<std::string, ResponseOutput> outputs{};
  Error err{GetOutputs(response, outputs)};
  if (!err.IsOk()) {
    std::cerr << "Failed to get outputs: " << err << std::endl;
  }

  bool is_final_response{flags == TRITONSERVER_RESPONSE_COMPLETE_FINAL};
  bool is_null_response{response == nullptr};

  InferResult* infer_result{};
  InferResult::Create(
      &infer_result, tc::Error::Success, async_request_info->request_id,
      std::move(outputs), is_final_response, is_null_response);

  async_request_info->callback(infer_result);

  if (is_final_response) {
    delete async_request_info;
  }

  CleanUp(response);
}

Error
TritonLoader::AsyncInfer(
    OnCompleteFn callback, const tc::InferOptions& options,
    const std::vector<tc::InferInput*>& inputs,
    const std::vector<const tc::InferRequestedOutput*>& outputs,
    bool enable_stats)
{
  Error error = Error::Success;
  if (!ServerIsReady() || !ModelIsLoaded()) {
    return Error("Server is not ready and/or requested model is not loaded");
  }

  TRITONSERVER_InferenceRequest* irequest = nullptr;
  TRITONSERVER_InferenceResponse* completed_response = nullptr;
  std::shared_ptr<tc::RequestTimers> timer{
      std::make_shared<tc::RequestTimers>()};
  timer->Reset();
  timer->CaptureTimestamp(tc::RequestTimers::Kind::REQUEST_START);

  RETURN_IF_ERROR(InitializeRequest(options, outputs, &irequest));
  RETURN_IF_ERROR(AddInputs(inputs, irequest));
  RETURN_IF_ERROR(AddOutputs(outputs, irequest));

  const char* cid = nullptr;
  RETURN_IF_TRITONSERVER_ERROR(
      request_id_fn_(irequest, &cid), "Failed to get request id");

  // Perform inference...
  timer->CaptureTimestamp(tc::RequestTimers::Kind::SEND_START);
  AsyncRequestInfo* async_request_info{new AsyncRequestInfo};
  async_request_info->alloc_payload = std::make_unique<AllocPayload>();
  async_request_info->request_id = cid;
  async_request_info->timer = timer;
  async_request_info->callback = callback;
  async_request_info->enable_stats = enable_stats;
  RETURN_IF_TRITONSERVER_ERROR(
      inference_request_set_response_callback_fn_(
          irequest, allocator_,
          async_request_info->alloc_payload
              .get() /* response_allocator_userp */,
          InferResponseCompleteAsyncNonMember, async_request_info),
      "setting response callback");
  RETURN_IF_TRITONSERVER_ERROR(
      infer_async_fn_((server_).get(), irequest, nullptr /* trace */),
      "running inference");
  timer->CaptureTimestamp(tc::RequestTimers::Kind::SEND_END);

  return error;
}

Error
TritonLoader::CleanUp(TRITONSERVER_InferenceResponse* completed_response)
{
  TRITONSERVER_Error* response_err = nullptr;
  if (completed_response != nullptr) {
    response_err = inference_response_delete_fn_(completed_response);
    RETURN_IF_TRITONSERVER_ERROR(response_err, "deleting inference response");
  }
  return Error::Success;
}

Error
TritonLoader::InitializeRequest(
    const tc::InferOptions& options,
    const std::vector<const tc::InferRequestedOutput*>& outputs,
    TRITONSERVER_InferenceRequest** irequest)
{
  // set up inference request
  RETURN_IF_TRITONSERVER_ERROR(
      inference_request_new_fn_(
          irequest, (server_).get(), model_name_.c_str(), model_version_),
      "creating inference request");
  RETURN_IF_TRITONSERVER_ERROR(
      inference_request_set_id_fn_(*irequest, options.request_id_.c_str()),
      "setting ID for the request");
  if ((options.sequence_id_ != 0) || (options.sequence_id_str_ != "") ||
      (options.priority_ != 0) || (options.server_timeout_ != 0) ||
      outputs.empty()) {
    if (options.sequence_id_ != 0) {
      RETURN_IF_TRITONSERVER_ERROR(
          set_correlation_id_fn_(*irequest, options.sequence_id_),
          "setting sequence ID for the request");
    } else if (options.sequence_id_str_ != "") {
      RETURN_IF_TRITONSERVER_ERROR(
          set_string_correlation_id_fn_(
              *irequest, options.sequence_id_str_.c_str()),
          "setting sequence ID for the request");
    }
    uint32_t flags = 0;
    if (options.sequence_start_) {
      flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
    }
    if (options.sequence_end_) {
      flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
    }
    RETURN_IF_TRITONSERVER_ERROR(
        set_flags_fn_(*irequest, flags),
        "setting inference flags for the request");
  }
  if (options.priority_ != 0) {
    RETURN_IF_TRITONSERVER_ERROR(
        set_priority_fn_(*irequest, options.priority_),
        "setting priority for the request");
  }
  if (options.server_timeout_ != 0) {
    RETURN_IF_TRITONSERVER_ERROR(
        set_timeout_ms_fn_(*irequest, options.server_timeout_),
        "setting timeout for the request");
  }
  RETURN_IF_TRITONSERVER_ERROR(
      inference_request_set_release_callback_fn_(
          *irequest, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");
  return Error::Success;
}

Error
TritonLoader::AddInputs(
    const std::vector<tc::InferInput*>& inputs,
    TRITONSERVER_InferenceRequest* irequest)
{
  for (auto io : inputs) {
    const char* input_name = io->Name().c_str();
    const char* datatype = io->Datatype().c_str();
    const TRITONSERVER_DataType dtype = string_to_datatype_fn_(datatype);
    std::vector<int64_t> shape_vec;
    for (const int64_t dim : io->Shape()) {  // this is a vector, just use it
      shape_vec.push_back(dim);
    }

    RETURN_IF_TRITONSERVER_ERROR(
        inference_request_add_input_fn_(
            irequest, input_name, dtype, &shape_vec[0], shape_vec.size()),
        "setting input for the request");
    size_t byte_size;
    tc::Error err = io->ByteSize(&byte_size);
    if (!err.IsOk()) {
      return Error(err.Message());
    }
    if (byte_size == 0) {
      RETURN_IF_TRITONSERVER_ERROR(
          inference_request_append_input_data_fn_(
              irequest, input_name, nullptr, 0 /* byte_size */,
              TRITONSERVER_MEMORY_CPU /* memory type */,
              0 /* memory_type_id */),
          "appending input data with byte size zero");
    } else {
      if (!io->IsSharedMemory()) {
        io->PrepareForRequest();
        bool end_of_input = false;
        while (!end_of_input) {
          const uint8_t* buf;
          size_t buf_size;
          io->GetNext(&buf, &buf_size, &end_of_input);
          if (buf != nullptr) {
            RETURN_IF_TRITONSERVER_ERROR(
                inference_request_append_input_data_fn_(
                    irequest, input_name, const_cast<uint8_t*>(buf), buf_size,
                    TRITONSERVER_MEMORY_CPU /* memory_type */,
                    0 /* memory_type_id */),
                "appending data to tritonserver");
          }
        }
      } else {
        std::string shm_name;
        size_t shm_byte_size;
        size_t offset;
        // TODO: Error handling
        io->SharedMemoryInfo(&shm_name, &shm_byte_size, &offset);
        void* buf;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        RETURN_IF_ERROR(shm_manager_->GetMemoryInfo(
            shm_name, offset, shm_byte_size, &buf, &memory_type,
            &memory_type_id));
        RETURN_IF_TRITONSERVER_ERROR(
            inference_request_append_input_data_fn_(
                irequest, input_name, buf, byte_size,
                memory_type /* memory_type */,
                memory_type_id /* memory_type_id */),
            "appending data to tritonserver");
      }
    }
  }


  return Error::Success;
}

Error
TritonLoader::AddOutputs(
    const std::vector<const tc::InferRequestedOutput*>& outputs,
    TRITONSERVER_InferenceRequest* irequest)
{
  for (auto io : outputs) {
    const char* output_name = io->Name().c_str();
    RETURN_IF_TRITONSERVER_ERROR(
        inference_request_add_requested_output_fn_(irequest, output_name),
        "setting output for the request");
  }
  return Error::Success;
}


Error
TritonLoader::ModelInferenceStatistics(
    const std::string& model_name, const std::string& model_version,
    rapidjson::Document* infer_stat)
{
  if (ServerIsReady() && ModelIsLoaded()) {
    TRITONSERVER_Message* model_stats_message = nullptr;
    int64_t requested_model_version;
    auto err =
        GetModelVersionFromString(model_version, &requested_model_version);
    if (err.IsOk()) {
      RETURN_IF_TRITONSERVER_ERROR(
          model_statistics_fn_(
              (server_).get(), model_name.c_str(), requested_model_version,
              &model_stats_message),
          "getting model statistics from server");

      const char* buffer;
      size_t byte_size;
      RETURN_IF_TRITONSERVER_ERROR(
          message_serialize_to_json_fn_(
              model_stats_message, &buffer, &byte_size),
          "serializing message to json");

      infer_stat->Parse(buffer, byte_size);
      if (infer_stat->HasParseError()) {
        return Error(
            "error: failed to parse server metadata from JSON: " +
            std::string(GetParseError_En(infer_stat->GetParseError())) +
            " at " + std::to_string(infer_stat->GetErrorOffset()));
      }
      RETURN_IF_TRITONSERVER_ERROR(
          message_delete_fn_(model_stats_message),
          "deleting inference statistics message");
    }
    return err;
  } else {
    return Error(
        "Trying to get model statistics while server is not started or model "
        "is not ready");
  }
}

TritonLoader*
TritonLoader::GetSingleton()
{
  static TritonLoader loader;
  return &loader;
}

TritonLoader::~TritonLoader()
{
  if (response_allocator_delete_fn_) {
    TRITONSERVER_Error* allocator_err =
        response_allocator_delete_fn_(allocator_);
    FAIL_IF_TRITONSERVER_ERROR(allocator_err, "deleting response allocator");
  }

  FAIL_IF_ERR(Delete(), "dereferencing server instance...");
  FAIL_IF_ERR(CloseLibraryHandle(dlhandle_), "error on closing triton loader");
  ClearHandles();
}

Error
TritonLoader::RegisterCudaMemory(
    const std::string& name, void* handle, const size_t byte_size)
{
  RETURN_IF_ERROR(shm_manager_->RegisterCUDAMemory(
      name, handle, byte_size, 0 /* device id */));
  return Error::Success;
}

Error
TritonLoader::RegisterSystemMemory(
    const std::string& name, void* ptr, const size_t byte_size)
{
  RETURN_IF_ERROR(shm_manager_->RegisterSystemMemory(name, ptr, byte_size));
  return Error::Success;
}

Error
TritonLoader::UnregisterAllSharedMemory()
{
  RETURN_IF_ERROR(shm_manager_->UnregisterAll(TRITONSERVER_MEMORY_GPU));
  RETURN_IF_ERROR(shm_manager_->UnregisterAll(TRITONSERVER_MEMORY_GPU));
  return Error::Success;
}

TRITONSERVER_Error*
TritonLoader::ErrorNew(TRITONSERVER_Error_Code code, const char* message)
{
  return error_new_fn_(code, message);
}

}}}}  // namespace triton::perfanalyzer::clientbackend::tritoncapi
