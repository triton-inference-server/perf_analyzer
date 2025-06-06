// Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unordered_map>

#include "client_backend/client_backend.h"
#include "inference_load_mode.h"
#include "perf_utils.h"

namespace triton { namespace perfanalyzer {

#ifndef DOCTEST_CONFIG_DISABLE
class TestModelParser;
class MockModelParser;
class InferenceProfiler;
#endif

struct ModelTensor {
  std::string name_{};
  std::string datatype_{};
  std::vector<int64_t> shape_{};
  // Indicates if this tensor holds shape information for other tensors
  bool is_shape_tensor_{};
  bool is_optional_{};
};

using ModelTensorMap = std::map<std::string, ModelTensor>;
using ComposingModelMap = std::map<std::string, std::set<cb::ModelIdentifier>>;

//==============================================================================
/// ModelParser is a helper class to parse the information about the target
/// model from the metadata and configuration returned by the server.
///
/// Perf Analyzer depends upon the various properties of the model to correctly
/// generate and issue inference request for the model. The object of this
/// class will provide these necessary details.
class ModelParser {
 public:
  enum ModelSchedulerType {
    NONE,
    DYNAMIC,
    SEQUENCE,
    ENSEMBLE,
    ENSEMBLE_SEQUENCE
  };

  explicit ModelParser(
      InferenceLoadMode inference_load_mode = InferenceLoadMode::None);

  /// Initializes the ModelParser with the metadata and config rapidjson DOM
  /// for the target model obtained from Triton service
  /// \param metadata The metadata of the target model.
  /// \param config The config of the target model.
  /// \param model_version The version of target model.
  /// \param bls_composing_models A list of BLS composing model identifiers
  /// \param input_shapes The user provided default shapes which will be use
  /// if a certain input has wildcard in its dimension.
  /// \param backend The backend object.
  /// \return cb::Error object indicating success or failure.
  cb::Error InitTriton(
      const rapidjson::Document& metadata, const rapidjson::Document& config,
      const std::string& model_version,
      const std::vector<cb::ModelIdentifier>& bls_composing_models,
      const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
      std::unique_ptr<cb::ClientBackend>& backend);

  /// Initializes the ModelParser with the metadata and config rapidjson DOM
  /// for the target model obtained from TF serving service.
  /// \param metadata The metadata of the target model.
  /// \param model_name The name of target model.
  /// \param model_version The version of target model.
  /// \param model_signature_name The signature name of target model.
  /// \param input_shapes The user provided default shapes which will be use
  /// if a certain input has wildcard in its dimension.
  /// \param backend The backend object.
  /// \return cb::Error object indicating success or failure.
  cb::Error InitTFServe(
      const rapidjson::Document& metadata, const std::string& model_name,
      const std::string& model_version, const std::string& model_signature_name,
      const int32_t batch_size,
      const std::unordered_map<std::string, std::vector<int64_t>>& input_shapes,
      std::unique_ptr<cb::ClientBackend>& backend);

  cb::Error InitDynamicGrpc(
      const std::string& model_name, const std::string& model_version,
      const int32_t batch_size);

  cb::Error InitOpenAI(
      const std::string& model_name, const std::string& model_version,
      const int32_t batch_size);

  cb::Error InitTorchServe(
      const std::string& model_name, const std::string& model_version,
      const int32_t batch_size);

  /// Get the name of the target model
  /// \return Model name as string
  const std::string& ModelName() const { return model_name_; }

  /// Get the version of target model
  /// \return Model version as string
  const std::string& ModelVersion() const { return model_version_; }

  /// Get the signature name of target model
  /// \return Model signature name as string
  const std::string& ModelSignatureName() const
  {
    return model_signature_name_;
  }

  /// Get the scheduler type for the model
  ModelSchedulerType SchedulerType() const { return scheduler_type_; }

  /// Get the max batch size supported by the model. Returns 0 if the model
  /// does not support batching.
  /// \return The maximum supported batch size.
  size_t MaxBatchSize() const { return max_batch_size_; }

  /// Returns whether or not the model is decoupled
  /// \return the truth value of whether the model is decoupled
  bool IsDecoupled() const { return is_decoupled_; }

  /// Returns whether or not response cache is enabled for this model
  /// \return the truth value of whether response cache is enabled for this
  /// model
  bool ResponseCacheEnabled() const { return response_cache_enabled_; }

  /// Returns whether or not top level request caching is enabled for this model
  /// \return the truth value of whether top level request caching is enabled
  /// for this model
  bool TopLevelResponseCachingEnabled() const
  {
    return top_level_response_caching_enabled_;
  }

/// Only for testing
#ifndef DOCTEST_CONFIG_DISABLE
  void SetTopLevelResponseCaching(bool enable_top_level_response_caching)
  {
    top_level_response_caching_enabled_ = enable_top_level_response_caching;
  }
#endif

  /// Get the details about the model inputs.
  /// \return The map with tensor_name and the tensor details
  /// stored as key-value pair.
  const std::shared_ptr<ModelTensorMap>& Inputs() { return inputs_; }

  /// Get the details about the model outputs.
  /// \return The map with tensor_name and the tensor details
  /// stored as key-value pair.
  const std::shared_ptr<ModelTensorMap>& Outputs() { return outputs_; }

  /// Get the composing maps for the target model.
  /// \return The pointer to the nested map describing the
  /// nested flow in the target model.
  const std::shared_ptr<ComposingModelMap>& GetComposingModelMap()
  {
    return composing_models_map_;
  }


 protected:
  ModelSchedulerType scheduler_type_;
  bool is_decoupled_;

 private:
  /// Populate composing_models_map_ based on any bls composing models passed in
  /// via the CLI as well as any ensemble or nested ensemble models
  cb::Error DetermineComposingModelMap(
      const std::vector<cb::ModelIdentifier>& bls_composing_models,
      const rapidjson::Document& config,
      std::unique_ptr<cb::ClientBackend>& backend);

  cb::Error AddBLSComposingModels(
      const std::vector<cb::ModelIdentifier>& bls_composing_models,
      const rapidjson::Document& config,
      std::unique_ptr<cb::ClientBackend>& backend);

  cb::Error AddEnsembleComposingModels(
      const rapidjson::Document& config,
      std::unique_ptr<cb::ClientBackend>& backend);

  /// Populate scheduler_type_ based on the scheduler type of the parent model
  /// as well as any composing models
  cb::Error DetermineSchedulerType(
      const rapidjson::Document& config,
      std::unique_ptr<cb::ClientBackend>& backend);

  /// Sets is_sequential to true if any of the composing models are sequential
  cb::Error GetComposingSchedulerType(
      std::unique_ptr<cb::ClientBackend>& backend, bool* is_sequential);

  /// In the json produced by protobuf, int64 and uint64 values are
  /// represented as strings. Protobuf doesn't provide an option to
  /// disable this (sigh) so we need to correctly parse these fields
  /// for ModelParser to receive appropriate requests.
  /// \param value The rapidjson value object with the int value.
  /// \param integer_value The output integer pointer.
  /// \return cb::Error object indicating success or failure.
  cb::Error GetInt(const rapidjson::Value& value, int64_t* integer_value);

  std::shared_ptr<ModelTensorMap> inputs_;
  std::shared_ptr<ModelTensorMap> outputs_;
  std::shared_ptr<ComposingModelMap> composing_models_map_;

  std::string model_name_;
  std::string model_version_;
  std::string model_signature_name_;
  size_t max_batch_size_;
  bool response_cache_enabled_;
  bool top_level_response_caching_enabled_;

#ifndef DOCTEST_CONFIG_DISABLE
  friend TestModelParser;
  friend MockModelParser;
  friend InferenceProfiler;
#endif
};

}}  // namespace triton::perfanalyzer
