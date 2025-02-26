// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "payload_dataset_manager.h"

#include <stddef.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../client_backend/client_backend.h"
#include "../data_loader.h"
#include "../model_parser.h"
#include "../tensor_data.h"

namespace triton::perfanalyzer {

PayloadDatasetManager::PayloadDatasetManager(
    std::shared_ptr<const DataLoader> data_loader,
    const std::shared_ptr<ModelParser> parser)
    : data_loader_(data_loader), parser_(parser)
{
}

std::vector<std::vector<size_t>>
PayloadDatasetManager::GroupPayloadsBySession() const
{
  if (data_loader_->GetDataStreamsCount() != 1) {
    throw std::runtime_error(
        "Expected input data JSON to have one stream. Session concurrency "
        "mode must have an input data JSON with a single flat array for the "
        "\"data\" field with one element per request payload.");
  }

  auto session_id_to_dataset_map{CreateSessionIdToPayloadsMap()};

  std::vector<std::vector<size_t>> session_datasets{};

  for (auto& [_, dataset] : session_id_to_dataset_map) {
    session_datasets.push_back(std::move(dataset));
  }

  return session_datasets;
}

std::optional<std::chrono::milliseconds>
PayloadDatasetManager::GetDelay(size_t dataset_index) const
{
  TensorData delay_ms_tensor_data{};

  const auto error{data_loader_->GetInputData(
      (*parser_->Inputs())["delay"], 0, dataset_index, delay_ms_tensor_data)};

  if (!error.IsOk()) {
    throw std::runtime_error(error.Message());
  }

  if (!delay_ms_tensor_data.data_ptr) {
    return {};
  }

  const uint64_t delay_ms{
      *reinterpret_cast<const uint64_t*>(delay_ms_tensor_data.data_ptr)};

  return std::chrono::milliseconds(delay_ms);
}

std::string
PayloadDatasetManager::GetPayload(size_t dataset_index) const
{
  TensorData payload_tensor_data{};

  const auto error{data_loader_->GetInputData(
      (*parser_->Inputs())["payload"], 0, dataset_index, payload_tensor_data)};

  if (!error.IsOk()) {
    throw std::runtime_error(error.Message());
  }

  const uint8_t* payload_buffer{payload_tensor_data.data_ptr};
  const size_t payload_byte_size{payload_tensor_data.batch1_size};

  const std::string payload(
      reinterpret_cast<const char*>(payload_buffer), payload_byte_size);

  return payload;
}

PayloadDatasetManager::PayloadsMapType
PayloadDatasetManager::CreateSessionIdToPayloadsMap() const
{
  PayloadsMapType session_id_to_dataset_map{};

  const size_t dataset_size{data_loader_->GetTotalSteps(0)};

  for (size_t dataset_index{0}; dataset_index < dataset_size; ++dataset_index) {
    const auto session_id{GetSessionID(dataset_index)};
    session_id_to_dataset_map[session_id].push_back(dataset_index);
  }

  return session_id_to_dataset_map;
}

std::string
PayloadDatasetManager::GetSessionID(size_t dataset_index) const
{
  TensorData session_id_tensor_data{};

  const auto error{data_loader_->GetInputData(
      (*parser_->Inputs())["session_id"], 0, dataset_index,
      session_id_tensor_data)};

  if (!error.IsOk()) {
    throw std::runtime_error(error.Message());
  }

  uint32_t session_id_byte_size{};
  std::memcpy(
      &session_id_byte_size, session_id_tensor_data.data_ptr, sizeof(uint32_t));

  const uint8_t* session_id_buffer{
      session_id_tensor_data.data_ptr + sizeof(uint32_t)};

  const std::string session_id(
      reinterpret_cast<const char*>(session_id_buffer), session_id_byte_size);

  return session_id;
}

}  // namespace triton::perfanalyzer
