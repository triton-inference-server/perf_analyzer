// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <rapidjson/allocators.h>
#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <stddef.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "client_backend/client_backend.h"
#include "load_manager.h"
#include "model_parser.h"
#include "perf_utils.h"
#include "request_record.h"
#include "tensor_data.h"

namespace triton::perfanalyzer {

class SessionConcurrencyManager : public LoadManager {
 public:
  SessionConcurrencyManager(
      const bool async, const bool streaming, const int32_t batch_size,
      const size_t max_threads, const SharedMemoryType shared_memory_type,
      const size_t output_shm_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::unordered_map<std::string, cb::RequestParameter>&
          request_parameters,
      const size_t session_concurrency)
      : LoadManager(
            async, streaming, batch_size, max_threads, shared_memory_type,
            output_shm_size, parser, factory, request_parameters),
        session_concurrency_(session_concurrency)
  {
    const auto error{factory_->CreateClientBackend(&client_backend_)};

    if (!error.IsOk()) {
      throw std::runtime_error(error.Message());
    }
  }

  std::vector<RequestRecord> Start()
  {
    const auto session_streams{SplitDatasetIntoSessionStreams()};
    MakeAndWaitForThreads(session_streams);
    return GetRequestRecords();
  }

 private:
  std::vector<std::vector<size_t>> SplitDatasetIntoSessionStreams()
  {
    std::unordered_map<std::string, std::vector<size_t>> session_streams_map{};

    if (data_loader_->GetDataStreamsCount() != 1) {
      throw std::runtime_error(
          "Expected input data JSON to have one stream. Session concurrency "
          "mode must have an input data JSON with a single flat array for the "
          "\"data\" field with one element per request payload.");
    }

    const size_t dataset_size{data_loader_->GetTotalSteps(0)};

    for (size_t dataset_index{0}; dataset_index < dataset_size;
         ++dataset_index) {
      const auto session_id{GetSessionID(dataset_index)};
      session_streams_map[session_id].push_back(dataset_index);
    }

    if (session_streams_map.size() < session_concurrency_) {
      throw std::runtime_error(
          "The input data file contains " +
          std::to_string(session_streams_map.size()) +
          " unique session IDs, which is fewer than the required session "
          "concurrency level of " +
          std::to_string(session_concurrency_) +
          ". Please add more unique session IDs to the file or lower the "
          "session concurrency level.");
    }

    // create list where each element contains all indexes for that session ID
    std::vector<std::vector<size_t>> session_streams{};

    for (auto& [_, stream] : session_streams_map) {
      session_streams.push_back(std::move(stream));
    }

    return session_streams;
  }

  std::string GetSessionID(size_t dataset_index)
  {
    const auto payload{GetPayload(dataset_index)};

    rapidjson::Document payload_document{};
    payload_document.Parse(payload.c_str());

    if (!payload_document.HasMember("session_id") ||
        !payload_document["session_id"].IsString()) {
      throw std::runtime_error("Payload session ID must be a uuid4 string.");
    }

    const auto session_id{payload_document["session_id"].GetString()};

    return session_id;
  }

  std::string GetPayload(size_t dataset_index)
  {
    TensorData payload_tensor_data{};

    const auto error{data_loader_->GetInputData(
        (*parser_->Inputs())["payload"], 0, dataset_index,
        payload_tensor_data)};

    if (!error.IsOk()) {
      throw std::runtime_error(error.Message());
    }

    const uint8_t* payload_buffer{payload_tensor_data.data_ptr};
    const size_t payload_byte_size{payload_tensor_data.batch1_size};

    const std::string payload(
        reinterpret_cast<const char*>(payload_buffer), payload_byte_size);

    return payload;
  }

  void MakeAndWaitForThreads(
      const std::vector<std::vector<size_t>>& session_streams)
  {
    if (!threads_.empty()) {
      throw std::runtime_error("Expected threads_ to be empty.");
    }

    if (session_streams.size() != session_concurrency_) {
      throw std::runtime_error(
          "Expected session_streams size (" +
          std::to_string(session_streams.size()) +
          ") to match session concurrency (" +
          std::to_string(session_concurrency_) + ").");
    }

    for (size_t i{0}; i < session_concurrency_; ++i) {
      threads_.emplace_back(
          &SessionConcurrencyManager::SendSequentialRequestsForSessions, this,
          session_streams);
    }

    for (auto& thread : threads_) {
      thread.join();
    }
  }

  void SendSequentialRequestsForSessions(
      const std::vector<std::vector<size_t>>& session_streams)
  {
    while (true) {
      const size_t stream_index{session_streams_index_++};

      if (stream_index >= session_concurrency_) {
        break;
      }

      const auto session_stream{session_streams[stream_index]};

      SendSequentialRequestsForSession(session_stream);
    }
  }

  void SendSequentialRequestsForSession(
      const std::vector<size_t>& session_stream)
  {
    rapidjson::Document chat_history(rapidjson::kArrayType);

    for (size_t i{0}; i < session_stream.size(); ++i) {
      const size_t dataset_index{session_stream[i]};

      SendRequestAndWaitForResponse(dataset_index, chat_history);

      const bool is_last_request{i == session_stream.size() - 1};

      if (is_last_request) {
        break;
      }

      GetAndWaitForDelayMs(dataset_index);
    }
  }

  void SendRequestAndWaitForResponse(
      size_t dataset_index, rapidjson::Document& chat_history)
  {
    auto payload{GetPayload(dataset_index)};

    UpdateHistoryAndAddToPayload(payload, chat_history);

    auto response_promise{std::make_shared<std::promise<void>>()};
    std::future<void> response_future{response_promise->get_future()};

    // FIXME: remove session_id from payload cuz nim doesn't allow it
    {
      rapidjson::Document payload_document{};
      payload_document.Parse(payload.c_str());
      payload_document.RemoveMember("session_id");
      rapidjson::StringBuffer buffer{};
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
      payload_document.Accept(writer);
      payload = buffer.GetString();
    }

    SendRequest(payload, std::move(response_promise), chat_history);

    WaitForResponse(std::move(response_future));
  }

  void UpdateHistoryAndAddToPayload(
      std::string& payload, rapidjson::Document& chat_history)
  {
    auto payload_document{GetPayloadDocument(payload)};

    AddPayloadToChatHistory(payload_document, chat_history);

    SetPayloadToChatHistory(payload_document, chat_history);

    payload = GetSerialziedPayload(payload_document);
  }

  rapidjson::Document GetPayloadDocument(const std::string& payload)
  {
    rapidjson::Document payload_document{};

    payload_document.Parse(payload.c_str());

    return payload_document;
  }

  void AddPayloadToChatHistory(
      rapidjson::Document& payload_document, rapidjson::Document& chat_history)
  {
    auto& payload_messages{payload_document["messages"]};

    for (auto it{payload_messages.Begin()}; it != payload_messages.End();
         ++it) {
      auto& payload_message{*it};
      chat_history.PushBack(payload_message, chat_history.GetAllocator());
    }
  }

  void SetPayloadToChatHistory(
      rapidjson::Document& payload_document,
      const rapidjson::Document& chat_history)
  {
    auto& payload_messages{payload_document["messages"]};

    auto& payload_allocator{payload_document.GetAllocator()};

    payload_messages.CopyFrom(chat_history, payload_allocator);
  }

  std::string GetSerialziedPayload(const rapidjson::Document& payload_document)
  {
    rapidjson::StringBuffer buffer{};
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    payload_document.Accept(writer);
    return buffer.GetString();
  }

  void SendRequest(
      const std::string& payload,
      std::shared_ptr<std::promise<void>>&& response_promise,
      rapidjson::Document& chat_history)
  {
    const auto callback{
        PrepareCallback(std::move(response_promise), chat_history)};

    const cb::InferOptions options(parser_->ModelName());

    const auto inputs{PrepareInputs(payload)};

    const auto outputs{PrepareOutputs()};

    const auto error{
        client_backend_->AsyncInfer(callback, options, inputs, outputs)};

    if (!error.IsOk()) {
      throw std::runtime_error(error.Message());
    }
  }

  const std::function<void(cb::InferResult*)> PrepareCallback(
      std::shared_ptr<std::promise<void>>&& response_promise,
      rapidjson::Document& chat_history)
  {
    return [response_promise,
            &chat_history](cb::InferResult* infer_result) mutable {
      const std::string output_name{"response"};
      std::vector<uint8_t> buf{};
      infer_result->RawData(output_name, buf);

      std::cout << "response: ";
      for (uint8_t c : buf) {
        std::cout << (char)c;
      }
      std::cout << std::endl;

      response_promise->set_value();
    };
  }

  const std::vector<cb::InferInput*> PrepareInputs(const std::string& payload)
  {
    cb::InferInput* infer_input{};

    const cb::BackendKind kind{factory_->Kind()};
    const std::string name{"payload"};
    const std::vector<int64_t> dims{1};
    const std::string datatype{"BYTES"};

    auto error{
        cb::InferInput::Create(&infer_input, kind, name, dims, datatype)};

    if (!error.IsOk()) {
      throw std::runtime_error(error.Message());
    }

    const uint8_t* payload_buffer{
        reinterpret_cast<const uint8_t*>(payload.data())};
    const size_t payload_byte_size{payload.size()};

    error = infer_input->AppendRaw(payload_buffer, payload_byte_size);

    if (!error.IsOk()) {
      throw std::runtime_error(error.Message());
    }

    return {infer_input};
  }

  const std::vector<const cb::InferRequestedOutput*> PrepareOutputs()
  {
    cb::InferRequestedOutput* infer_output{};

    const cb::BackendKind kind{factory_->Kind()};
    const std::string& name{"response"};
    const std::string& datatype{"JSON"};
    const auto error{
        cb::InferRequestedOutput::Create(&infer_output, kind, name, datatype)};

    if (!error.IsOk()) {
      throw std::runtime_error(error.Message());
    }

    return {infer_output};
  }

  void WaitForResponse(std::future<void>&& response_future)
  {
    response_future.get();
  }

  void GetAndWaitForDelayMs(size_t dataset_index)
  {
    const uint64_t delay_ms{GetDelayMs(dataset_index)};

    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  }

  uint64_t GetDelayMs(size_t dataset_index)
  {
    TensorData delay_ms_tensor_data{};

    const auto error{data_loader_->GetInputData(
        (*parser_->Inputs())["delay"], 0, dataset_index, delay_ms_tensor_data)};

    if (!error.IsOk()) {
      throw std::runtime_error(error.Message());
    }

    const uint64_t delay_ms{
        *reinterpret_cast<const uint64_t*>(delay_ms_tensor_data.data_ptr)};

    return delay_ms;
  }

  std::vector<RequestRecord> GetRequestRecords()
  {
    std::vector<RequestRecord> request_records{};
    for (const auto& thread_stat : threads_stat_) {
      request_records.insert(
          request_records.end(),
          std::make_move_iterator(thread_stat->request_records_.begin()),
          std::make_move_iterator(thread_stat->request_records_.end()));
    }
    return request_records;
  }

  const size_t session_concurrency_{};
  std::unique_ptr<cb::ClientBackend> client_backend_{};
  std::atomic<size_t> session_streams_index_{};
};

}  // namespace triton::perfanalyzer
