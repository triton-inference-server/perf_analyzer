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

#include "request_handler.h"

#include <rapidjson/allocators.h>
#include <rapidjson/document.h>
#include <stddef.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../client_backend/client_backend.h"
#include "../model_parser.h"
#include "../request_record.h"
#include "../rapidjson_utils.h"
#include "payload_dataset_manager.h"
#include "payload_json_utils.h"
#include "response_json_utils.h"

namespace triton::perfanalyzer {

RequestHandler::RequestHandler(
    std::shared_ptr<cb::ClientBackendFactory> factory,
    const std::shared_ptr<ModelParser> parser,
    std::shared_ptr<PayloadDatasetManager> payload_dataset_manager)
    : factory_(factory), parser_(parser),
      payload_dataset_manager_(payload_dataset_manager)
{
  const auto error{factory->CreateClientBackend(&client_backend_)};

  if (!error.IsOk()) {
    throw std::runtime_error(error.Message());
  }
}

void
RequestHandler::SendRequestAndWaitForResponse(
    size_t dataset_index, rapidjson::Document& chat_history,
    std::vector<std::pair<size_t, size_t>>& one_session_chunk_ranges,
    RequestRecord& request_record)
{
  auto payload{payload_dataset_manager_->GetPayload(dataset_index)};

  PayloadJsonUtils::UpdateHistoryAndAddToPayload(payload, chat_history, one_session_chunk_ranges);

  auto& request_inputs{request_record.request_inputs_.emplace_back()};

  RecordRequestInputs(payload, dataset_index, request_inputs);

  auto response_promise{std::make_shared<std::promise<void>>()};
  std::future<void> response_future{response_promise->get_future()};

  size_t last_index_chunk_ranges = 0;
  if (!one_session_chunk_ranges.empty()) {
    auto& last_range = one_session_chunk_ranges.back();
    last_index_chunk_ranges = last_range.second;
  }
  SendRequest(
      payload, std::move(response_promise), chat_history, one_session_chunk_ranges, last_index_chunk_ranges, request_record);

  WaitForResponse(std::move(response_future));
}

void
RequestHandler::RecordRequestInputs(
    const std::string& payload, size_t dataset_index,
    RequestRecord::RequestInput& request_inputs) const
{
  RecordPayloadInput(payload, request_inputs);

  const auto session_id{payload_dataset_manager_->GetSessionID(dataset_index)};
  RecordSessionIDInput(session_id, request_inputs);

  const auto delay{payload_dataset_manager_->GetDelay(dataset_index)};
  if (delay) {
    RecordDelayInput(*delay, request_inputs);
  }
}

void
RequestHandler::RecordPayloadInput(
    const std::string& payload,
    RequestRecord::RequestInput& request_inputs) const
{
  std::vector<uint8_t> payload_buf(payload.begin(), payload.end());
  request_inputs.emplace("payload", RecordData(std::move(payload_buf), "JSON"));
}

void
RequestHandler::RecordSessionIDInput(
    const std::string& session_id,
    RequestRecord::RequestInput& request_inputs) const
{
  std::vector<uint8_t> session_id_buf(session_id.begin(), session_id.end());
  request_inputs.emplace(
      "session_id", RecordData(std::move(session_id_buf), "BYTES"));
}

void
RequestHandler::RecordDelayInput(
    const std::chrono::milliseconds delay,
    RequestRecord::RequestInput& request_inputs) const
{
  const uint64_t delay_ms{static_cast<uint64_t>(delay.count())};
  std::vector<uint8_t> delay_buf(sizeof(uint64_t));
  std::memcpy(delay_buf.data(), &delay_ms, sizeof(uint64_t));
  request_inputs.emplace("delay", RecordData(std::move(delay_buf), "UINT64"));
}

void
RequestHandler::SendRequest(
    const std::string& payload,
    std::shared_ptr<std::promise<void>>&& response_promise,
    rapidjson::Document& chat_history,
    std::vector<std::pair<size_t, size_t>>& one_session_chunk_ranges,
    size_t last_index_chunk_ranges,
    RequestRecord& request_record)
{
  const auto requested_outputs{PrepareRequestedOutputs()};

  const auto callback{PrepareCallback(
      std::move(response_promise), requested_outputs, request_record,
      chat_history, one_session_chunk_ranges, last_index_chunk_ranges)};

  const cb::InferOptions options(parser_->ModelName());

  const auto inputs{PrepareInputs(payload)};

  request_record.start_time_ = std::chrono::system_clock::now();

  const auto error{client_backend_->AsyncInfer(
      callback, options, inputs, requested_outputs)};

  if (!error.IsOk()) {
    throw std::runtime_error(error.Message());
  }
}

const std::vector<const cb::InferRequestedOutput*>
RequestHandler::PrepareRequestedOutputs() const
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

const std::function<void(cb::InferResult*)>
RequestHandler::PrepareCallback(
    std::shared_ptr<std::promise<void>>&& response_promise,
    const std::vector<const cb::InferRequestedOutput*>& requested_outputs,
    RequestRecord& request_record, rapidjson::Document& chat_history,
    std::vector<std::pair<size_t, size_t>>& one_session_chunk_ranges,
    size_t last_index_chunk_ranges) const
{
  return [response_promise, requested_outputs, &request_record, &chat_history,
          &one_session_chunk_ranges, last_index_chunk_ranges, this](cb::InferResult* infer_result) mutable {
    if (!infer_result) {
      throw std::runtime_error("infer_result was null");
    } else if (!infer_result->RequestStatus().IsOk()) {
      throw std::runtime_error(infer_result->RequestStatus().Message());
    }

    RecordResponse(infer_result, requested_outputs, request_record);

    const auto response_buffer{GetResponseBuffer(infer_result)};

    const auto& response_document{
        ResponseJsonUtils::GetResponseDocument(response_buffer)};

    bool is_stream{false};
    auto error = infer_result->IsStreamResponse(&is_stream);
    if (!error.IsOk()) {
      // Forcibly set false to `is_stream` because
      // this `infer_result` object is a subclass which
      // does not implement `IsStreamResponse()`.
      is_stream = false;
    }
    bool is_final{false};
    error = infer_result->IsFinalResponse(&is_final);
    if (!error.IsOk()) {
      // Forcibly set false to `is_final`.
      is_final = false;
    }

    if (!response_document.IsNull()) {
      // `response_document` should not be null
      // when the response text is empty ("").
      // Null can happen only when response is `data: [DONE]`.

      if (is_stream && is_final) {
        // Unexpected response.
        throw std::runtime_error(
          "In the case of streaming and the last chunk, response object must be null:\n\n" +
          RapidJsonUtils::Serialize(response_document) + "\n\n\n"
        );
      }

      rapidjson::Value response_message_copy{};
      if (is_stream) {
        response_message_copy.CopyFrom(
            ResponseJsonUtils::GetDelta(response_document),
            chat_history.GetAllocator());
      } else {
        response_message_copy.CopyFrom(
            ResponseJsonUtils::GetMessage(response_document),
            chat_history.GetAllocator());
      }

      chat_history.PushBack(response_message_copy, chat_history.GetAllocator());
      auto& last_range = one_session_chunk_ranges.back();
      size_t head_idx = last_range.first;
      size_t tail_idx = last_range.second;

      if (tail_idx == last_index_chunk_ranges) {
        one_session_chunk_ranges.emplace_back(last_index_chunk_ranges, last_index_chunk_ranges + 1);
      } else {
        last_range.second++;
      }
    }

    if (is_final) {
      response_promise->set_value();
    }
  };
}

void
RequestHandler::RecordResponse(
    cb::InferResult* infer_result,
    const std::vector<const cb::InferRequestedOutput*>& requested_outputs,
    RequestRecord& request_record) const
{
  const auto& end_time{std::chrono::system_clock::now()};

  request_record.response_timestamps_.push_back(end_time);

  auto& response_outputs{request_record.response_outputs_.emplace_back()};

  RecordResponseOutputs(infer_result, requested_outputs, response_outputs);
}

void
RequestHandler::RecordResponseOutputs(
    cb::InferResult* infer_result,
    const std::vector<const cb::InferRequestedOutput*>& requested_outputs,
    RequestRecord::ResponseOutput& response_outputs) const
{
  for (const auto& requested_output : requested_outputs) {
    const auto& name{requested_output->Name()};

    const auto& data_type{requested_output->Datatype()};

    std::vector<uint8_t> buf{};
    infer_result->RawData(name, buf);

    response_outputs.emplace(name, RecordData(std::move(buf), data_type));
  }
}

const std::vector<uint8_t>
RequestHandler::GetResponseBuffer(cb::InferResult* infer_result) const
{
  const std::string output_name{"response"};
  std::vector<uint8_t> buffer{};

  const auto error{infer_result->RawData(output_name, buffer)};

  if (!error.IsOk()) {
    throw std::runtime_error(error.Message());
  }

  return buffer;
}

const std::vector<cb::InferInput*>
RequestHandler::PrepareInputs(const std::string& payload) const
{
  cb::InferInput* infer_input{};

  const cb::BackendKind kind{factory_->Kind()};
  const std::string name{"payload"};
  const std::vector<int64_t> dims{1};
  const std::string datatype{"BYTES"};

  auto error{cb::InferInput::Create(&infer_input, kind, name, dims, datatype)};

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

void
RequestHandler::WaitForResponse(std::future<void>&& response_future) const
{
  response_future.get();
}

}  // namespace triton::perfanalyzer
