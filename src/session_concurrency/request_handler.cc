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
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <stddef.h>

#include <cstdint>
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
    RequestRecord& request_record)
{
  auto payload{payload_dataset_manager_->GetPayload(dataset_index)};

  PayloadJsonUtils::RemoveSessionID(payload);

  PayloadJsonUtils::UpdateHistoryAndAddToPayload(payload, chat_history);

  auto response_promise{std::make_shared<std::promise<void>>()};
  std::future<void> response_future{response_promise->get_future()};

  SendRequest(
      payload, std::move(response_promise), chat_history, request_record);

  WaitForResponse(std::move(response_future));
}

void
RequestHandler::SendRequest(
    const std::string& payload,
    std::shared_ptr<std::promise<void>>&& response_promise,
    rapidjson::Document& chat_history, RequestRecord& request_record)
{
  const auto requested_outputs{PrepareRequestedOutputs()};

  const auto callback{PrepareCallback(
      std::move(response_promise), requested_outputs, request_record,
      chat_history)};

  const cb::InferOptions options(parser_->ModelName());

  const auto inputs{PrepareInputs(payload)};

  RecordRequest(inputs, request_record);

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
    RequestRecord& request_record, rapidjson::Document& chat_history) const
{
  return [response_promise, requested_outputs, &request_record, &chat_history,
          this](cb::InferResult* infer_result) mutable {
    if (!infer_result) {
      throw std::runtime_error("infer_result was null");
    } else if (!infer_result->RequestStatus().IsOk()) {
      throw std::runtime_error(infer_result->RequestStatus().Message());
    }

    RecordResponse(infer_result, requested_outputs, request_record);

    const auto response_buffer{GetResponseBuffer(infer_result)};

    const auto& response_document{
        ResponseJsonUtils::GetResponseDocument(response_buffer)};

    const auto& response_message{
        ResponseJsonUtils::GetMessage(response_document)};

    rapidjson::Value response_message_copy{};
    response_message_copy.CopyFrom(
        response_message, chat_history.GetAllocator());

    chat_history.PushBack(response_message_copy, chat_history.GetAllocator());

    response_promise->set_value();
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

  request_record.response_outputs_.emplace_back();

  auto& response_outputs{request_record.response_outputs_.back()};

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
RequestHandler::RecordRequest(
    const std::vector<cb::InferInput*> inputs,
    RequestRecord& request_record) const
{
  request_record.request_inputs_.emplace_back();

  auto& request_inputs{request_record.request_inputs_.back()};

  RecordRequestInputs(inputs, request_inputs);

  request_record.start_time_ = std::chrono::system_clock::now();
}

void
RequestHandler::RecordRequestInputs(
    const std::vector<cb::InferInput*> inputs,
    RequestRecord::RequestInput& request_inputs) const
{
  for (const auto& input : inputs) {
    std::string data_type{input->Datatype()};
    const uint8_t* buf{nullptr};
    size_t byte_size{0};
    input->RawData(&buf, &byte_size);

    std::vector<uint8_t> buf_vec(buf, buf + byte_size);

    request_inputs.emplace(
        input->Name(), RecordData(std::move(buf_vec), data_type));
  }
}

void
RequestHandler::WaitForResponse(std::future<void>&& response_future) const
{
  response_future.get();
}

}  // namespace triton::perfanalyzer
