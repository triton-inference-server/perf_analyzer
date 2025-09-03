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

#include "payload_json_utils.h"

#include <rapidjson/allocators.h>
#include <rapidjson/document.h>
#include <rapidjson/error/error.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "../rapidjson_utils.h"

namespace triton::perfanalyzer {

void
PayloadJsonUtils::UpdateHistoryAndAddToPayload(
    std::string& payload, rapidjson::Document& chat_history)
{
  auto payload_document{GetPayloadDocument(payload)};

  AddPayloadToChatHistory(payload_document, chat_history);

  SetPayloadToChatHistory(payload_document, chat_history);

  payload = GetSerializedPayload(payload_document);
}

void
PayloadJsonUtils::AddPayloadToChatHistory(
    const rapidjson::Document& payload_document,
    rapidjson::Document& chat_history)
{
  const auto& payload_messages{GetPayloadMessages(payload_document)};

  rapidjson::Value payload_messages_copy{};
  payload_messages_copy.CopyFrom(payload_messages, chat_history.GetAllocator());

  for (auto& payload_message : payload_messages_copy.GetArray()) {
    chat_history.PushBack(payload_message, chat_history.GetAllocator());
  }
}

const rapidjson::Value&
PayloadJsonUtils::GetPayloadMessages(
    const rapidjson::Document& payload_document)
{
  ValidatePayloadMessages(payload_document);
  return payload_document["messages"];
}

rapidjson::Value&
PayloadJsonUtils::GetPayloadMessages(rapidjson::Document& payload_document)
{
  ValidatePayloadMessages(payload_document);
  return payload_document["messages"];
}

void
PayloadJsonUtils::ValidatePayloadMessages(
    const rapidjson::Document& payload_document)
{
  if (!payload_document.IsObject() || !payload_document.HasMember("messages") ||
      !payload_document["messages"].IsArray()) {
    throw std::runtime_error(
        "Request body must be an object and it must have a 'messages' field "
        "that is an array. Request body:\n\n" +
        RapidJsonUtils::Serialize(payload_document) + "\n\n\n");
  }
}

void
PayloadJsonUtils::UpdateContent(
    rapidjson::Value& item,
    std::string& buffer,
    rapidjson::Document::AllocatorType& allocator)
{
  std::string c = std::string(item["content"].GetString()) + buffer;
  item["content"].SetString(c.c_str(), c.size(), allocator);
}

void
PayloadJsonUtils::SetPayloadToChatHistory(
    rapidjson::Document& payload_document,
    const rapidjson::Document& chat_history)
{
  auto& payload_messages{GetPayloadMessages(payload_document)};

  // Merge chunked responses in streaming mode.
  rapidjson::Document merged_history{};
  merged_history.Parse("[]");
  auto& allocator = merged_history.GetAllocator();
  std::vector<rapidjson::Value> values{};
  std::string content_buffer{};
  for (auto& h : chat_history.GetArray()) {
    // This merge sequence assumes that:
    // 1. the order of arrivals is preserved in chat_history,
    // 2. for request payload and non-streaming response,
    //    each entry in chat_history includes the entire text which is not chunked,
    // 3. for streaming response, each chunk has "role" field,
    //    but the value of chunks execpt for the first one is null,
    // 4. each chunk doesn't have inconsistent value,
    //    that is, "role" and/or "function_call" field don't have
    //    different values for one sequence.
    //    (e.g., the situation, chunks[0]["role"]: "assistant" and chunks[1]["role"]: "user", never happens)
    auto& role{h["role"]};

    if (role.IsNull()) {
      // Intermediate streaming chunks corresponding to one request.
      content_buffer.append(h["content"].GetString());
    } else {
      std::string role_str{role.GetString()};

      if (!content_buffer.empty()) {
        auto& new_item = values.back();
        UpdateContent(new_item, content_buffer, allocator);
        content_buffer.clear();
      }

      // First streaming chunk or Request payload.
      auto& new_item = values.emplace_back();
      new_item.CopyFrom(h, allocator);
    }
  }

  // Store the final entry if it exists.
  if (!content_buffer.empty()) {
    auto& new_item = values.back();
    UpdateContent(new_item, content_buffer, allocator);
    content_buffer.clear();
  }

  // Convert multiple Value objects into one Value instance.
  for (auto& v : values) {
    merged_history.PushBack(v, allocator);
  }

  payload_messages.CopyFrom(merged_history, payload_document.GetAllocator());
}

std::string
PayloadJsonUtils::GetSerializedPayload(
    const rapidjson::Document& payload_document)
{
  rapidjson::StringBuffer buffer{};
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  payload_document.Accept(writer);
  return buffer.GetString();
}

rapidjson::Document
PayloadJsonUtils::GetPayloadDocument(const std::string& payload)
{
  rapidjson::Document payload_document{};

  payload_document.Parse(payload.c_str());

  if (payload_document.HasParseError()) {
    throw std::runtime_error(
        "RapidJSON parse error " +
        std::to_string(payload_document.GetParseError()) +
        ". Review JSON for formatting errors:\n\n" + payload + "\n\n\n");
  }

  return payload_document;
}

}  // namespace triton::perfanalyzer
