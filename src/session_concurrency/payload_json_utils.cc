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
    std::string& payload, rapidjson::Document& chat_history,
    std::vector<std::pair<size_t, size_t>>& one_session_chunk_ranges)
{
  auto payload_document{GetPayloadDocument(payload)};

  AddPayloadToChatHistory(payload_document, chat_history, one_session_chunk_ranges);

  SetPayloadToChatHistory(payload_document, chat_history, one_session_chunk_ranges);

  payload = GetSerializedPayload(payload_document);
}

void
PayloadJsonUtils::AddPayloadToChatHistory(
    const rapidjson::Document& payload_document,
    rapidjson::Document& chat_history,
    std::vector<std::pair<size_t, size_t>>& one_session_chunk_ranges)
{
  const auto& payload_messages{GetPayloadMessages(payload_document)};

  rapidjson::Value payload_messages_copy{};
  payload_messages_copy.CopyFrom(payload_messages, chat_history.GetAllocator());

  size_t last_index_chunk_ranges = 0;
  if (!one_session_chunk_ranges.empty()) {
    auto& last_range = one_session_chunk_ranges.back();
    last_index_chunk_ranges = last_range.second;
  }

  for (auto& payload_message : payload_messages_copy.GetArray()) {
    chat_history.PushBack(payload_message, chat_history.GetAllocator());
    one_session_chunk_ranges.emplace_back(last_index_chunk_ranges, last_index_chunk_ranges + 1);
    last_index_chunk_ranges++;
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
  // NOTE: Content key is hardcoded.
  //       This may change depending on the target inference framework.
  std::string c = std::string(item["content"].GetString()) + buffer;
  item["content"].SetString(c.c_str(), c.size(), allocator);
}

void
PayloadJsonUtils::SetPayloadToChatHistory(
    rapidjson::Document& payload_document,
    const rapidjson::Document& chat_history,
    std::vector<std::pair<size_t, size_t>>& one_session_chunk_ranges)
{
  auto& payload_messages{GetPayloadMessages(payload_document)};

  // Merge chunked responses in streaming mode.
  rapidjson::Document merged_history{};
  merged_history.Parse("[]");
  auto& allocator = merged_history.GetAllocator();
  std::vector<rapidjson::Value> values{};
  std::string content_buffer{};
  std::string content_key{};
  size_t history_index = 0;
  size_t chunk_range_index = 0;
  for (auto& h : chat_history.GetArray()) {
    // This merge sequence assumes that:
    // 1. the order of arrivals is preserved in chat_history,
    // 2. for request payload and non-streaming response,
    //    each entry in chat_history includes the entire text which is not chunked,
    // 3. for streaming response, each chunk has "role" field.
    //    note that it's depending on inference framework about
    //    what fields and values are filled.
    // 4. each chunk doesn't have inconsistent value,
    //    that is, "role" and/or "function_call" field don't have
    //    different values for one sequence.
    //    (e.g., the situation, chunks[0]["role"]: "assistant" and chunks[1]["role"]: "user", never happens)
    auto& chunk_range{one_session_chunk_ranges[chunk_range_index]};
    size_t range_head_index = chunk_range.first;
    size_t range_tail_index = chunk_range.second;  // NOTE: exclusive

    bool is_first = (history_index == range_head_index);
    bool is_last = (history_index == (range_tail_index - 1));

    if (is_first) {
      // First chunk of this range.
      // Create new object for this range and
      // copy entire object into new instance.
      auto& new_item = values.emplace_back();
      new_item.CopyFrom(h, allocator);
      if (!new_item.HasMember("content")) {
        // For trtllm-serve, empty string must be set as "content".
        new_item.AddMember("content", "", allocator);
      }
    } else {
      // If not first chunk, append each chunk into buffer.
      if (h.HasMember("content") && !h["content"].IsNull()) {
        content_key = "content";
        content_buffer.append(h[content_key.c_str()].GetString());
      } else if (h.HasMember("reasoning_content") && !h["reasoning_content"].IsNull()) {
        content_key = "reasoning_content";
        content_buffer.append(h[content_key.c_str()].GetString());
      } else if (!is_last) {
        // Depending on inference framework, first or last chunk doesn't have
        // content or reasoning_content field, or these fields can be null.
        // But, if intermediate chunks don't have these fields or null value,
        // the situation is unexpected.
        throw std::runtime_error(
          "Request payload or response chunks must have at least one content or reasoning_content: history_index = "
          + std::to_string(history_index)
          + ", chunk_range_index = "
          + std::to_string(chunk_range_index)
          + "\n\n\n");
      }
    }

    if (is_last) {
      // Last chunk of this range
      if (!is_first) {
        // Apply the buffer text into the object for this range and clear buffer.
        // Note that in the case of a single request payload, this should be skipped.
        auto& new_item = values.back();
        UpdateContent(new_item, content_buffer, allocator);
      }

      content_buffer.clear();
      chunk_range_index++;
    }

    // Count up index for chat_history.
    history_index++;
  }

  // Store the final entry if it exists.
  if (!content_buffer.empty()) {
    // Apply the buffer text into the object for this range and clear buffer.
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
