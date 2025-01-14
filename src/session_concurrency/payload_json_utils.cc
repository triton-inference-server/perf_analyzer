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

#include "../rapidjson_utils.h"

namespace triton::perfanalyzer {

std::string
PayloadJsonUtils::GetSessionID(const std::string& payload)
{
  const auto payload_document{GetPayloadDocument(payload)};

  if (!payload_document.IsObject() ||
      !payload_document.HasMember("session_id") ||
      !payload_document["session_id"].IsString()) {
    throw std::runtime_error(
        "Request body must be an object and it must have a 'session_id' "
        "field that is a string. Request body:\n\n" +
        RapidJsonUtils::Serialize(payload_document) + "\n\n\n");
  }

  return payload_document["session_id"].GetString();
}

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
PayloadJsonUtils::SetPayloadToChatHistory(
    rapidjson::Document& payload_document,
    const rapidjson::Document& chat_history)
{
  auto& payload_messages{GetPayloadMessages(payload_document)};

  payload_messages.CopyFrom(chat_history, payload_document.GetAllocator());
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
