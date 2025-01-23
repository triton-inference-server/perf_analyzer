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

#include "response_json_utils.h"

#include <rapidjson/document.h>
#include <rapidjson/error/error.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "../rapidjson_utils.h"

namespace triton::perfanalyzer {

const rapidjson::Document
ResponseJsonUtils::GetResponseDocument(
    const std::vector<uint8_t>& response_buffer)
{
  rapidjson::Document response_document{};

  const std::string response_buffer_str(
      response_buffer.begin(), response_buffer.end());

  response_document.Parse(response_buffer_str.c_str(), response_buffer.size());

  if (response_document.HasParseError()) {
    throw std::runtime_error(
        "RapidJSON parse error " +
        std::to_string(response_document.GetParseError()) +
        ". Review JSON for formatting errors:\n\n" + response_buffer_str +
        "\n\n\n");
  }

  return response_document;
}

const rapidjson::Value&
ResponseJsonUtils::GetMessage(const rapidjson::Document& response_document)
{
  if (!response_document.IsObject() ||
      !response_document.HasMember("choices") ||
      !response_document["choices"].IsArray() ||
      response_document["choices"].Empty()) {
    throw std::runtime_error(
        "Response body must be an object and have a 'choices' field that is "
        "an array with at least one element. Response body:\n\n" +
        RapidJsonUtils::Serialize(response_document) + "\n\n\n");
  }

  const auto& response_first_choice{response_document["choices"][0]};

  if (!response_first_choice.IsObject() ||
      !response_first_choice.HasMember("message") ||
      !response_first_choice["message"].IsObject()) {
    throw std::runtime_error(
        "Response body 'choices' field's first element must be an object and "
        "have a 'message' field that is an object. Response body:\n\n" +
        RapidJsonUtils::Serialize(response_document) + "\n\n\n");
  }

  return response_first_choice["message"];
}

}  // namespace triton::perfanalyzer
