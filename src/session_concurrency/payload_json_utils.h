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
#pragma once

#include <rapidjson/document.h>

#include <string>

namespace triton::perfanalyzer {

class PayloadJsonUtils {
 public:
  static std::string GetSessionID(const std::string& payload);

  static void RemoveSessionID(std::string& payload);

  static void UpdateHistoryAndAddToPayload(
      std::string& payload, rapidjson::Document& chat_history);

 private:
  static void AddPayloadToChatHistory(
      const rapidjson::Document& payload_document,
      rapidjson::Document& chat_history);

  static const rapidjson::Value& GetPayloadMessages(
      const rapidjson::Document& payload_document);

  static rapidjson::Value& GetPayloadMessages(
      rapidjson::Document& payload_document);

  static void ValidatePayloadMessages(
      const rapidjson::Document& payload_document);

  static void SetPayloadToChatHistory(
      rapidjson::Document& payload_document,
      const rapidjson::Document& chat_history);

  static std::string GetSerializedPayload(
      const rapidjson::Document& payload_document);

  static rapidjson::Document GetPayloadDocument(const std::string& payload);
};

}  // namespace triton::perfanalyzer
