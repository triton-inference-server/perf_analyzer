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

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "doctest.h"
#include "session_concurrency/response_json_utils.h"

namespace triton::perfanalyzer {

TEST_CASE("ResponseJsonUtils::GetResponseMessage")
{
  SUBCASE("valid json")
  {
    const std::string response{R"(
        {
          "choices": [
            {
              "message": {
                "role": "my_role",
                "content": "my_content"
              }
            }
          ]
        }
        )"};
    const std::vector<uint8_t> response_buffer(
        response.begin(), response.end());

    const auto& message_value{
        ResponseJsonUtils::GetResponseMessage(response_buffer)};

    REQUIRE(message_value.IsObject());
    REQUIRE(message_value.MemberCount() == 2);
    REQUIRE(message_value.HasMember("role"));
    REQUIRE(message_value["role"].IsString());

    const auto& role_value{message_value["role"]};
    const std::string role(
        role_value.GetString(), role_value.GetStringLength());

    CHECK(role == "my_role");

    REQUIRE(message_value.HasMember("content"));
    REQUIRE(message_value["content"].IsString());

    const auto& content_value{message_value["content"]};
    const std::string content(
        content_value.GetString(), content_value.GetStringLength());

    CHECK(content == "my_content");
  }

  SUBCASE("invalid json")
  {
    const std::string raw{""};
    const std::vector<uint8_t> response_buffer(raw.begin(), raw.end());

    CHECK_THROWS_WITH_AS(
        ResponseJsonUtils::GetResponseMessage(response_buffer),
        "rapidjson parse error 1", std::runtime_error);
  }

  SUBCASE("invalid choices")
  {
    const std::string response{R"(
        {
          "choices": []
        }
        )"};
    const std::vector<uint8_t> response_buffer(
        response.begin(), response.end());

    CHECK_THROWS_WITH_AS(
        ResponseJsonUtils::GetResponseMessage(response_buffer),
        "Response body must be an object and have a 'choices' field that is an "
        "array with at least one element.",
        std::runtime_error);
  }

  SUBCASE("invalid message")
  {
    const std::string response{R"(
        {
          "choices": [
            {
              "message": false
            }
          ]
        }
        )"};
    const std::vector<uint8_t> response_buffer(
        response.begin(), response.end());

    CHECK_THROWS_WITH_AS(
        ResponseJsonUtils::GetResponseMessage(response_buffer),
        "Response body 'choices' field's first element must be an object and "
        "have a 'message' field that is an object.",
        std::runtime_error);
  }
}

}  // namespace triton::perfanalyzer
