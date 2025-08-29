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

#include <stdexcept>
#include <string>

#include "doctest.h"
#include "session_concurrency/payload_json_utils.h"

namespace triton::perfanalyzer {

TEST_CASE("PayloadJsonUtils::UpdateHistoryAndAddToPayload")
{
  SUBCASE("valid payload and chat history")
  {
    std::string payload{R"(
        {
          "messages": [
            {
              "role": "my_role_2",
              "content": "my_content_2"
            }
          ]
        }
        )"};

    const std::string chat_history_raw{R"(
        [
          {
            "role": "my_role_1",
            "content": "my_content_1"
          }
        ]
        )"};
    rapidjson::Document chat_history{};
    chat_history.Parse(chat_history_raw.c_str());

    PayloadJsonUtils::UpdateHistoryAndAddToPayload(payload, chat_history);

    rapidjson::Document payload_document{};
    payload_document.Parse(payload.c_str());

    const std::string expected_payload{R"(
        {
          "messages": [
            {
              "role": "my_role_1",
              "content": "my_content_1"
            },
            {
              "role": "my_role_2",
              "content": "my_content_2"
            }
          ]
        }
        )"};
    rapidjson::Document expected_payload_document{};
    expected_payload_document.Parse(expected_payload.c_str());

    CHECK(payload_document == expected_payload_document);

    const std::string expected_chat_history_raw{R"(
        [
          {
            "role": "my_role_1",
            "content": "my_content_1"
          },
          {
            "role": "my_role_2",
            "content": "my_content_2"
          }
        ]
        )"};
    rapidjson::Document expected_chat_history{};
    expected_chat_history.Parse(expected_chat_history_raw.c_str());

    CHECK(chat_history == expected_chat_history);
  }

  SUBCASE("invalid payload - not parsable")
  {
    std::string payload{""};
    rapidjson::Document chat_history{};

    CHECK_THROWS_WITH_AS(
        PayloadJsonUtils::UpdateHistoryAndAddToPayload(payload, chat_history),
        "RapidJSON parse error 1. Review JSON for formatting errors:\n\n\n\n\n",
        std::runtime_error);
  }

  SUBCASE("invalid payload - messages not an array")
  {
    std::string payload{R"(
        {
          "messages": false
        }
        )"};
    rapidjson::Document chat_history{};

    CHECK_THROWS_WITH_AS(
        PayloadJsonUtils::UpdateHistoryAndAddToPayload(payload, chat_history),
        "Request body must be an object and it must have a 'messages' field "
        "that is an array. Request body:\n\n{\"messages\":false}\n\n\n",
        std::runtime_error);
  }
}

TEST_CASE("PayloadJsonUtils::UpdateHistoryAndAddToPayload for multi-turn session")
{
  SUBCASE("valid payload and chat history with chunked responsens")
  {
    std::string payload{R"(
        {
          "messages": [
            {
              "role": "my_role_1",
              "content": "my_content_3"
            }
          ]
        }
        )"};

    const std::string chat_history_raw{R"(
        [
          {
            "role": "my_role_1",
            "content": "my_content_1"
          },
          {
            "role": "my_role_2",
            "content": "my_content_2-1",
            "function_call": null
          },
          {
            "role": null,
            "content": ":my_content_2-1:",
            "function_call": null
          },
          {
            "role": null,
            "content": ":my_content_2-2:",
            "function_call": null
          },
          {
            "role": null,
            "content": ":my_content_2-3:",
            "function_call": null
          }
        ]
        )"};
    rapidjson::Document chat_history{};
    rapidjson::ParseResult parse_ok = chat_history.Parse(chat_history_raw.c_str());
    CHECK(parse_ok);

    PayloadJsonUtils::UpdateHistoryAndAddToPayload(payload, chat_history);

    rapidjson::Document payload_document{};
    parse_ok = payload_document.Parse(payload.c_str());
    CHECK(parse_ok);

    const std::string expected_payload{R"(
        {
          "messages": [
            {
              "role": "my_role_1",
              "content": "my_content_1"
            },
            {
              "role": "my_role_2",
              "content": "my_content_2-1:my_content_2-1::my_content_2-2::my_content_2-3:",
              "function_call": null
            },
            {
              "role": "my_role_1",
              "content": "my_content_3"
            }
          ]
        }
        )"};
    rapidjson::Document expected_payload_document{};
    parse_ok = expected_payload_document.Parse(expected_payload.c_str());
    CHECK(parse_ok);

    CHECK(payload_document == expected_payload_document);

    const std::string expected_chat_history_raw{R"(
        [
          {
            "role": "my_role_1",
            "content": "my_content_1"
          },
          {
            "role": "my_role_2",
            "content": "my_content_2-1",
            "function_call": null
          },
          {
            "role": null,
            "content": ":my_content_2-1:",
            "function_call": null
          },
          {
            "role": null,
            "content": ":my_content_2-2:",
            "function_call": null
          },
          {
            "role": null,
            "content": ":my_content_2-3:",
            "function_call": null
          },
          {
            "role": "my_role_1",
            "content": "my_content_3"
          }
        ]
        )"};
    rapidjson::Document expected_chat_history{};
    parse_ok = expected_chat_history.Parse(expected_chat_history_raw.c_str());
    CHECK(parse_ok);

    CHECK(chat_history == expected_chat_history);
  }

  SUBCASE("valid payload but unexpected chat history with chunked responsens - split into two parts")
  {
    std::string payload{R"(
        {
          "messages": [
            {
              "role": "my_role_1",
              "content": "my_content_3"
            }
          ]
        }
        )"};

    const std::string chat_history_raw{R"(
        [
          {
            "role": "my_role_1",
            "content": "my_content_1"
          },
          {
            "role": "my_role_2",
            "content": "my_content_2-1",
            "function_call": null
          },
          {
            "role": null,
            "content": ":my_content_2-1:",
            "function_call": null
          },
          {
            "role": "unexpected_role_2",
            "content": ":my_content_2-2:",
            "function_call": null
          },
          {
            "role": null,
            "content": ":my_content_2-3:",
            "function_call": null
          }
        ]
        )"};
    rapidjson::Document chat_history{};
    rapidjson::ParseResult parse_ok = chat_history.Parse(chat_history_raw.c_str());
    CHECK(parse_ok);

    PayloadJsonUtils::UpdateHistoryAndAddToPayload(payload, chat_history);

    rapidjson::Document payload_document{};
    parse_ok = payload_document.Parse(payload.c_str());
    CHECK(parse_ok);

    const std::string expected_payload{R"(
        {
          "messages": [
            {
              "role": "my_role_1",
              "content": "my_content_1"
            },
            {
              "role": "my_role_2",
              "content": "my_content_2-1:my_content_2-1:",
              "function_call": null
            },
            {
              "role": "unexpected_role_2",
              "content": ":my_content_2-2::my_content_2-3:",
              "function_call": null
            },
            {
              "role": "my_role_1",
              "content": "my_content_3"
            }
          ]
        }
        )"};
    rapidjson::Document expected_payload_document{};
    parse_ok = expected_payload_document.Parse(expected_payload.c_str());
    CHECK(parse_ok);

    CHECK(payload_document == expected_payload_document);

    const std::string expected_chat_history_raw{R"(
        [
          {
            "role": "my_role_1",
            "content": "my_content_1"
          },
          {
            "role": "my_role_2",
            "content": "my_content_2-1",
            "function_call": null
          },
          {
            "role": null,
            "content": ":my_content_2-1:",
            "function_call": null
          },
          {
            "role": "unexpected_role_2",
            "content": ":my_content_2-2:",
            "function_call": null
          },
          {
            "role": null,
            "content": ":my_content_2-3:",
            "function_call": null
          },
          {
            "role": "my_role_1",
            "content": "my_content_3"
          }
        ]
        )"};
    rapidjson::Document expected_chat_history{};
    parse_ok = expected_chat_history.Parse(expected_chat_history_raw.c_str());
    CHECK(parse_ok);

    CHECK(chat_history == expected_chat_history);
  }
}

}  // namespace triton::perfanalyzer
