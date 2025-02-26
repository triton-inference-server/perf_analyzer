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

#include <stddef.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "client_backend/client_backend.h"
#include "doctest.h"
#include "mock_data_loader.h"
#include "model_parser.h"
#include "session_concurrency/payload_dataset_manager.h"

namespace triton::perfanalyzer {

TEST_CASE("PayloadDatasetManager::GroupPayloadsBySession")
{
  SUBCASE("valid input data")
  {
    const std::string name{"session_id"};
    const std::string datatype{"BYTES"};
    const std::vector<int64_t> shape{{1}};
    const bool is_shape_tensor{false};
    const bool is_optional{false};

    const ModelTensor session_id_model_tensor{
        name, datatype, shape, is_shape_tensor, is_optional};

    auto model_parser{std::make_shared<ModelParser>()};

    (*model_parser->Inputs())[name] = session_id_model_tensor;

    const std::string input_data_json{R"(
        {
          "data": [
            {
              "session_id": ["my_session_id_1"]
            },
            {
              "session_id": ["my_session_id_2"]
            },
            {
              "session_id": ["my_session_id_1"]
            },
            {
              "session_id": ["my_session_id_2"]
            }
          ]
        }
        )"};

    auto data_loader{std::make_shared<MockDataLoader>()};

    const auto error{data_loader->ReadDataFromJSON(
        model_parser->Inputs(), model_parser->Outputs(), input_data_json)};

    REQUIRE(error.IsOk());

    const PayloadDatasetManager payload_dataset_manager(
        data_loader, model_parser);

    const auto all_session_payloads{
        payload_dataset_manager.GroupPayloadsBySession()};

    REQUIRE(all_session_payloads.size() == 2);
    CHECK(all_session_payloads[0].size() == 2);
    CHECK(all_session_payloads[1].size() == 2);

    const std::set<size_t> session_set_1(
        all_session_payloads[0].begin(), all_session_payloads[0].end());
    const std::set<size_t> session_set_2(
        all_session_payloads[1].begin(), all_session_payloads[1].end());

    const std::set<std::set<size_t>> session_sets{session_set_1, session_set_2};

    const std::set<std::set<size_t>> expected_session_sets{{0, 2}, {1, 3}};

    CHECK(session_sets == expected_session_sets);
  }

  SUBCASE("invalid input data - multiple streams")
  {
    auto data_loader{std::make_shared<MockDataLoader>()};

    const std::shared_ptr<ModelTensorMap> inputs{};
    const std::shared_ptr<ModelTensorMap> outputs{};

    const std::string input_data_json{R"(
        {"data": [[],[]]}
        )"};

    const auto error{
        data_loader->ReadDataFromJSON(inputs, outputs, input_data_json)};

    REQUIRE(error.IsOk());

    const PayloadDatasetManager payload_dataset_manager(data_loader, nullptr);

    CHECK_THROWS_WITH_AS(
        payload_dataset_manager.GroupPayloadsBySession(),
        "Expected input data JSON to have one stream. Session concurrency mode "
        "must have an input data JSON with a single flat array for the "
        "\"data\" field with one element per request payload.",
        std::runtime_error);
  }
}

TEST_CASE("PayloadDatasetManager::GetDelayForPayload")
{
  const std::string name{"delay"};
  const std::string datatype{"UINT64"};
  const std::vector<int64_t> shape{{1}};
  const bool is_shape_tensor{false};
  const bool is_optional{true};

  const ModelTensor delay_model_tensor{
      name, datatype, shape, is_shape_tensor, is_optional};

  auto model_parser{std::make_shared<ModelParser>()};

  (*model_parser->Inputs())[name] = delay_model_tensor;

  SUBCASE("valid delay")
  {
    const std::string input_data_json{R"(
      {
        "data": [
          {
            "delay": [1000]
          }
        ]
      }
      )"};

    auto data_loader{std::make_shared<MockDataLoader>()};

    const auto error{data_loader->ReadDataFromJSON(
        model_parser->Inputs(), model_parser->Outputs(), input_data_json)};

    REQUIRE(error.IsOk());

    const PayloadDatasetManager payload_dataset_manager(
        data_loader, model_parser);

    const size_t dataset_index{0};

    const auto delay{payload_dataset_manager.GetDelayForPayload(dataset_index)};

    CHECK(delay == std::chrono::milliseconds(1000));
  }

  SUBCASE("missing delay")
  {
    const std::string input_data_json{R"(
      {"data": [{}]}
      )"};

    auto data_loader{std::make_shared<MockDataLoader>()};

    const auto error{data_loader->ReadDataFromJSON(
        model_parser->Inputs(), model_parser->Outputs(), input_data_json)};

    REQUIRE(error.IsOk());

    const PayloadDatasetManager payload_dataset_manager(
        data_loader, model_parser);

    const size_t dataset_index{0};

    CHECK_THROWS_WITH_AS(
        payload_dataset_manager.GetDelayForPayload(dataset_index),
        "Missing 'delay' input for payload with index 0", std::runtime_error);
  }
}

TEST_CASE("PayloadDatasetManager::GetPayload")
{
  const std::string name{"payload"};
  const std::string datatype{"JSON"};
  const std::vector<int64_t> shape{{1}};
  const bool is_shape_tensor{false};
  const bool is_optional{false};

  const ModelTensor payload_model_tensor{
      name, datatype, shape, is_shape_tensor, is_optional};

  auto model_parser{std::make_shared<ModelParser>()};

  (*model_parser->Inputs())[name] = payload_model_tensor;

  const std::string input_data_json{R"(
      {
        "data": [
          {
            "payload": [{}]
          }
        ]
      }
      )"};

  auto data_loader{std::make_shared<MockDataLoader>()};

  const auto error{data_loader->ReadDataFromJSON(
      model_parser->Inputs(), model_parser->Outputs(), input_data_json)};

  REQUIRE(error.IsOk());

  const PayloadDatasetManager payload_dataset_manager(
      data_loader, model_parser);

  const size_t dataset_index{0};

  const auto payload{payload_dataset_manager.GetPayload(dataset_index)};

  CHECK(payload == "{}");
}

}  // namespace triton::perfanalyzer
