// Copyright 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>

#include "client_backend/client_backend.h"
#include "constants.h"
#include "doctest.h"
#include "mock_client_backend.h"
#include "mock_model_parser.h"

namespace cb = triton::perfanalyzer::clientbackend;

namespace triton { namespace perfanalyzer {

class TestModelParser {
 public:
  constexpr static const char* no_batching =
      R"({ "name": "NoBatchingModel", "platform":"not_ensemble" })";

  constexpr static const char* seq_batching =
      R"({ "name": "SeqBatchingModel", "platform":"not_ensemble", "sequence_batching":{} })";

  constexpr static const char* dyn_batching =
      R"({ "name": "DynBatchingModel", "platform":"not_ensemble", "dynamic_batching":{} })";

  constexpr static const char* ensemble = R"({
    "name": "EnsembleModel",
    "platform": "ensemble",
    "ensemble_scheduling": {
      "step": [{
          "model_name": "ModelA",
          "model_version": 2
        },
        {
          "model_name": "ModelB",
          "model_version": -1
        }
      ]
    }
  })";

  constexpr static const char* nested_ensemble = R"({
    "name": "ModelA",
    "platform": "ensemble",
    "ensemble_scheduling": {
      "step": [{
          "model_name": "ModelC",
          "model_version": -1
        },
        {
          "model_name": "ModelD",
          "model_version": -1
        }
      ]
    }
  })";

  static cb::Error SetJsonPtrNoSeq(rapidjson::Document* model_config)
  {
    model_config->Parse(no_batching);
    return cb::Error::Success;
  };

  static cb::Error SetJsonPtrYesSeq(rapidjson::Document* model_config)
  {
    model_config->Parse(seq_batching);
    return cb::Error::Success;
  };

  static cb::Error SetJsonPtrNestedEnsemble(rapidjson::Document* model_config)
  {
    model_config->Parse(nested_ensemble);
    return cb::Error::Success;
  };
};

TEST_CASE("ModelParser: test Init* functions")
{
  MockModelParser mmp;

  std::string model_name{"some_model"};
  std::string model_version{"1234"};
  int32_t batch_size{123};

  std::vector<std::string> expected_input_names;
  std::vector<std::string> expected_output_names;
  cb::Error status;

  SUBCASE("InitTriton")
  {
    rapidjson::Document metadata;
    std::string metadata_json{R"(
    {
      "name": "some_model",
      "inputs": [
        {
          "name": "INPUT0",
          "datatype": "UINT8",
          "shape": [1]
        }
      ],
      "outputs": [
        {
          "name": "OUTPUT0",
          "datatype": "UINT8",
          "shape": [1]
        }
      ]
    }
    )"};
    metadata.Parse(metadata_json.c_str());

    rapidjson::Document config;
    std::string config_json{R"(
    {
      "name": "some_model",
      "max_batch_size": 123,
      "model_transaction_policy": {
        "decoupled": false
      },
      "scheduling_choice": {
        "dynamic_batching": {
          "max_queue_delay_microseconds": 100
        }
      },
      "input": [
        {
          "name": "INPUT0",
          "data_type": "UINT8",
          "dims": [1]
        }
      ]
    }
    )"};
    config.Parse(config_json.c_str());

    std::vector<cb::ModelIdentifier> bls_composing_models;
    std::unordered_map<std::string, std::vector<int64_t>> input_shapes;
    std::unique_ptr<cb::ClientBackend> backend;

    status = mmp.InitTriton(
        metadata, config, model_version, bls_composing_models, input_shapes,
        backend);
    expected_input_names.push_back("INPUT0");
    expected_output_names.push_back("OUTPUT0");
  }

  SUBCASE("InitDynamicGrpc")
  {
    status = mmp.InitDynamicGrpc(model_name, model_version, batch_size);
    expected_input_names.push_back("message_generator");
    expected_output_names.push_back("response");
  }

  SUBCASE("InitOpenAI")
  {
    status = mmp.InitOpenAI(model_name, model_version, batch_size);
    expected_input_names.push_back("payload");
    expected_output_names.push_back("response");
  }

  SUBCASE("InitTorchServe")
  {
    status = mmp.InitTorchServe(model_name, model_version, batch_size);
    expected_input_names.push_back("TORCHSERVE_INPUT");
  }

  CHECK(status.IsOk());
  CHECK(mmp.ModelName() == model_name);
  CHECK(mmp.ModelVersion() == model_version);
  CHECK(mmp.MaxBatchSize() == batch_size);

  auto actual_inputs = mmp.Inputs();
  for (const auto& name : expected_input_names) {
    CHECK(actual_inputs->find(name) != actual_inputs->end());
  }

  auto actual_outputs = mmp.Outputs();
  for (const auto& name : expected_output_names) {
    CHECK(actual_outputs->find(name) != actual_outputs->end());
  }
}

TEST_CASE("ModelParser: testing the GetInt function")
{
  int64_t integer_value{0};
  MockModelParser mmp;

  SUBCASE("valid string")
  {
    rapidjson::Value value("100");
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == SUCCESS);
    CHECK(integer_value == 100);
  }

  SUBCASE("invalid string, alphabet")
  {
    rapidjson::Value value("abc");
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(result.Message() == "unable to convert 'abc' to integer");
    CHECK(integer_value == 0);
  }

  SUBCASE("invalid string, number out of range")
  {
    rapidjson::Value value("9223372036854775808");
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(
        result.Message() ==
        "unable to convert '9223372036854775808' to integer");
    CHECK(integer_value == 0);
  }

  SUBCASE("valid int, lowest Int64")
  {
    rapidjson::Value value(2147483648);
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == SUCCESS);
    CHECK(integer_value == 2147483648);
  }

  SUBCASE("valid int, highest Int32")
  {
    rapidjson::Value value(2147483647);
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == SUCCESS);
    CHECK(integer_value == 2147483647);
  }

  SUBCASE("invalid floating point")
  {
    rapidjson::Value value(100.1);
    cb::Error result{mmp.GetInt(value, &integer_value)};
    CHECK(result.Err() == GENERIC_ERROR);
    CHECK(result.Message() == "failed to parse the integer value");
    CHECK(integer_value == 0);
  }
}

TEST_CASE(
    "ModelParser: DetermineComposingModelMap" *
    doctest::description(
        "This test confirms that the composing model map will be correctly "
        "populated by DetermineComposingModelMap()"))
{
  std::shared_ptr<cb::MockClientStats> stats =
      std::make_shared<cb::MockClientStats>();
  std::unique_ptr<cb::MockClientBackend> mock_backend =
      std::make_unique<cb::MockClientBackend>(stats);

  rapidjson::Document config;
  std::vector<cb::ModelIdentifier> input_bls_composing_models;
  ComposingModelMap expected_composing_model_map;

  std::string parent_model_name;


  const auto& ParameterizeListedComposingModels{[&]() {
    SUBCASE("No listed composing models") {}
    SUBCASE("Yes listed composing models")
    {
      input_bls_composing_models.push_back({"ListedModelA", ""});
      input_bls_composing_models.push_back({"ListedModelB", ""});
      expected_composing_model_map[parent_model_name].emplace(
          "ListedModelA", "");
      expected_composing_model_map[parent_model_name].emplace(
          "ListedModelB", "");
    }
    EXPECT_CALL(*mock_backend, ModelConfig(testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq));
  }};

  SUBCASE("No Ensemble")
  {
    config.Parse(TestModelParser::no_batching);
    parent_model_name = "NoBatchingModel";
    ParameterizeListedComposingModels();
  }
  SUBCASE("Ensemble")
  {
    config.Parse(TestModelParser::ensemble);
    parent_model_name = "EnsembleModel";
    ParameterizeListedComposingModels();

    expected_composing_model_map["EnsembleModel"].emplace("ModelA", "2");
    expected_composing_model_map["EnsembleModel"].emplace("ModelB", "");
    EXPECT_CALL(*mock_backend, ModelConfig(testing::_, testing::_, testing::_))
        .WillRepeatedly(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq));
  }
  SUBCASE("Nested Ensemble")
  {
    config.Parse(TestModelParser::ensemble);
    parent_model_name = "EnsembleModel";
    ParameterizeListedComposingModels();

    expected_composing_model_map["EnsembleModel"].emplace("ModelA", "2");
    expected_composing_model_map["EnsembleModel"].emplace("ModelB", "");
    expected_composing_model_map["ModelA"].emplace("ModelC", "");
    expected_composing_model_map["ModelA"].emplace("ModelD", "");
    EXPECT_CALL(*mock_backend, ModelConfig(testing::_, testing::_, testing::_))
        .WillOnce(
            testing::WithArg<0>(TestModelParser::SetJsonPtrNestedEnsemble))
        .WillRepeatedly(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq));
  }
  SUBCASE("BLS with an Ensemble")
  {
    config.Parse(TestModelParser::no_batching);
    parent_model_name = "NoBatchingModel";

    input_bls_composing_models.push_back({"ModelA", ""});
    input_bls_composing_models.push_back({"ModelB", ""});

    expected_composing_model_map[parent_model_name].emplace("ModelA", "");
    expected_composing_model_map[parent_model_name].emplace("ModelB", "");
    expected_composing_model_map["ModelA"].emplace("ModelC", "");
    expected_composing_model_map["ModelA"].emplace("ModelD", "");
    EXPECT_CALL(*mock_backend, ModelConfig(testing::_, testing::_, testing::_))
        .WillOnce(
            testing::WithArg<0>(TestModelParser::SetJsonPtrNestedEnsemble))
        .WillRepeatedly(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq));
  }

  std::unique_ptr<cb::ClientBackend> backend = std::move(mock_backend);

  MockModelParser mmp;

  mmp.DetermineComposingModelMap(input_bls_composing_models, config, backend);

  auto actual_composing_model_map = *mmp.GetComposingModelMap().get();
  CHECK(actual_composing_model_map == expected_composing_model_map);

  // Destruct gmock objects to determine gmock-related test failure
  backend.reset();
}

TEST_CASE(
    "ModelParser: determining scheduler type" *
    doctest::description("This test confirms that scheduler_type_ will be set "
                         "correctly by DetermineSchedulerType()"))
{
  std::shared_ptr<cb::MockClientStats> stats =
      std::make_shared<cb::MockClientStats>();
  std::unique_ptr<cb::MockClientBackend> mock_backend =
      std::make_unique<cb::MockClientBackend>(stats);


  rapidjson::Document config;
  ModelParser::ModelSchedulerType expected_type;

  ComposingModelMap input_composing_model_map;


  SUBCASE("No batching")
  {
    config.Parse(TestModelParser::no_batching);
    expected_type = ModelParser::ModelSchedulerType::NONE;
  }
  SUBCASE("Sequence batching")
  {
    config.Parse(TestModelParser::seq_batching);
    expected_type = ModelParser::ModelSchedulerType::SEQUENCE;
  }
  SUBCASE("Dynamic batching")
  {
    config.Parse(TestModelParser::dyn_batching);
    expected_type = ModelParser::ModelSchedulerType::DYNAMIC;
  }
  SUBCASE("Ensemble")
  {
    config.Parse(TestModelParser::ensemble);

    input_composing_model_map["EnsembleModel"].emplace("ModelA", "2");
    input_composing_model_map["EnsembleModel"].emplace("ModelB", "");

    SUBCASE("no sequences")
    {
      EXPECT_CALL(
          *mock_backend, ModelConfig(testing::_, testing::_, testing::_))
          .WillOnce(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq));

      expected_type = ModelParser::ModelSchedulerType::ENSEMBLE;
    }
    SUBCASE("yes sequences")
    {
      EXPECT_CALL(
          *mock_backend, ModelConfig(testing::_, testing::_, testing::_))
          .WillOnce(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(TestModelParser::SetJsonPtrYesSeq));

      expected_type = ModelParser::ModelSchedulerType::ENSEMBLE_SEQUENCE;
    }
  }
  SUBCASE("Nested Ensemble")
  {
    config.Parse(TestModelParser::ensemble);

    input_composing_model_map["EnsembleModel"].emplace("ModelA", "2");
    input_composing_model_map["EnsembleModel"].emplace("ModelB", "");
    input_composing_model_map["ModelA"].emplace("ModelC", "");
    input_composing_model_map["ModelA"].emplace("ModelD", "");

    SUBCASE("no sequences")
    {
      EXPECT_CALL(
          *mock_backend, ModelConfig(testing::_, testing::_, testing::_))
          .WillOnce(
              testing::WithArg<0>(TestModelParser::SetJsonPtrNestedEnsemble))
          .WillOnce(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq));

      expected_type = ModelParser::ModelSchedulerType::ENSEMBLE;
    }
    SUBCASE("yes sequences")
    {
      EXPECT_CALL(
          *mock_backend, ModelConfig(testing::_, testing::_, testing::_))
          .WillOnce(
              testing::WithArg<0>(TestModelParser::SetJsonPtrNestedEnsemble))
          .WillOnce(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq))
          .WillOnce(testing::WithArg<0>(TestModelParser::SetJsonPtrYesSeq))
          .WillOnce(testing::WithArg<0>(TestModelParser::SetJsonPtrNoSeq));

      expected_type = ModelParser::ModelSchedulerType::ENSEMBLE_SEQUENCE;
    }
  }

  std::unique_ptr<cb::ClientBackend> backend = std::move(mock_backend);

  MockModelParser mmp;
  mmp.composing_models_map_ =
      std::make_shared<ComposingModelMap>(input_composing_model_map);
  mmp.DetermineSchedulerType(config, backend);

  auto actual_type = mmp.SchedulerType();
  CHECK(actual_type == expected_type);

  // Destruct gmock objects to determine gmock-related test failure
  backend.reset();
}

}}  // namespace triton::perfanalyzer
