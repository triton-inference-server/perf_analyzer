// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "client_backend/mock_client_backend.h"
#include "doctest.h"
#include "gmock/gmock.h"
#include "infer_context.h"
#include "mock_data_loader.h"
#include "mock_infer_context.h"
#include "mock_infer_data_manager.h"
#include "mock_sequence_manager.h"

namespace triton { namespace perfanalyzer {

namespace {

struct RawDataCallCounts {
  size_t input{0};
  size_t output{0};
};

class TestInferInput : public cb::InferInput {
 public:
  TestInferInput(
      std::shared_ptr<RawDataCallCounts> call_counts, std::vector<uint8_t> data)
      : InferInput(cb::BackendKind::TRITON, "INPUT0", "UINT8"),
        call_counts_(std::move(call_counts)), data_(std::move(data))
  {
  }

  const std::vector<int64_t>& Shape() const override { return shape_; }

  cb::Error RawData(const uint8_t** buf, size_t* byte_size) override
  {
    call_counts_->input++;
    *buf = data_.data();
    *byte_size = data_.size();
    return cb::Error::Success;
  }

 private:
  std::shared_ptr<RawDataCallCounts> call_counts_;
  std::vector<uint8_t> data_;
  const std::vector<int64_t> shape_{4};
};

class TestInferRequestedOutput : public cb::InferRequestedOutput {
 public:
  TestInferRequestedOutput()
      : InferRequestedOutput(cb::BackendKind::TRITON, "OUTPUT0", "UINT8")
  {
  }
};

class TestInferResult : public cb::InferResult {
 public:
  TestInferResult(
      std::string request_id, std::shared_ptr<RawDataCallCounts> call_counts,
      std::vector<uint8_t> data)
      : request_id_(std::move(request_id)),
        call_counts_(std::move(call_counts)), data_(std::move(data))
  {
  }

  cb::Error Id(std::string* id) const override
  {
    *id = request_id_;
    return cb::Error::Success;
  }

  cb::Error RequestStatus() const override { return cb::Error::Success; }

  cb::Error RawData(
      const std::string&, std::vector<uint8_t>& buf) const override
  {
    call_counts_->output++;
    buf = data_;
    return cb::Error::Success;
  }

  cb::Error IsFinalResponse(bool* is_final_response) const override
  {
    *is_final_response = true;
    return cb::Error::Success;
  }

  cb::Error IsNullResponse(bool* is_null_response) const override
  {
    *is_null_response = false;
    return cb::Error::Success;
  }

 private:
  std::string request_id_;
  std::shared_ptr<RawDataCallCounts> call_counts_;
  std::vector<uint8_t> data_;
};

class TestClientBackend : public cb::ClientBackend {
 public:
  TestClientBackend(
      std::shared_ptr<RawDataCallCounts> call_counts,
      std::vector<uint8_t> output_data)
      : ClientBackend(cb::BackendKind::TRITON),
        call_counts_(std::move(call_counts)),
        output_data_(std::move(output_data))
  {
  }

  cb::Error Infer(
      cb::InferResult** result, const cb::InferOptions& options,
      const std::vector<cb::InferInput*>&,
      const std::vector<const cb::InferRequestedOutput*>&) override
  {
    *result =
        new TestInferResult(options.request_id_, call_counts_, output_data_);
    return cb::Error::Success;
  }

  cb::Error AsyncInfer(
      cb::OnCompleteFn callback, const cb::InferOptions& options,
      const std::vector<cb::InferInput*>&,
      const std::vector<const cb::InferRequestedOutput*>&) override
  {
    callback_ = std::move(callback);
    request_id_ = options.request_id_;
    return cb::Error::Success;
  }

  cb::Error ClientInferStat(cb::InferStat*) override
  {
    return cb::Error::Success;
  }

  void CompleteAsyncRequest()
  {
    callback_(new TestInferResult(request_id_, call_counts_, output_data_));
  }

 private:
  std::shared_ptr<RawDataCallCounts> call_counts_;
  std::vector<uint8_t> output_data_;
  cb::OnCompleteFn callback_;
  std::string request_id_;
};

void
TestProfilePayloadCapture(bool async, bool capture_profile_data)
{
  MockInferContext infer_context{};
  infer_context.thread_stat_ = std::make_shared<ThreadStat>();
  infer_context.thread_stat_->contexts_stat_.emplace_back();
  infer_context.async_ = async;
  infer_context.streaming_ = false;
  infer_context.capture_profile_data_ = capture_profile_data;
  infer_context.infer_data_.options_ =
      std::make_unique<cb::InferOptions>("model");

  auto call_counts = std::make_shared<RawDataCallCounts>();
  const std::vector<uint8_t> input_data{1, 2, 3, 4};
  const std::vector<uint8_t> output_data{5, 6, 7, 8};
  auto* input = new TestInferInput(call_counts, input_data);
  infer_context.infer_data_.inputs_.push_back(input);
  infer_context.infer_data_.valid_inputs_.push_back(input);
  infer_context.infer_data_.outputs_.push_back(new TestInferRequestedOutput());

  auto backend = std::make_unique<TestClientBackend>(call_counts, output_data);
  auto* backend_ptr = backend.get();
  infer_context.infer_backend_ = std::move(backend);

  infer_context.SendRequest(1, false, 0);
  if (async) {
    backend_ptr->CompleteAsyncRequest();
  }

  REQUIRE(infer_context.thread_stat_->request_records_.size() == 1);
  const auto& record = infer_context.thread_stat_->request_records_.front();
  if (capture_profile_data) {
    CHECK(call_counts->input == 1);
    CHECK(call_counts->output == 1);
    REQUIRE(record.request_inputs_.size() == 1);
    REQUIRE(record.response_outputs_.size() == 1);
    const auto& captured_input = record.request_inputs_.front().at("INPUT0");
    const auto& captured_output =
        record.response_outputs_.front().at("OUTPUT0");
    CHECK(captured_input.data_ == input_data);
    CHECK(captured_input.size_ == input_data.size());
    CHECK(captured_output.data_ == output_data);
    CHECK(captured_output.size_ == output_data.size());
  } else {
    CHECK(call_counts->input == 0);
    CHECK(call_counts->output == 0);
    CHECK(record.request_inputs_.empty());
    CHECK(record.response_outputs_.empty());
  }
}

}  // namespace

/// Tests the round robin ordering of json input data
///
TEST_CASE("update_seq_json_data: testing the UpdateSeqJsonData function")
{
  std::shared_ptr<MockSequenceManager> mock_sequence_manager{
      std::make_shared<MockSequenceManager>()};

  EXPECT_CALL(
      *mock_sequence_manager, SetInferSequenceOptions(testing::_, testing::_))
      .Times(6)
      .WillRepeatedly(testing::Return());

  mock_sequence_manager->InitSequenceStatuses(1);

  std::shared_ptr<MockDataLoader> mock_data_loader{
      std::make_shared<MockDataLoader>()};

  EXPECT_CALL(*mock_data_loader, GetTotalSteps(testing::_))
      .Times(6)
      .WillRepeatedly(testing::Return(3));

  std::shared_ptr<MockInferDataManager> mock_infer_data_manager{
      std::make_shared<MockInferDataManager>()};

  testing::Sequence seq;
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 0, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 1, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 2, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 0, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 1, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));
  EXPECT_CALL(
      *mock_infer_data_manager,
      UpdateInferData(testing::_, testing::_, 2, testing::_))
      .InSequence(seq)
      .WillOnce(testing::Return(cb::Error::Success));

  std::shared_ptr<MockInferContext> mic{std::make_shared<MockInferContext>()};

  EXPECT_CALL(*mic, SendRequest(testing::_, testing::_, testing::_))
      .Times(6)
      .WillRepeatedly(testing::Return());

  mic->sequence_manager_ = mock_sequence_manager;
  mic->data_loader_ = mock_data_loader;
  mic->infer_data_manager_ = mock_infer_data_manager;
  mic->thread_stat_ = std::make_shared<ThreadStat>();
  bool execute{true};
  mic->execute_ = execute;
  mic->using_json_data_ = true;

  size_t seq_stat_index{0};
  bool delayed{false};

  mic->SendSequenceInferRequest(seq_stat_index, delayed);
  mic->SendSequenceInferRequest(seq_stat_index, delayed);
  mic->SendSequenceInferRequest(seq_stat_index, delayed);
  mic->SendSequenceInferRequest(seq_stat_index, delayed);
  mic->SendSequenceInferRequest(seq_stat_index, delayed);
  mic->SendSequenceInferRequest(seq_stat_index, delayed);

  // Destruct gmock objects to determine gmock-related test failure
  mock_sequence_manager.reset();
  mock_data_loader.reset();
  mock_infer_data_manager.reset();
  mic.reset();
  REQUIRE(testing::Test::HasFailure() == false);
}

TEST_CASE("send_request: testing the SendRequest function")
{
  MockInferContext mock_infer_context{};

  SUBCASE("testing logic relevant to request record sequence ID")
  {
    mock_infer_context.thread_stat_ = std::make_shared<ThreadStat>();
    mock_infer_context.thread_stat_->contexts_stat_.emplace_back();
    mock_infer_context.async_ = true;
    mock_infer_context.streaming_ = true;
    mock_infer_context.infer_data_.options_ =
        std::make_unique<cb::InferOptions>("my_model");
    std::shared_ptr<cb::MockClientStats> mock_client_stats{
        std::make_shared<cb::MockClientStats>()};
    mock_infer_context.infer_backend_ =
        std::make_unique<cb::MockClientBackend>(mock_client_stats);

    const uint64_t request_id{5};
    const bool delayed{false};
    const uint64_t sequence_id{2};

    mock_infer_context.infer_data_.options_->request_id_ =
        std::to_string(request_id);

    cb::MockInferResult* mock_infer_result{
        new cb::MockInferResult(*mock_infer_context.infer_data_.options_)};

    cb::OnCompleteFn& stream_callback{mock_infer_context.async_callback_func_};

    EXPECT_CALL(
        dynamic_cast<cb::MockClientBackend&>(
            *mock_infer_context.infer_backend_),
        AsyncStreamInfer(testing::_, testing::_, testing::_))
        .WillOnce(
            [&mock_infer_result, &stream_callback](
                const cb::InferOptions& options,
                const std::vector<cb::InferInput*>& inputs,
                const std::vector<const cb::InferRequestedOutput*>& outputs)
                -> cb::Error {
              stream_callback(mock_infer_result);
              return cb::Error::Success;
            });

    mock_infer_context.SendRequest(request_id, delayed, sequence_id);

    CHECK(mock_infer_context.thread_stat_->request_records_.size() == 1);
    CHECK(
        mock_infer_context.thread_stat_->request_records_[0].sequence_id_ ==
        sequence_id);
  }
}

TEST_CASE("send_request: profile payload capture is opt-in")
{
  SUBCASE("synchronous request without profile export")
  {
    TestProfilePayloadCapture(false, false);
  }
  SUBCASE("synchronous request with profile export")
  {
    TestProfilePayloadCapture(false, true);
  }
  SUBCASE("asynchronous request without profile export")
  {
    TestProfilePayloadCapture(true, false);
  }
  SUBCASE("asynchronous request with profile export")
  {
    TestProfilePayloadCapture(true, true);
  }
}

}}  // namespace triton::perfanalyzer
