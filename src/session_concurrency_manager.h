// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <thread>

#include "load_manager.h"

namespace triton::perfanalyzer {

class SessionConcurrencyManager : public LoadManager {
 public:
  SessionConcurrencyManager(
      const bool async, const bool streaming, const int32_t batch_size,
      const size_t max_threads, const SharedMemoryType shared_memory_type,
      const size_t output_shm_size, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::unordered_map<std::string, cb::RequestParameter>&
          request_parameters,
      const size_t session_concurrency)
      : LoadManager(
            async, streaming, batch_size, max_threads, shared_memory_type,
            output_shm_size, parser, factory, request_parameters),
        session_concurrency_(session_concurrency)
  {
  }

  std::vector<Range<size_t>> SplitStreamIndexes()
  {
    std::vector<Range<size_t>> stream_ranges{};

    size_t num_streams{data_loader_->GetDataStreamsCount()};
    size_t num_sessions{session_concurrency_};

    if (num_streams < num_sessions) {
      throw std::runtime_error(
          "The input data file must contain at least as many streams as the "
          "requested session concurrency level. Please either add more streams "
          "to the input data file or reduce the session concurrency level.");
    }

    size_t base_size{num_streams / num_sessions};
    size_t remainder{num_streams % num_sessions};

    size_t start{0};
    for (size_t i{0}; i < num_sessions; ++i) {
      size_t end{start + base_size - 1};
      if (i < remainder) {
        ++end;
      }
      size_t step{1};
      stream_ranges.push_back({start, end, step});
      start = end + 1;
    }

    return stream_ranges;
  }

  void SendRequestsForStreams(Range<size_t> stream_range)
  {
    std::cout << std::this_thread::get_id() << " " << stream_range.start << " "
              << stream_range.end << std::endl;

    // each thread is responsible for keeping one session active at all times

    // each thread, for its current session, just has a loop that sends requests

    // when session is complete, it grabs next session (stream ID)
  }

  std::vector<std::thread> MakeThreads(std::vector<Range<size_t>> stream_ranges)
  {
    std::vector<std::thread> threads{};

    for (auto stream_range : stream_ranges) {
      threads.emplace_back(
          &SessionConcurrencyManager::SendRequestsForStreams, this,
          stream_range);
    }

    return threads;
  }

  std::vector<RequestRecord> Start()
  {
    // split up set of data_loader_ streams into session_concurrency_ portions
    auto stream_ranges{SplitStreamIndexes()};

    // make session_concurrency_ threads
    auto threads{MakeThreads(stream_ranges)};

    for (auto& thread : threads) {
      thread.join();
    }

    return GetRequestRecords();
  }

 private:
  cb::Error InitManagerInputs(
      const size_t string_length, const std::string& string_data,
      const bool zero_input, std::vector<std::string>& user_data) override
  {
    // modify parser_->Inputs() to include "delay" input
    auto& inputs_with_delay_added{parser_->Inputs()};

    std::string name{"delay"};
    std::string datatype{"UINT64"};
    std::vector<int64_t> shape{1};
    bool is_shape_tensor{false};
    bool is_optional{true};

    ModelTensor delay_model_tensor{
        name, datatype, shape, is_shape_tensor, is_optional};

    (*inputs_with_delay_added)["delay"] = delay_model_tensor;

    using_json_data_ = true;
    for (const auto& json_file : user_data) {
      RETURN_IF_ERROR(data_loader_->ReadDataFromJSON(
          inputs_with_delay_added, parser_->Outputs(), json_file));
    }
    std::cout << " Successfully read data for "
              << data_loader_->GetDataStreamsCount() << " stream/streams";
    if (data_loader_->GetDataStreamsCount() == 1) {
      std::cout << " with " << data_loader_->GetTotalSteps(0) << " step/steps";
    }
    std::cout << "." << std::endl;

    // Reserve the required vector space
    threads_stat_.reserve(max_threads_);

    return cb::Error::Success;
  }

  std::vector<RequestRecord> GetRequestRecords()
  {
    std::vector<RequestRecord> request_records{};
    for (const auto& thread_stat : threads_stat_) {
      request_records.insert(
          request_records.end(),
          std::make_move_iterator(thread_stat->request_records_.begin()),
          std::make_move_iterator(thread_stat->request_records_.end()));
    }
    return request_records;
  }

  size_t session_concurrency_{0};
};

}  // namespace triton::perfanalyzer
