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

#include "session_concurrency_manager.h"

#include <rapidjson/document.h>
#include <rapidjson/rapidjson.h>
#include <stddef.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <utility>

#include "../client_backend/client_backend.h"
#include "../load_manager.h"
#include "../model_parser.h"
#include "../perf_utils.h"
#include "../request_record.h"
#include "payload_dataset_manager.h"
#include "request_handler.h"

namespace triton::perfanalyzer {

SessionConcurrencyManager::SessionConcurrencyManager(
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
      session_concurrency_(session_concurrency),
      payload_dataset_manager_(
          std::make_shared<PayloadDatasetManager>(data_loader_, parser)),
      request_handler_(std::make_shared<RequestHandler>(
          factory, parser, payload_dataset_manager_))
{
}

std::vector<RequestRecord>
SessionConcurrencyManager::Start()
{
  const auto all_session_payloads{
      payload_dataset_manager_->GroupPayloadsBySession()};
  MakeAndWaitForThreads(all_session_payloads);
  return GetRequestRecords();
}

void
SessionConcurrencyManager::MakeAndWaitForThreads(
    const std::vector<std::vector<size_t>>& all_session_payloads)
{
  if (!threads_.empty()) {
    throw std::runtime_error("Expected threads_ to be empty.");
  } else if (!all_threads_request_records_.empty()) {
    throw std::runtime_error(
        "Expected all_threads_request_records_ to be empty.");
  } else if (all_session_payloads.size() < session_concurrency_) {
    throw std::runtime_error(
        "The input data file contains " +
        std::to_string(all_session_payloads.size()) +
        " unique session IDs, which is fewer than the required session "
        "concurrency level of " +
        std::to_string(session_concurrency_) +
        ". Please add more unique session IDs to the file or lower the "
        "session concurrency level.");
  }

  all_threads_request_records_.resize(session_concurrency_);

  for (size_t i{0}; i < session_concurrency_; ++i) {
    auto& request_records{all_threads_request_records_[i]};

    threads_.emplace_back(
        &SessionConcurrencyManager::ProcessSessionsUntilComplete, this,
        std::cref(all_session_payloads), std::ref(request_records));
  }

  for (auto& thread : threads_) {
    thread.join();
  }
}

void
SessionConcurrencyManager::ProcessSessionsUntilComplete(
    const std::vector<std::vector<size_t>>& all_session_payloads,
    std::vector<RequestRecord>& request_records)
{
  while (true) {
    const size_t session_index{next_session_index_++};

    if (session_index >= all_session_payloads.size()) {
      break;
    }

    const auto& one_session_payloads{all_session_payloads[session_index]};

    SendSequentialRequestsForOneSession(one_session_payloads, request_records);
  }
}

void
SessionConcurrencyManager::SendSequentialRequestsForOneSession(
    const std::vector<size_t>& one_session_payloads,
    std::vector<RequestRecord>& request_records)
{
  rapidjson::Document chat_history(rapidjson::kArrayType);
  // This vector contains pairs representing head and tail index of
  // a single payload or a group of chunks corresponding to one request payload.
  // (exclusive at tail)
  // Example:
  // one_session_chunk_ranges[0] = (0, 1)
  //   --> chat_history[0] - chat_history[0] is a single request payload
  // one_session_chunk_ranges[1] = (1, 10)
  //   --> chat_history[1] - chat_history[9] is a range of payload chunks for the first request
  // one_session_chunk_ranges[2] = (10, 11)
  //   --> chat_history[10] - chat_history[10] is a single request payload
  // one_session_chunk_ranges[3] = (11, 18)
  //   --> chat_history[11] - chat_history[17] is a range of payload chunks for the second request
  std::vector<std::pair<size_t, size_t>> one_session_chunk_ranges;

  for (size_t i{0}; i < one_session_payloads.size(); ++i) {
    const size_t payload_dataset_index{one_session_payloads[i]};

    auto& request_record{request_records.emplace_back()};

    request_handler_->SendRequestAndWaitForResponse(
        payload_dataset_index, chat_history, one_session_chunk_ranges, request_record);

    const bool is_last_request{i == one_session_payloads.size() - 1};

    if (is_last_request) {
      break;
    }

    GetAndWaitForDelay(payload_dataset_index);
  }
}

void
SessionConcurrencyManager::GetAndWaitForDelay(
    size_t payload_dataset_index) const
{
  const auto delay{payload_dataset_manager_->GetDelay(payload_dataset_index)};

  if (!delay) {
    throw std::runtime_error(
        "Missing 'delay' input for payload with index " +
        std::to_string(payload_dataset_index));
  }

  std::this_thread::sleep_for(*delay);
}

std::vector<RequestRecord>
SessionConcurrencyManager::GetRequestRecords() const
{
  std::vector<RequestRecord> request_records{};
  for (const auto& one_thread_request_records : all_threads_request_records_) {
    request_records.insert(
        request_records.end(),
        std::make_move_iterator(one_thread_request_records.begin()),
        std::make_move_iterator(one_thread_request_records.end()));
  }
  return request_records;
}

}  // namespace triton::perfanalyzer
