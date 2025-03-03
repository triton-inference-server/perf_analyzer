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

#include <stddef.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../client_backend/client_backend.h"
#include "../load_manager.h"
#include "../model_parser.h"
#include "../perf_utils.h"
#include "../request_record.h"
#include "payload_dataset_manager.h"
#include "request_handler.h"

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
      const size_t session_concurrency);

  std::vector<RequestRecord> Start();

 private:
  void MakeAndWaitForThreads(
      const std::vector<std::vector<size_t>>& all_session_payloads);

  void ProcessSessionsUntilComplete(
      const std::vector<std::vector<size_t>>& all_session_payloads,
      std::vector<RequestRecord>& request_records);

  void SendSequentialRequestsForOneSession(
      const std::vector<size_t>& session_payloads,
      std::vector<RequestRecord>& request_records);

  void GetAndWaitForDelay(size_t dataset_index) const;

  std::vector<RequestRecord> GetRequestRecords() const;

  const size_t session_concurrency_{};
  std::atomic<size_t> next_session_index_{};
  std::shared_ptr<PayloadDatasetManager> payload_dataset_manager_{};
  std::shared_ptr<RequestHandler> request_handler_{};
  std::vector<std::vector<RequestRecord>> all_threads_request_records_{};
};

}  // namespace triton::perfanalyzer
