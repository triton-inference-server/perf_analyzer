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

#include "load_manager.h"

namespace triton::perfanalyzer {

void
SessionConcurrencyWorkerLoop(std::shared_ptr<cb::ClientBackendFactory> factory)
{
  std::unique_ptr<cb::ClientBackend> client_backend{nullptr};
  factory->CreateClientBackend(&client_backend);

  while (1) {
    // client_backend->AsyncInfer();
  }
}

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

  std::vector<RequestRecord> Start()
  {
    LoopThroughDataset();
    return GetRequestRecords();
    size_t active_session_count{0};
    while (active_session_count < session_concurrency_) {
      std::thread session_concurrency_worker(
          SessionConcurrencyWorkerLoop, factory_);
    }
    return {};
  }

 private:
  cb::Error InitManagerInputs(
      const size_t string_length, const std::string& string_data,
      const bool zero_input, std::vector<std::string>& user_data) override
  {
    // modify parser_->Inputs() to include "delay" input
    using_json_data_ = true;
    for (const auto& json_file : user_data) {
      RETURN_IF_ERROR(data_loader_->ReadDataFromJSON(
          parser_->Inputs(), parser_->Outputs(), json_file));
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
