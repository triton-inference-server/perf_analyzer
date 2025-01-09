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
#include <stddef.h>

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "../client_backend/client_backend.h"
#include "../model_parser.h"
#include "payload_dataset_manager.h"

namespace triton::perfanalyzer {

class RequestHandler {
 public:
  RequestHandler(
      std::shared_ptr<cb::ClientBackendFactory> factory,
      const std::shared_ptr<ModelParser> parser,
      std::shared_ptr<PayloadDatasetManager> payload_dataset_manager);

  void SendRequestAndWaitForResponse(
      size_t dataset_index, rapidjson::Document& chat_history);

 private:
  void SendRequest(
      const std::string& payload,
      std::shared_ptr<std::promise<void>>&& response_promise,
      rapidjson::Document& chat_history);

  const std::function<void(cb::InferResult*)> PrepareCallback(
      std::shared_ptr<std::promise<void>>&& response_promise,
      rapidjson::Document& chat_history) const;

  const std::vector<uint8_t> GetResponseBuffer(
      cb::InferResult* infer_result) const;

  const std::vector<cb::InferInput*> PrepareInputs(
      const std::string& payload) const;

  const std::vector<const cb::InferRequestedOutput*> PrepareOutputs() const;

  void WaitForResponse(std::future<void>&& response_future) const;

  std::unique_ptr<cb::ClientBackend> client_backend_{};
  std::shared_ptr<cb::ClientBackendFactory> factory_{};
  const std::shared_ptr<ModelParser> parser_{};
  std::shared_ptr<PayloadDatasetManager> payload_dataset_manager_{};
};

}  // namespace triton::perfanalyzer
