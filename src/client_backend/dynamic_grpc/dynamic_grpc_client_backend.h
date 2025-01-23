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

#include <string>

#include "../../perf_utils.h"
#include "../client_backend.h"
#include "dynamic_grpc_client.h"
#include "dynamic_grpc_infer_input.h"

#define RETURN_IF_TRITON_ERROR(S)       \
  do {                                  \
    const tc::Error& status__ = (S);    \
    if (!status__.IsOk()) {             \
      return Error(status__.Message()); \
    }                                   \
  } while (false)

namespace tc = triton::client;
namespace cb = triton::perfanalyzer::clientbackend;

namespace triton::perfanalyzer::clientbackend::dynamicgrpc {

//==============================================================================
//
class DynamicGrpcClientBackend : public ClientBackend {
 public:
  /// Create a gRPC client backend which can be used to interact with an
  /// arbitrary grpc service.
  /// \param url The grpc service url and port.
  /// \param protocol The protocol type used.
  /// \param ssl_options The SSL options used with client backend.
  /// \param compression_algorithm The compression algorithm to be used
  /// on the grpc requests.
  /// \param http_headers Map of HTTP headers. The map key/value indicates
  /// the header name/value.
  /// \param verbose Enables the verbose mode.
  /// \param client_backend Returns a new DynamicGrpcClientBackend
  /// object.
  /// \return Error object indicating success or failure.
  static Error Create(
      const std::string& url, const ProtocolType protocol,
      const SslOptionsBase& ssl_options,
      const grpc_compression_algorithm compression_algorithm,
      std::shared_ptr<Headers> http_headers, const std::string& proto_file,
      const std::string& grpc_method, const bool verbose,
      std::unique_ptr<ClientBackend>* client_backend);

  /// See ClientBackend::StreamInfer()
  Error StreamInfer(
      cb::InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs);

  /// See ClientBackend::StartStream()
  Error StartStream(OnCompleteFn callback, bool enable_stats) override;

  /// See ClientBackend::ClientInferStat()
  Error ClientInferStat(InferStat* infer_stat) override;

 private:
  DynamicGrpcClientBackend(
      const grpc_compression_algorithm compression_algorithm,
      std::shared_ptr<Headers> http_headers)
      : ClientBackend(BackendKind::DYNAMIC_GRPC),
        compression_algorithm_(compression_algorithm),
        http_headers_(http_headers)
  {
  }

  std::unique_ptr<DynamicGrpcClient> grpc_client_;

  grpc_compression_algorithm compression_algorithm_;
  std::shared_ptr<Headers> http_headers_;
};

//==============================================================
/// DynamicGrpcInferRequestedOutput is a wrapper around
/// InferRequestedOutput object of triton common client library.
///
class DynamicGrpcInferRequestedOutput : public InferRequestedOutput {
 public:
  static Error Create(
      InferRequestedOutput** infer_output, const std::string& name);
  /// Returns the raw InferRequestedOutput object required by gRPC client
  /// library.
  tc::InferRequestedOutput* Get() const { return output_.get(); }

 private:
  explicit DynamicGrpcInferRequestedOutput(const std::string& name);

  std::unique_ptr<tc::InferRequestedOutput> output_;
};

}  // namespace triton::perfanalyzer::clientbackend::dynamicgrpc
