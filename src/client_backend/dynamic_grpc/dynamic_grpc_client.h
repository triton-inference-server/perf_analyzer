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

#include <grpcpp/generic/generic_stub.h>
#include <grpcpp/grpcpp.h>

#include "../client_backend.h"
#include "common.h"
#include "dynamic_grpc_infer_input.h"

namespace tc = triton::client;

namespace triton::perfanalyzer::clientbackend::dynamicgrpc {

struct SslOptions {
  explicit SslOptions() {}
  // File containing the PEM encoding of the server root certificates.
  // If this parameter is empty, the default roots will be used. The
  // default roots can be overridden using the
  // GRPC_DEFAULT_SSL_ROOTS_FILE_PATH environment variable pointing
  // to a file on the file system containing the roots.
  std::string root_certificates;
  // File containing the PEM encoding of the client's private key.
  // This parameter can be empty if the client does not have a
  // private key.
  std::string private_key;
  // File containing the PEM encoding of the client's certificate chain.
  // This parameter can be empty if the client does not have a
  // certificate chain.
  std::string certificate_chain;
};

class DynamicGrpcClient;

//==============================================================================
// An DynamicGrpcRequest represents an inflght request on gRPC.
//
class DynamicGrpcRequest {
 public:
  DynamicGrpcRequest(OnCompleteFn callback = nullptr)
      : callback_(callback), grpc_status_()
  {
  }

  tc::RequestTimers& Timer() { return timer_; }
  friend DynamicGrpcClient;

 private:
  OnCompleteFn callback_;
  // Variables for GRPC call
  grpc::ClientContext grpc_context_;
  grpc::Status grpc_status_;

  // The timers for infer request.
  tc::RequestTimers timer_;
};

//==============================================================
///
class DynamicGrpcInferResult : public InferResult {
 public:
  DynamicGrpcInferResult(
      bool request_status = false,
      std::vector<std::chrono::time_point<std::chrono::system_clock>>
          response_timestamps = {})
      : request_status_(request_status),
        response_timestamps_(response_timestamps)
  {
  }

  /// See InferResult::Id()
  Error Id(std::string* id) const override;
  /// See InferResult::RequestStatus()
  Error RequestStatus() const override;
  /// See InferResult::RawData()
  Error RawData(
      const std::string& output_name, std::vector<uint8_t>& buf) const override;
  Error ResponseTimestamps(
      std::vector<std::chrono::time_point<std::chrono::system_clock>>*
          response_timestamps) const override
  {
    if (response_timestamps == nullptr) {
      return cb::Error("Failed to store response timestamps.");
    }
    *response_timestamps = response_timestamps_;
    return cb::Error::Success;
  }

 private:
  bool request_status_;
  std::vector<std::chrono::time_point<std::chrono::system_clock>>
      response_timestamps_;
};

//==============================================================================
class DynamicGrpcClient {
 public:
  DynamicGrpcClient(
      const std::string& url, const std::string& grpc_method, bool verbose,
      bool use_ssl, const SslOptions& ssl_options);

  ~DynamicGrpcClient();

  /// Runs an synchronous inference over gRPC bi-directional streaming API.
  /// A stream must be established with a call to StartStream() before calling
  /// this function.
  /// \param result Returns the result of inference.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs Optional vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the model
  /// config will be returned as default settings.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in the metadata of gRPC request.
  /// \return Error object indicating success or failure of the
  /// request.
  Error BidiStreamRPC(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>());

  /// Starts a grpc bi-directional stream to send streaming inferences.
  /// \param callback The callback function to be invoked on receiving a
  /// response at the stream.
  /// \param enable_stats Indicates whether client library should record the
  /// the client-side statistics for inference requests on stream or not.
  /// The library does not support client side statistics for decoupled
  /// streaming. Set this option false when there is no 1:1 mapping between
  /// request and response on the stream.
  /// \param headers Optional map specifying additional HTTP headers to
  /// include in the metadata of gRPC request.
  /// \param compression_algorithm The compression algorithm to be used
  /// by gRPC when sending requests. By default compression is not used.
  /// \return Error object indicating success or failure of the request.
  Error StartStream(
      OnCompleteFn callback = [](InferResult*) {}, bool enable_stats = true,
      const Headers& headers = Headers(),
      grpc_compression_algorithm compression_algorithm = GRPC_COMPRESS_NONE);

  /// Stops an active grpc bi-directional stream, if one available.
  /// \return Error object indicating success or failure of the request.
  Error StopStream();

  /// Returns the inference statistics of the client.
  const InferStat& ClientInferStat() { return infer_stat_; }

 protected:
  // Update the infer stat with the given timer
  Error UpdateInferStat(const tc::RequestTimers& timer);
  // Enables verbose operation in the client.
  bool verbose_;

  // The inference statistic of the current client
  InferStat infer_stat_;

 private:
  // Generic bi-directional stream using dynamic protobuf message.
  std::unique_ptr<grpc::GenericClientAsyncReaderWriter> bidi_stream_;
  std::unique_ptr<grpc::ClientContext> grpc_context_;
  std::unique_ptr<grpc::CompletionQueue> completion_queue_;
  bool stream_started_{false};
  bool writesdone_called_{false};

  // Generic gRPC stub for dynamic calls.
  std::unique_ptr<grpc::GenericStub> stub_;
  const std::string grpc_method_;
};


}  // namespace triton::perfanalyzer::clientbackend::dynamicgrpc
