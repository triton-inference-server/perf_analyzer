// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <grpcpp/grpcpp.h>

#include <queue>

#include "../client_backend.h"
#include "common.h"

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

// GRPC KeepAlive: https://grpc.github.io/grpc/cpp/md_doc_keepalive.html
struct KeepAliveOptions {
  explicit KeepAliveOptions()
      : keepalive_time_ms(INT_MAX), keepalive_timeout_ms(20000),
        keepalive_permit_without_calls(false), http2_max_pings_without_data(2)
  {
  }
  // The period (in milliseconds) after which a keepalive ping is sent on the
  // transport
  int keepalive_time_ms;
  // The amount of time (in milliseconds) the sender of the keepalive ping waits
  // for an acknowledgement. If it does not receive an acknowledgment within
  // this time, it will close the connection.
  int keepalive_timeout_ms;
  // If true, allow keepalive pings to be sent even if there are no calls in
  // flight.
  bool keepalive_permit_without_calls;
  // The maximum number of pings that can be sent when there is no data/header
  // frame to be sent. gRPC Core will not continue sending pings if we run over
  // the limit. Setting it to 0 allows sending pings without such a restriction.
  int http2_max_pings_without_data;
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
    // TODO: store response message (e.g. AnimateResponse)
    // grpc_response_(std::make_shared<inference::ModelInferResponse>())
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
      // TODO: need to store grpc response
      // std::shared_ptr<tensorflow::serving::PredictResponse> response,
      Error& request_status)
      : request_status_(request_status)
  {
    // TODO: add grpc response
  }

  /// See InferResult::Id()
  Error Id(std::string* id) const override;
  /// See InferResult::RequestStatus()
  Error RequestStatus() const override;
  /// See InferResult::RawData()
  Error RawData(
      const std::string& output_name, std::vector<uint8_t>& buf) const override;

 private:
  // TODO: need to store grpc response
  // std::shared_ptr<tensorflow::serving::PredictResponse> response_;
  Error request_status_;
};

//==============================================================================
class DynamicGrpcClient {
 public:
  DynamicGrpcClient(
      const std::string& url, bool verbose, bool use_ssl,
      const SslOptions& ssl_options);

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
  /// \param compression_algorithm The compression algorithm to be used
  /// by gRPC when sending requests. By default compression is not used.
  /// \return Error object indicating success or failure of the
  /// request.
  Error BidiStreamRPC(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>(),
      const Headers& headers = Headers(),
      grpc_compression_algorithm compression_algorithm = GRPC_COMPRESS_NONE);

  /// Starts a grpc bi-directional stream to send streaming inferences.
  /// \param callback The callback function to be invoked on receiving a
  /// response at the stream.
  /// \param enable_stats Indicates whether client library should record the
  /// the client-side statistics for inference requests on stream or not.
  /// The library does not support client side statistics for decoupled
  /// streaming. Set this option false when there is no 1:1 mapping between
  /// request and response on the stream.
  /// \param stream_timeout Specifies the end-to-end timeout for the streaming
  /// connection in microseconds. The default value is 0 which means that
  /// there is no limitation on deadline. The stream will be closed once
  /// the specified time elapses.
  /// \param headers Optional map specifying additional HTTP headers to
  /// include in the metadata of gRPC request.
  /// \param compression_algorithm The compression algorithm to be used
  /// by gRPC when sending requests. By default compression is not used.
  /// \return Error object indicating success or failure of the request.
  Error StartStream(
      OnCompleteFn callback, bool enable_stats = true,
      uint32_t stream_timeout = 0, const Headers& headers = Headers(),
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
  Error PreRunProcessing(
      const InferOptions& options, const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs);

  // TODO: not needed since we are using synchronous stream?
  // Required to support the grpc bi-directional streaming API.
  // std::thread stream_worker_;

  // TODO: update
  // std::shared_ptr<grpc::ClientReaderWriter<
  //     inference::ModelInferRequest, inference::ModelStreamInferResponse>>
  //     grpc_stream_;
  grpc::ClientContext grpc_context_;

  bool enable_stream_stats_;
  std::queue<std::unique_ptr<tc::RequestTimers>> ongoing_stream_request_timers_;

  // TODO: not needed since we are using synchronous stream?
  // std::mutex stream_mutex_;

  // TODO: update
  // GRPC end point.
  // std::shared_ptr<inference::GRPCInferenceService::Stub> stub_;

  // TODO: update
  // request for GRPC call, one request object can be used for multiple calls
  // since it can be overwritten as soon as the GRPC send finishes.
  // inference::ModelInferRequest infer_request_;
};


}  // namespace triton::perfanalyzer::clientbackend::dynamicgrpc
