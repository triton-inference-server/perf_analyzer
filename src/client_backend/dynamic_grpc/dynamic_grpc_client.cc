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

#include "dynamic_grpc_client.h"


namespace triton::perfanalyzer::clientbackend::dynamicgrpc {

Error
DynamicGrpcInferResult::RequestStatus() const
{
  if (!request_status_) {
    return Error("Dynamic gRPC client request returned with error.");
  }
  return Error::Success;
}

Error
DynamicGrpcInferResult::Id(std::string* id) const
{
  return Error("DynamicGrpcInferResult::Id is not supported.");
}

Error
DynamicGrpcInferResult::RawData(
    const std::string& output_name, std::vector<uint8_t>& buf) const
{
  return Error("DynamicGrpcInferResult::RawData is not supported.");
}

//==============================================================================
//
DynamicGrpcClient::DynamicGrpcClient(
    const std::string& url, const std::string& grpc_method, bool verbose,
    bool use_ssl, const SslOptions& ssl_options)
    : verbose_(verbose), grpc_method_(grpc_method)
{
  if (verbose) {
    std::cout << "Creating new channel with url: " << url << std::endl;
  }
  auto channel = grpc::CreateChannel(url, grpc::InsecureChannelCredentials());
  stub_ = std::make_unique<grpc::GenericStub>(channel);

  // TODO: Get streaming information so determine if we need to start the
  // stream. For now, we always start the stream as dynamic gRPC client only
  // supports bidirectional streaming RPC.
  StartStream();
}

DynamicGrpcClient::~DynamicGrpcClient()
{
  if (stream_started_) {
    StopStream();
  }
}

Error
DynamicGrpcClient::BidiStreamRPC(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  // Dynamic grpc client requires to restart the stream before every new request
  // because the for each request, the stream is half-closed from the client
  // side due to calling WritesDone.
  if (writesdone_called_) {
    StopStream();
    StartStream();
    writesdone_called_ = false;
  }

  auto request = std::make_shared<DynamicGrpcRequest>();
  request->Timer().Reset();

  // For bidirectional streaming, there's only one input data that contains
  // the entire serialized protobuf messages
  auto stream_input = dynamic_cast<DynamicGrpcInferInput*>(inputs[0]);
  auto messages = stream_input->GetSerializedMessages();

  request->Timer().CaptureTimestamp(tc::RequestTimers::Kind::REQUEST_START);
  request->Timer().CaptureTimestamp(tc::RequestTimers::Kind::SEND_START);

  void* tag;
  bool ok;

  for (const auto& message : messages) {
    grpc::Slice slice(message.data(), message.size());
    grpc::ByteBuffer write_buffer(&slice, 1);
    bidi_stream_->Write(write_buffer, nullptr);
    completion_queue_->Next(&tag, &ok);
  }

  if (!ok) {
    return Error(
        "Failed to write data to the gRPC stream. This could happen when the "
        "call is dead or server dropped the channel.");
  }

  bidi_stream_->WritesDone(nullptr);
  completion_queue_->Next(&tag, &ok);
  writesdone_called_ = true;

  request->Timer().CaptureTimestamp(tc::RequestTimers::Kind::SEND_END);
  request->Timer().CaptureTimestamp(tc::RequestTimers::Kind::RECV_START);

  while (true) {
    grpc::ByteBuffer read_buffer;
    bidi_stream_->Read(&read_buffer, nullptr);
    bool status = completion_queue_->Next(&tag, &ok);
    if (!ok) {
      *result = new DynamicGrpcInferResult(status);
      break;
    }
  }

  request->Timer().CaptureTimestamp(tc::RequestTimers::Kind::RECV_END);
  request->Timer().CaptureTimestamp(tc::RequestTimers::Kind::REQUEST_END);

  Error update_status = UpdateInferStat(request->Timer());
  if (!update_status.IsOk()) {
    std::cerr << "Failed to update infer stats: " << update_status << std::endl;
  }
  return Error::Success;
}

Error
DynamicGrpcClient::StartStream(
    OnCompleteFn callback, bool enable_stats, const Headers& headers,
    grpc_compression_algorithm compression_algorithm)
{
  // Reset the context and queue
  grpc_context_ = std::make_unique<grpc::ClientContext>();
  completion_queue_ = std::make_unique<grpc::CompletionQueue>();

  // Set grpc contexts
  for (const auto& it : headers) {
    grpc_context_->AddMetadata(it.first, it.second);
  }
  grpc_context_->set_compression_algorithm(compression_algorithm);

  bidi_stream_ = stub_->PrepareCall(
      grpc_context_.get(), "/" + grpc_method_, completion_queue_.get());

  // Wait for StartCall to complete before proceeding with other operations
  void* tag;
  bool ok;
  bidi_stream_->StartCall(nullptr);
  completion_queue_->Next(&tag, &ok);

  stream_started_ = true;

  if (verbose_) {
    std::cout << "Started stream..." << std::endl;
  }

  return Error::Success;
}

Error
DynamicGrpcClient::StopStream()
{
  grpc::Status grpc_status_;
  bidi_stream_->Finish(&grpc_status_, nullptr);
  completion_queue_->Shutdown();
  stream_started_ = false;

  if (verbose_) {
    std::cout << "Stopped stream..." << std::endl;
  }
  return Error::Success;
}

Error
DynamicGrpcClient::UpdateInferStat(const tc::RequestTimers& timer)
{
  const uint64_t request_time_ns = timer.Duration(
      triton::client::RequestTimers::Kind::REQUEST_START,
      triton::client::RequestTimers::Kind::REQUEST_END);
  const uint64_t send_time_ns = timer.Duration(
      triton::client::RequestTimers::Kind::SEND_START,
      triton::client::RequestTimers::Kind::SEND_END);
  const uint64_t recv_time_ns = timer.Duration(
      triton::client::RequestTimers::Kind::RECV_START,
      triton::client::RequestTimers::Kind::RECV_END);

  if ((request_time_ns == std::numeric_limits<uint64_t>::max()) ||
      (send_time_ns == std::numeric_limits<uint64_t>::max()) ||
      (recv_time_ns == std::numeric_limits<uint64_t>::max())) {
    return Error(
        "Timer not set correctly." +
        ((timer.Timestamp(triton::client::RequestTimers::Kind::REQUEST_START) >
          timer.Timestamp(triton::client::RequestTimers::Kind::REQUEST_END))
             ? (" Request time from " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::REQUEST_START)) +
                " to " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::REQUEST_END)) +
                ".")
             : "") +
        ((timer.Timestamp(triton::client::RequestTimers::Kind::SEND_START) >
          timer.Timestamp(triton::client::RequestTimers::Kind::SEND_END))
             ? (" Send time from " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::SEND_START)) +
                " to " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::SEND_END)) +
                ".")
             : "") +
        ((timer.Timestamp(triton::client::RequestTimers::Kind::RECV_START) >
          timer.Timestamp(triton::client::RequestTimers::Kind::RECV_END))
             ? (" Receive time from " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::RECV_START)) +
                " to " +
                std::to_string(timer.Timestamp(
                    triton::client::RequestTimers::Kind::RECV_END)) +
                ".")
             : ""));
  }

  infer_stat_.completed_request_count++;
  infer_stat_.cumulative_total_request_time_ns += request_time_ns;
  infer_stat_.cumulative_send_time_ns += send_time_ns;
  infer_stat_.cumulative_receive_time_ns += recv_time_ns;

  return Error::Success;
}

//==============================================================================

}  // namespace triton::perfanalyzer::clientbackend::dynamicgrpc
