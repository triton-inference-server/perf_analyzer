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

#include "dynamic_grpc_client_backend.h"


namespace triton::perfanalyzer::clientbackend::dynamicgrpc {

std::pair<bool, SslOptions>
ParseGrpcSslOptions(const cb::SslOptionsBase& ssl_options)
{
  bool use_ssl = ssl_options.ssl_grpc_use_ssl;

  SslOptions grpc_ssl_options;
  grpc_ssl_options.root_certificates =
      ssl_options.ssl_grpc_root_certifications_file;
  grpc_ssl_options.private_key = ssl_options.ssl_grpc_private_key_file;
  grpc_ssl_options.certificate_chain =
      ssl_options.ssl_grpc_certificate_chain_file;

  return std::pair<bool, SslOptions>{use_ssl, grpc_ssl_options};
}

Error
DynamicGrpcClientBackend::Create(
    const std::string& url, const ProtocolType protocol,
    const SslOptionsBase& ssl_options,
    const grpc_compression_algorithm compression_algorithm,
    std::shared_ptr<Headers> http_headers, const std::string& proto_file,
    const std::string& grpc_method, const bool verbose,
    std::unique_ptr<ClientBackend>* client_backend)
{
  if (protocol == ProtocolType::HTTP) {
    return Error(
        "perf_analyzer does not support http protocol with gRPC services.");
  }
  std::unique_ptr<DynamicGrpcClientBackend> grpc_client_backend(
      new DynamicGrpcClientBackend(compression_algorithm, http_headers));

  // Initialize gRPC client
  std::pair<bool, SslOptions> grpc_ssl_options_pair =
      ParseGrpcSslOptions(ssl_options);
  bool use_ssl = grpc_ssl_options_pair.first;
  SslOptions grpc_ssl_options = grpc_ssl_options_pair.second;
  grpc_client_backend->grpc_client_ = std::make_unique<DynamicGrpcClient>(
      url, proto_file, grpc_method, verbose, use_ssl, grpc_ssl_options);

  *client_backend = std::move(grpc_client_backend);

  return Error::Success;
}

Error
DynamicGrpcClientBackend::StreamInfer(
    cb::InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  auto raw_input = dynamic_cast<DynamicGrpcInferInput*>(inputs[0]);
  raw_input->PrepareForRequest();
  RETURN_IF_CB_ERROR(grpc_client_->BidiStreamRPC(
      result, options, inputs, outputs, compression_algorithm_));

  return Error::Success;
}

Error
DynamicGrpcClientBackend::StartStream(OnCompleteFn callback, bool enable_stats)
{
  RETURN_IF_CB_ERROR(grpc_client_->StartStream(
      callback, enable_stats, *http_headers_, compression_algorithm_));

  return Error::Success;
}

Error
DynamicGrpcClientBackend::StopStream()
{
  RETURN_IF_CB_ERROR(grpc_client_->StopStream());
  return Error::Success;
}

Error
DynamicGrpcClientBackend::ClientInferStat(InferStat* infer_stat)
{
  *infer_stat = grpc_client_->ClientInferStat();
  return Error::Success;
}

//==============================================================================

Error
DynamicGrpcInferRequestedOutput::Create(
    InferRequestedOutput** infer_output, const std::string& name)
{
  DynamicGrpcInferRequestedOutput* local_infer_output =
      new DynamicGrpcInferRequestedOutput(name);

  tc::InferRequestedOutput* dynamic_grpc_infer_output;
  RETURN_IF_TRITON_ERROR(
      tc::InferRequestedOutput::Create(&dynamic_grpc_infer_output, name));
  local_infer_output->output_.reset(dynamic_grpc_infer_output);

  *infer_output = local_infer_output;

  return Error::Success;
}

DynamicGrpcInferRequestedOutput::DynamicGrpcInferRequestedOutput(
    const std::string& name)
    : InferRequestedOutput(BackendKind::DYNAMIC_GRPC, name)
{
}

}  // namespace triton::perfanalyzer::clientbackend::dynamicgrpc
