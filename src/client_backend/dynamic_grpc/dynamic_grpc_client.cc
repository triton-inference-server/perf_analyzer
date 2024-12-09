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

#include "dynamic_grpc_client.h"

#include <chrono>
#include <cstdint>
#include <string>

namespace triton::perfanalyzer::clientbackend::dynamicgrpc {

Error
GrpcInferResult::RequestStatus() const
{
  // TODO: Not required for synchronous streaming?
  return Error("Not implemented yet.");
}

Error
GrpcInferResult::Id(std::string* id) const
{
  // TODO: Not required for synchronous streaming?
  return Error("Not implemented yet.");
}

Error
GrpcInferResult::RawData(
    const std::string& output_name, const uint8_t** buf,
    size_t* byte_size) const
{
  // TODO
  return Error("Not implemented yet.");
}

//==============================================================================
//
GrpcClient::GrpcClient(
    const std::string& url, bool verbose, bool use_ssl,
    const SslOptions& ssl_options)
    : verbose_(verbose)
{
  // TODO
  std::cerr << "Not implemented yet." << std::endl;
  throw PerfAnalyzerException(GENERIC_ERROR);
}

GrpcClient::~GrpcClient()
{
  // TODO
  std::cerr << "Not implemented yet." << std::endl;
  throw PerfAnalyzerException(GENERIC_ERROR);
}

Error
GrpcClient::BidiStreamRPC(
    InferResult** result, const InferOptions& options,
    const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs,
    const Headers& headers, grpc_compression_algorithm compression_algorithm)
{
  // TODO
  return Error("Not implemented yet.");
}

Error
GrpcClient::StartStream(
    OnCompleteFn callback, bool enable_stats, uint32_t stream_timeout,
    const Headers& headers, grpc_compression_algorithm compression_algorithm)
{
  // TODO
  return Error("Not implemented yet.");
}

Error
GrpcClient::StopStream()
{
  // TODO
  return Error("Not implemented yet.");
}

Error
GrpcClient::PreRunProcessing(
    const InferOptions& options, const std::vector<InferInput*>& inputs,
    const std::vector<const InferRequestedOutput*>& outputs)
{
  // TODO
  return Error("Not implemented yet.");
}

Error
GrpcClient::UpdateInferStat(const tc::RequestTimers& timer)
{
  // TODO
  return Error("Not implemented yet.");
}

//==============================================================================

}  // namespace triton::perfanalyzer::clientbackend::dynamicgrpc
