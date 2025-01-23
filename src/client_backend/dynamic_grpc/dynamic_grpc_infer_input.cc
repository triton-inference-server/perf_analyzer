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

#include "dynamic_grpc_infer_input.h"

namespace triton::perfanalyzer::clientbackend::dynamicgrpc {

Error
DynamicGrpcInferInput::Create(
    InferInput** infer_input, const std::string& name,
    const std::vector<int64_t>& dims, const std::string& datatype,
    const bool streaming)
{
  DynamicGrpcInferInput* local_infer_input =
      new DynamicGrpcInferInput(name, dims, datatype, streaming);

  *infer_input = local_infer_input;
  return Error::Success;
}

Error
DynamicGrpcInferInput::AppendRaw(const uint8_t* input, size_t input_byte_size)
{
  if (bufs_.size() > 0) {
    return Error(
        "Dynamic gRPC does not support multi-batch at the moment.",
        pa::GENERIC_ERROR);
  }

  byte_size_ += input_byte_size;

  bufs_.push_back(input);
  buf_byte_sizes_.push_back(input_byte_size);
  return Error::Success;
}

Error
DynamicGrpcInferInput::PrepareForRequest()
{
  // Reset position so request sends entire input.
  bufs_idx_ = 0;
  buf_pos_ = 0;
  return Error::Success;
}

std::vector<std::vector<char>>
DynamicGrpcInferInput::GetSerializedMessages()
{
  if (!streaming_) {
    std::cerr << "Cannot get next serialized message for non-streaming RPC."
              << std::endl;
    throw PerfAnalyzerException(GENERIC_ERROR);
  }

  std::vector<std::vector<char>> messages;
  std::vector<char> message;
  uint32_t message_size = 4;

  while (buf_pos_ + message_size < buf_byte_sizes_[bufs_idx_]) {
    // Read message size
    std::memcpy(&message_size, (bufs_[bufs_idx_] + buf_pos_), sizeof(uint32_t));
    buf_pos_ += sizeof(uint32_t);

    if (message.size() != message_size) {
      message.resize(message_size);
    }

    // Read message
    std::memcpy(message.data(), (bufs_[bufs_idx_] + buf_pos_), message_size);
    buf_pos_ += message_size;

    messages.push_back(message);
    message_size = 4;
  }

  return messages;
}

DynamicGrpcInferInput::DynamicGrpcInferInput(
    const std::string& name, const std::vector<int64_t>& dims,
    const std::string& datatype, const bool streaming)
    : InferInput(BackendKind::DYNAMIC_GRPC, name, datatype), shape_(dims),
      streaming_(streaming)
{
}

}  // namespace triton::perfanalyzer::clientbackend::dynamicgrpc
