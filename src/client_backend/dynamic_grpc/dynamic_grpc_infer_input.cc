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
    const std::vector<int64_t>& dims, const std::string& datatype)
{
  DynamicGrpcInferInput* local_infer_input =
      new DynamicGrpcInferInput(name, dims, datatype);

  *infer_input = local_infer_input;
  return Error::Success;
}

Error
DynamicGrpcInferInput::SetShape(const std::vector<int64_t>& shape)
{
  // TODO: Not required for synchronous streaming?
  return Error("Not implemented yet.");
}

Error
DynamicGrpcInferInput::Reset()
{
  // TODO: Not required for synchronous streaming?
  return Error("Not implemented yet.");
}

Error
DynamicGrpcInferInput::AppendRaw(const uint8_t* input, size_t input_byte_size)
{
  // TODO: Not required for synchronous streaming?
  return Error("Not implemented yet.");
}

Error
DynamicGrpcInferInput::ByteSize(size_t* byte_size) const
{
  // TODO: Not required for synchronous streaming?
  return Error("Not implemented yet.");
}

Error
DynamicGrpcInferInput::PrepareForRequest()
{
  // TODO: Not required for synchronous streaming?
  return Error("Not implemented yet.");
}

Error
DynamicGrpcInferInput::GetNext(
    const uint8_t** buf, size_t* input_bytes, bool* end_of_input)
{
  // TODO: Not required for synchronous streaming?
  return Error("Not implemented yet.");
}

DynamicGrpcInferInput::DynamicGrpcInferInput(
    const std::string& name, const std::vector<int64_t>& dims,
    const std::string& datatype)
    : InferInput(BackendKind::DYNAMIC_GRPC, name, datatype), shape_(dims)
{
}

}  // namespace triton::perfanalyzer::clientbackend::dynamicgrpc
