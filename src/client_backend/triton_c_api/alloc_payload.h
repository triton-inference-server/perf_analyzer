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

#include <triton/core/tritonserver.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace triton { namespace perfanalyzer { namespace clientbackend {
namespace tritoncapi {

struct AllocPayload {
  struct OutputInfo {
    enum Kind { BINARY, SHM };

    Kind kind_;
    void* base_;
    uint64_t byte_size_;
    TRITONSERVER_MemoryType memory_type_;
    int64_t device_id_;

    // For shared memory
    OutputInfo(
        void* base, uint64_t byte_size, TRITONSERVER_MemoryType memory_type,
        int64_t device_id)
        : kind_(SHM), base_(base), byte_size_(byte_size),
          memory_type_(memory_type), device_id_(device_id)
    {
    }
  };

  ~AllocPayload()
  {
    for (auto it : output_map_) {
      delete it.second;
    }
  }

  std::unordered_map<std::string, OutputInfo*> output_map_;
};

}}}}  // namespace triton::perfanalyzer::clientbackend::tritoncapi
