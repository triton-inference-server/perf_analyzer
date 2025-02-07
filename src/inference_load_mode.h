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

#include <stdexcept>
#include <string>

namespace triton::perfanalyzer {

enum class InferenceLoadMode {
  None,
  Concurrency,
  RequestRate,
  CustomIntervals,
  PeriodicConcurrency,
  SessionConcurrency,
  FixedSchedule
};

inline std::string
to_string(InferenceLoadMode inference_load_mode)
{
  switch (inference_load_mode) {
    case InferenceLoadMode::None:
      return "None";
    case InferenceLoadMode::Concurrency:
      return "Concurrency";
    case InferenceLoadMode::CustomIntervals:
      return "CustomIntervals";
    case InferenceLoadMode::PeriodicConcurrency:
      return "PeriodicConcurrency";
    case InferenceLoadMode::SessionConcurrency:
      return "SessionConcurrency";
    case InferenceLoadMode::FixedSchedule:
      return "FixedSchedule";
  }
  throw std::invalid_argument("Invalid InferenceLoadMode value");
}

}  // namespace triton::perfanalyzer
