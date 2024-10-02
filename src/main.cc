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

#ifdef TRITON_ENABLE_PERF_ANALYZER_C_API
#include "client_backend/triton_c_api/triton_loader.h"
#endif  // TRITON_ENABLE_PERF_ANALYZER_C_API

#include "perf_analyzer.h"
#include "perf_analyzer_exception.h"

namespace pa = triton::perfanalyzer;

int
main(int argc, char* argv[])
{
  int exit_code = 0;
  try {
    triton::perfanalyzer::CLParser clp;
    pa::PAParamsPtr params = clp.Parse(argc, argv);

    PerfAnalyzer analyzer(params);
    analyzer.Run();
  }
  catch (pa::PerfAnalyzerException& e) {
    std::cerr << e.what() << std::endl;
    exit_code = e.GetError();
  }


#ifdef TRITON_ENABLE_PERF_ANALYZER_C_API
  // destruct static variable before end of program as underlying libraries may
  // use resources in their destruction that won't exist anymore if static
  // variable is destructed at the end of process rather than here explicitly
  pa::clientbackend::tritoncapi::TritonLoader::GetSingleton()->Delete();
#endif  // TRITON_ENABLE_PERF_ANALYZER_C_API

  return exit_code;
}
