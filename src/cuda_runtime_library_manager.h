// Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stddef.h>

#include <stdexcept>

namespace triton::perfanalyzer {

class CUDARuntimeLibraryManager {
 public:
  CUDARuntimeLibraryManager();

  ~CUDARuntimeLibraryManager();

  using cudaError_t = ::cudaError_t;
  using cudaIpcMemHandle_t = ::cudaIpcMemHandle_t;
  using cudaMemcpyKind = ::cudaMemcpyKind;

  // Wrapper functions for CUDA API calls
  cudaError_t cudaMalloc(void** devPtr, size_t size);
  cudaError_t cudaFree(void* devPtr);
  const char* cudaGetErrorName(cudaError_t error);
  const char* cudaGetErrorString(cudaError_t error);
  cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle, void* devPtr);
  cudaError_t cudaMemcpy(
      void* dst, const void* src, size_t count, cudaMemcpyKind kind);
  cudaError_t cudaSetDevice(int device);

 private:
  cudaError_t (*cuda_malloc_func_)(void**, size_t){nullptr};
  cudaError_t (*cuda_free_func_)(void*){nullptr};
  const char* (*cuda_get_error_name_func_)(cudaError_t){nullptr};
  const char* (*cuda_get_error_string_func_)(cudaError_t){nullptr};
  cudaError_t (*cuda_ipc_get_mem_handle_func_)(cudaIpcMemHandle_t*, void*){
      nullptr};
  cudaError_t (*cuda_memcpy_func_)(void*, const void*, size_t, cudaMemcpyKind){
      nullptr};
  cudaError_t (*cuda_set_device_func_)(int){nullptr};

  void* handle_{nullptr};

  void LoadFunctions();
};

}  // namespace triton::perfanalyzer
