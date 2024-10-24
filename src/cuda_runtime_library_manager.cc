// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_runtime_library_manager.h"

namespace triton::perfanalyzer {

CUDARuntimeLibraryManager::CUDARuntimeLibraryManager()
{
  handle_ = dlopen("libcudart.so", RTLD_LAZY);
  if (handle_ == nullptr) {
    throw std::runtime_error(
        std::string("Cannot open CUDA runtime library: ") + dlerror() + "\n");
  }

  LoadFunctions();
}

CUDARuntimeLibraryManager::~CUDARuntimeLibraryManager()
{
  if (handle_) {
    dlclose(handle_);
  }
}

cudaError_t
CUDARuntimeLibraryManager::cudaMalloc(void** devPtr, size_t size)
{
  return cuda_malloc_func_(devPtr, size);
}

cudaError_t
CUDARuntimeLibraryManager::cudaFree(void* devPtr)
{
  return cuda_free_func_(devPtr);
}

const char*
CUDARuntimeLibraryManager::cudaGetErrorName(cudaError_t error)
{
  return cuda_get_error_name_func_(error);
}

const char*
CUDARuntimeLibraryManager::cudaGetErrorString(cudaError_t error)
{
  return cuda_get_error_string_func_(error);
}

cudaError_t
CUDARuntimeLibraryManager::cudaIpcGetMemHandle(
    cudaIpcMemHandle_t* handle, void* devPtr)
{
  return cuda_ipc_get_mem_handle_func_(handle, devPtr);
}

cudaError_t
CUDARuntimeLibraryManager::cudaMemcpy(
    void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
  return cuda_memcpy_func_(dst, src, count, kind);
}

cudaError_t
CUDARuntimeLibraryManager::cudaSetDevice(int device)
{
  return cuda_set_device_func_(device);
}

void
CUDARuntimeLibraryManager::LoadFunctions()
{
  *(void**)(&cuda_malloc_func_) = dlsym(handle_, "cudaMalloc");
  *(void**)(&cuda_free_func_) = dlsym(handle_, "cudaFree");
  *(void**)(&cuda_get_error_name_func_) = dlsym(handle_, "cudaGetErrorName");
  *(void**)(&cuda_get_error_string_func_) =
      dlsym(handle_, "cudaGetErrorString");
  *(void**)(&cuda_ipc_get_mem_handle_func_) =
      dlsym(handle_, "cudaIpcGetMemHandle");
  *(void**)(&cuda_memcpy_func_) = dlsym(handle_, "cudaMemcpy");
  *(void**)(&cuda_set_device_func_) = dlsym(handle_, "cudaSetDevice");

  if (cuda_malloc_func_ == nullptr || cuda_free_func_ == nullptr ||
      cuda_get_error_name_func_ == nullptr ||
      cuda_get_error_string_func_ == nullptr ||
      cuda_ipc_get_mem_handle_func_ == nullptr ||
      cuda_memcpy_func_ == nullptr || cuda_set_device_func_ == nullptr) {
    throw std::runtime_error(
        std::string("Failed to load one or more CUDA functions: ") + dlerror() +
        "\n");
  }
}

}  // namespace triton::perfanalyzer
