# Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required (VERSION 3.18)

set(
    TS_CLIENT_BACKEND_SRCS
    torchserve_client_backend.cc
    torchserve_infer_input.cc
    torchserve_http_client.cc
)

set(
    TS_CLIENT_BACKEND_HDRS
    torchserve_client_backend.h
    torchserve_infer_input.h
    torchserve_http_client.h
)

add_library(
    ts-client-backend-library  EXCLUDE_FROM_ALL OBJECT
    ${TS_CLIENT_BACKEND_SRCS}
    ${TS_CLIENT_BACKEND_HDRS}
)

# Include header files installed by CC client
target_include_directories(
  ts-client-backend-library
  PRIVATE
    ${CMAKE_INSTALL_PREFIX}/include
)

target_link_libraries(
  ts-client-backend-library
  PUBLIC CURL::libcurl
  PUBLIC ${TRITON_HTTP_STATIC_LIB}
)

if(${TRITON_ENABLE_GPU})
  target_include_directories(ts-client-backend-library
                             PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

  # Need to enable this compilation flag, otherwise, ipc.h
  # will select mock-struct as opposed to the correct cuda
  # runtime header file.
  target_compile_definitions(
    ts-client-backend-library
    PUBLIC TRITON_ENABLE_GPU=1
  )
endif() # TRITON_ENABLE_GPU
