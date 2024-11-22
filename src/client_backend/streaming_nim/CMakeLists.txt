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

FetchContent_Declare(speechliveportrait-repo
  GIT_REPOSITORY "https://oauth2:${GITLAB_PERSONAL_ACCESS_TOKEN}@gitlab-master.nvidia.com/maxine-nvcf/features/maxine-nvcf-speechliveportrait.git"
  GIT_TAG "main"
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/speechliveportrait-repo/src/speechliveportrait"
)
FetchContent_MakeAvailable(speechliveportrait-repo)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/protos)
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/speechliveportrait-repo/src/speechliveportrait
                                                           ${CMAKE_BINARY_DIR}/protos/speechliveportrait)

set(SPEECH_LIVE_PORTRAIT_PROTO ${CMAKE_BINARY_DIR}/protos/speechliveportrait/grpc/protos/nvidia/maxine/speechliveportrait/v1/speechliveportrait.proto)
set(SPEECH_LIVE_PORTRAIT_PROTO_DIR ${CMAKE_BINARY_DIR}/protos/speechliveportrait/grpc/protos/nvidia/maxine/speechliveportrait/v1)
set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/compiled")

file(MAKE_DIRECTORY ${PROTO_BINARY_DIR})

add_library(speechliveportrait-proto OBJECT ${SPEECH_LIVE_PORTRAIT_PROTO})

target_link_libraries(speechliveportrait-proto PUBLIC protobuf::libprotobuf gRPC::grpc++)

protobuf_generate(
  TARGET speechliveportrait-proto
  LANGUAGE cpp
  GENERATE_EXTENSIONS .pb.h .pb.cc
  IMPORT_DIRS ${SPEECH_LIVE_PORTRAIT_PROTO_DIR}
  PROTOC_OUT_DIR "${PROTO_BINARY_DIR}"
)

protobuf_generate(
  TARGET speechliveportrait-proto
  LANGUAGE grpc
  GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
  PLUGIN protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
  IMPORT_DIRS ${SPEECH_LIVE_PORTRAIT_PROTO_DIR}
  PROTOC_OUT_DIR "${PROTO_BINARY_DIR}"
)
