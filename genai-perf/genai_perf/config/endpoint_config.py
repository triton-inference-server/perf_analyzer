# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional

from genai_perf.inputs.input_constants import OutputFormat


@dataclass
class EndpointConfig:
    endpoint: Optional[str]
    service_kind: str
    output_format: OutputFormat


endpoint_type_map = {
    "chat": EndpointConfig(
        "v1/chat/completions", "openai", OutputFormat.OPENAI_CHAT_COMPLETIONS
    ),
    "completions": EndpointConfig(
        "v1/completions", "openai", OutputFormat.OPENAI_COMPLETIONS
    ),
    "dynamic_grpc": EndpointConfig(None, "dynamic_grpc", OutputFormat.DYNAMIC_GRPC),
    "embeddings": EndpointConfig(
        "v1/embeddings", "openai", OutputFormat.OPENAI_EMBEDDINGS
    ),
    # HuggingFace TGI only exposes root endpoint, so use that as endpoint
    "huggingface_generate": EndpointConfig(
        ".", "openai", OutputFormat.HUGGINGFACE_GENERATE
    ),
    "image_retrieval": EndpointConfig(
        "v1/infer", "openai", OutputFormat.IMAGE_RETRIEVAL
    ),
    "nvclip": EndpointConfig("v1/embeddings", "openai", OutputFormat.NVCLIP),
    "rankings": EndpointConfig("v1/ranking", "openai", OutputFormat.RANKINGS),
    "multimodal": EndpointConfig(
        "v1/chat/completions", "openai", OutputFormat.OPENAI_MULTIMODAL
    ),
    "generate": EndpointConfig(
        "v2/models/{MODEL_NAME}/generate", "openai", OutputFormat.TRITON_GENERATE
    ),
    "kserve": EndpointConfig(None, "triton", OutputFormat.TENSORRTLLM),
    "template": EndpointConfig(None, "triton", OutputFormat.TEMPLATE),
    "tensorrtllm_engine": EndpointConfig(
        None, "tensorrtllm_engine", OutputFormat.TENSORRTLLM_ENGINE
    ),
    # TODO: Deprecate this endpoint. Currently we have it for backward compatibility.
    "vision": EndpointConfig(
        "v1/chat/completions", "openai", OutputFormat.OPENAI_MULTIMODAL
    ),
}
