# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters import *
from genai_perf.inputs.input_constants import OutputFormat


class OutputFormatConverterFactory:
    """
    This class converts the generic JSON to the specific format
    used by a given endpoint.
    """

    @staticmethod
    def create(output_format: OutputFormat):
        converters = {
            OutputFormat.IMAGE_RETRIEVAL: OpenAIChatCompletionsConverter,
            OutputFormat.NVCLIP: NVClipConverter,
            OutputFormat.OPENAI_CHAT_COMPLETIONS: OpenAIChatCompletionsConverter,
            OutputFormat.OPENAI_COMPLETIONS: OpenAICompletionsConverter,
            OutputFormat.OPENAI_EMBEDDINGS: OpenAIEmbeddingsConverter,
            OutputFormat.OPENAI_VISION: OpenAIChatCompletionsConverter,
            OutputFormat.RANKINGS: RankingsConverter,
            OutputFormat.TENSORRTLLM: TensorRTLLMConverter,
            OutputFormat.TENSORRTLLM_ENGINE: TensorRTLLMEngineConverter,
            OutputFormat.VLLM: VLLMConverter,
        }
        if output_format not in converters:
            raise GenAIPerfException(f"Output format {output_format} is not supported")
        return converters[output_format]()
