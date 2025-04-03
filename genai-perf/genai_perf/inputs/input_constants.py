# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum, auto


class Subcommand(Enum):
    CONFIG = "config"
    PROFILE = "profile"
    ANALYZE = "analyze"
    TEMPLATE = "create-template"
    PROCESS = "process-export-files"


class ModelSelectionStrategy(Enum):
    ROUND_ROBIN = "ROUND_ROBIN"
    RANDOM = "RANDOM"


class PromptSource(Enum):
    SYNTHETIC = "SYNTHETIC"
    FILE = "FILE"
    PAYLOAD = "PAYLOAD"


class AudioFormat(Enum):
    WAV = "WAV"
    MP3 = "MP3"


class ImageFormat(Enum):
    PNG = "PNG"
    JPEG = "JPEG"


class PerfAnalyzerMeasurementMode(Enum):
    REQUEST_COUNT = "REQUEST_COUNT"
    INTERVAL = "INTERVAL"


class OutputFormat(Enum):
    ################################################################
    # Triton backends
    ################################################################
    TENSORRTLLM = "TENSORRTLLM"
    VLLM = "VLLM"
    ################################################################
    # Other output formats
    ################################################################
    IMAGE_RETRIEVAL = "IMAGE_RETRIEVAL"
    DYNAMIC_GRPC = "DYNAMIC_GRPC"
    NVCLIP = "NVCLIP"
    OPENAI_CHAT_COMPLETIONS = "OPENAI_CHAT_COMPLETIONS"
    OPENAI_COMPLETIONS = "OPENAI_COMPLETIONS"
    OPENAI_EMBEDDINGS = "OPENAI_EMBEDDINGS"
    OPENAI_MULTIMODAL = "OPENAI_MULTIMODAL"
    RANKINGS = "RANKINGS"
    TEMPLATE = "TEMPLATE"
    TENSORRTLLM_ENGINE = "TENSORRTLLM_ENGINE"
    TRITON_GENERATE = "TRITON_GENERATE"
    HUGGINGFACE_GENERATE = "HUGGINGFACE_GENERATE"

    def to_lowercase(self):
        return self.name.lower()


###########################
# General Parameters
###########################
DEFAULT_INPUT_DATA_JSON = "inputs.json"
DEFAULT_SYNTHETIC_FILENAME = "synthetic_data.json"
PAYLOAD_METADATA_FIELDS = ["timestamp", "delay", "session_id"]
PAYLOAD_METADATA_INT_FIELDS = ["timestamp", "delay"]

###########################
# Default Prompt Parameters
###########################
DEFAULT_CORPUS_FILE = "shakespeare.txt"
MINIMUM_LENGTH = 1
DEFAULT_TENSORRTLLM_MAX_TOKENS = 256
