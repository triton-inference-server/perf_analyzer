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
    DYANMIC_GRPC = "DYANMIC_GRPC"
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
DEFAULT_BATCH_SIZE = 1
DEFAULT_INPUT_DATA_JSON = "inputs.json"
DEFAULT_MEASUREMENT_INTERVAL = 0
DEFAULT_RANDOM_SEED = 0
DEFAULT_REQUEST_COUNT = 10
DEFAULT_SYNTHETIC_FILENAME = "synthetic_data.json"
DEFAULT_WARMUP_REQUEST_COUNT = 0
DEFAULT_BACKEND = "tensorrtllm"
PAYLOAD_METADATA_FIELDS = ["timestamp", "delay", "session_id"]
PAYLOAD_METADATA_INT_FIELDS = ["timestamp", "delay"]

###########################
# Default Audio Parameters
###########################
DEFAULT_AUDIO_LENGTH_MEAN = 0
DEFAULT_AUDIO_LENGTH_STDDEV = 0
DEFAULT_AUDIO_FORMAT = "wav"
DEFAULT_AUDIO_DEPTHS = [16]
DEFAULT_AUDIO_SAMPLE_RATES = [16]
DEFAULT_AUDIO_NUM_CHANNELS = 1

###########################
# Default Prompt Parameters
###########################
DEFAULT_CORPUS_FILE = "shakespeare.txt"
DEFAULT_STARTING_INDEX = 0
MINIMUM_STARTING_INDEX = 0
DEFAULT_LENGTH = 100
MINIMUM_LENGTH = 1
DEFAULT_TENSORRTLLM_MAX_TOKENS = 256
DEFAULT_PROMPT_TOKENS_MEAN = 550
DEFAULT_PROMPT_TOKENS_STDDEV = 0
DEFAULT_OUTPUT_TOKENS_MEAN = -1
DEFAULT_OUTPUT_TOKENS_STDDEV = 0
DEFAULT_NUM_DATASET_ENTRIES = 100
DEFAULT_NUM_PREFIX_PROMPTS = 0
DEFAULT_PREFIX_PROMPT_LENGTH = 100

###########################
# Default Image Parameters
###########################
DEFAULT_IMAGE_WIDTH_MEAN = 0
DEFAULT_IMAGE_WIDTH_STDDEV = 0
DEFAULT_IMAGE_HEIGHT_MEAN = 0
DEFAULT_IMAGE_HEIGHT_STDDEV = 0

###########################
# Default Session Parameters
###########################
DEFAULT_NUM_SESSIONS = 0
DEFAULT_SESSION_CONCURRENCY = 1
DEFAULT_SESSION_DELAY_RATIO = 1.0
DEFAULT_SESSION_TURN_DELAY_MEAN_MS = 0
DEFAULT_SESSION_TURN_DELAY_STDDEV_MS = 0
DEFAULT_SESSION_TURNS_MEAN = 1
DEFAULT_SESSION_TURNS_STDDEV = 0
