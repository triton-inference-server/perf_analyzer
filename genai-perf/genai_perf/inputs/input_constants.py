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

from enum import Enum, auto
from typing import Dict


class ModelSelectionStrategy(Enum):
    ROUND_ROBIN = auto()
    RANDOM = auto()


class PromptSource(Enum):
    SYNTHETIC = auto()
    FILE = auto()


class OutputFormat(Enum):
    ################################################################
    # Triton backends
    ################################################################
    TENSORRTLLM = auto()
    VLLM = auto()
    ################################################################
    # Other output formats
    ################################################################
    IMAGE_RETRIEVAL = auto()
    NVCLIP = auto()
    OPENAI_CHAT_COMPLETIONS = auto()
    OPENAI_COMPLETIONS = auto()
    OPENAI_EMBEDDINGS = auto()
    OPENAI_VISION = auto()
    RANKINGS = auto()
    TENSORRTLLM_ENGINE = auto()

    def to_lowercase(self):
        return self.name.lower()


###########################
# General Parameters
###########################
DEFAULT_INPUT_DATA_JSON = "inputs.json"
DEFAULT_RANDOM_SEED = 0
DEFAULT_SYNTHETIC_FILENAME = "synthetic_data.json"


###########################
# Default Prompt Parameters
###########################
DEFAULT_STARTING_INDEX = 0
MINIMUM_STARTING_INDEX = 0
DEFAULT_LENGTH = 100
MINIMUM_LENGTH = 1
DEFAULT_TENSORRTLLM_MAX_TOKENS = 256
DEFAULT_BATCH_SIZE = 1
DEFAULT_PROMPT_TOKENS_MEAN = 550
DEFAULT_PROMPT_TOKENS_STDDEV = 0
DEFAULT_OUTPUT_TOKENS_MEAN = -1
DEFAULT_OUTPUT_TOKENS_STDDEV = 0
DEFAULT_NUM_PROMPTS = 100

###########################
# Default Image Parameters
###########################
DEFAULT_IMAGE_WIDTH_MEAN = 100
DEFAULT_IMAGE_WIDTH_STDDEV = 0
DEFAULT_IMAGE_HEIGHT_MEAN = 100
DEFAULT_IMAGE_HEIGHT_STDDEV = 0
