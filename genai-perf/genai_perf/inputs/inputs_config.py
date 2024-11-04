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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from genai_perf.inputs.input_constants import (
    DEFAULT_IMAGE_HEIGHT_MEAN,
    DEFAULT_IMAGE_HEIGHT_STDDEV,
    DEFAULT_IMAGE_WIDTH_MEAN,
    DEFAULT_IMAGE_WIDTH_STDDEV,
    DEFAULT_LENGTH,
    DEFAULT_NUM_PROMPTS,
    DEFAULT_OUTPUT_TOKENS_MEAN,
    DEFAULT_OUTPUT_TOKENS_STDDEV,
    DEFAULT_PROMPT_TOKENS_MEAN,
    DEFAULT_PROMPT_TOKENS_STDDEV,
    DEFAULT_RANDOM_SEED,
    DEFAULT_STARTING_INDEX,
    ModelSelectionStrategy,
    OutputFormat,
    PromptSource,
)
from genai_perf.inputs.retrievers.synthetic_image_generator import ImageFormat
from genai_perf.tokenizer import Tokenizer


@dataclass
class InputsConfig:
    """
    A class to hold all of the arguments used for creating the inputs
    """

    ####################
    # General Parameters
    ####################

    # The tokenizer to use when generating synthetic prompts
    tokenizer: Tokenizer

    # If true, adds a steam field to each payload
    add_stream: bool = False

    # The number of image inputs per request
    batch_size_image: int = 1

    # The number of text inputs per request
    batch_size_text: int = 1

    # If provided, append these inputs to every request
    extra_inputs: Dict = field(default_factory=dict)

    # The filename where the input data is available
    input_filename: Optional[Path] = Path("")

    # The filenames used for synthetic data generation
    synthetic_input_filenames: Optional[List[str]] = field(default_factory=list)

    # The compression format of the images.
    image_format: ImageFormat = ImageFormat.PNG

    # The mean height of images when generating synthetic image data.
    image_height_mean: int = DEFAULT_IMAGE_HEIGHT_MEAN

    # The standard deviation of height of images when generating synthetic image data.
    image_height_stddev: int = DEFAULT_IMAGE_HEIGHT_STDDEV

    # The mean width of images when generating synthetic image data.
    image_width_mean: int = DEFAULT_IMAGE_WIDTH_MEAN

    # The standard deviation of width of images when generating synthetic image data.
    image_width_stddev: int = DEFAULT_IMAGE_WIDTH_STDDEV

    # Specify how the input is received
    input_type: PromptSource = PromptSource.SYNTHETIC

    # Number of entries to gather
    length: int = DEFAULT_LENGTH

    # The model name
    model_name: List[str] = field(default_factory=list)

    # The strategy to use when selecting models when multiple models are provided
    model_selection_strategy: ModelSelectionStrategy = (
        ModelSelectionStrategy.ROUND_ROBIN
    )

    # Specify the output format
    output_format: OutputFormat = OutputFormat.TENSORRTLLM

    # If true, the output tokens will set the minimum and maximum tokens to be equivalent.
    output_tokens_deterministic: bool = False

    # The mean length of the output to generate. If not using fixed output lengths, this should be set to -1.
    output_tokens_mean: int = DEFAULT_OUTPUT_TOKENS_MEAN

    # The standard deviation of the length of the output to generate. This is only used if output_tokens_mean is provided.
    output_tokens_stddev: int = DEFAULT_OUTPUT_TOKENS_STDDEV

    # Offset from within the list to start gathering inputs
    starting_index: int = DEFAULT_STARTING_INDEX

    # The directory where all arifacts are saved
    output_dir: Path = Path("")

    ########################################
    # Synthetic Prompt Generation Parameters
    ########################################

    # The number of synthetic output prompts to generate
    num_prompts: int = DEFAULT_NUM_PROMPTS

    # The mean length of the prompt to generate
    prompt_tokens_mean: int = DEFAULT_PROMPT_TOKENS_MEAN

    # The standard deviation of the length of the prompt to generate
    prompt_tokens_stddev: int = DEFAULT_PROMPT_TOKENS_STDDEV

    # Seed used to generate random values
    random_seed: int = DEFAULT_RANDOM_SEED
