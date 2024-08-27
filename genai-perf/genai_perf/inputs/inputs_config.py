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
from typing import Dict, List

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
from genai_perf.inputs.synthetic_image_generator import ImageFormat
from genai_perf.tokenizer import DEFAULT_TOKENIZER, Tokenizer, get_tokenizer


@dataclass
class InputsConfig:
    """
    A class to hold all of the arguments used for creating the inputs
    """

    ####################
    # General Parameters
    ####################

    # add_stream:
    #       If true, adds a steam field to each payload
    add_stream: bool = False

    # batch_size:
    #       The number of inputs per request (currently only used for the embeddings, image retrieval, and rankings endpoints)
    batch_size: int = 1

    # dataset_name:
    #           The name of the dataset
    dataset_name: str = ""
    # extra_inputs:
    #       If provided, append these inputs to every request
    extra_inputs: Dict = field(default_factory=dict)

    # input_filename:
    #       The filename where the input data is available
    input_filename: Path = Path("")

    # image_format:
    #       The compression format of the images.
    image_format: ImageFormat = ImageFormat.PNG

    # image_height_mean:
    #       The mean height of images when generating synthetic image data.
    image_height_mean: int = DEFAULT_IMAGE_HEIGHT_MEAN

    # image_height_stddev:
    #       The standard deviation of height of images when generating synthetic image data.
    image_height_stddev: int = DEFAULT_IMAGE_HEIGHT_STDDEV

    # image_width_mean:
    #       The mean width of images when generating synthetic image data.
    image_width_mean: int = DEFAULT_IMAGE_WIDTH_MEAN

    # image_width_stddev:
    #       The standard deviation of width of images when generating synthetic image data.
    image_width_stddev: int = DEFAULT_IMAGE_WIDTH_STDDEV

    # input_type:
    #       Specify how the input is received
    input_type: PromptSource = PromptSource.SYNTHETIC

    # length:
    #       Number of entries to gather
    length: int = DEFAULT_LENGTH

    # model_name:
    #       The model name
    model_name: List[str] = field(default_factory=list)

    # model_selection_strategy:
    #       The strategy to use when selecting models when multiple models are provided
    model_selection_strategy: ModelSelectionStrategy = (
        ModelSelectionStrategy.ROUND_ROBIN
    )
    # output_format:
    #       Specify the output format
    output_format: OutputFormat = OutputFormat.TENSORRTLLM

    # output_tokens_deterministic:
    #           If true, the output tokens will set the minimum and maximum tokens to be equivalent.
    output_tokens_deterministic: bool = False

    # output_tokens_mean:
    #       The mean length of the output to generate. If not using fixed output lengths, this should be set to -1.
    output_tokens_mean: int = DEFAULT_OUTPUT_TOKENS_MEAN

    # output_tokens_stddev:
    #       The standard deviation of the length of the output to generate. This is only used if output_tokens_mean is provided.
    output_tokens_stddev: int = DEFAULT_OUTPUT_TOKENS_STDDEV

    # starting_index:
    #       Offset from within the list to start gathering inputs
    starting_index: int = DEFAULT_STARTING_INDEX

    # output_dir:
    #       The directory where all arifacts are saved
    output_dir: Path = Path("")

    ########################################
    # Synthetic Prompt Generation Parameters
    ########################################

    # num_prompts:
    #       The number of synthetic output prompts to generate
    num_prompts: int = DEFAULT_NUM_PROMPTS

    # prompt_tokens_mean:
    #       The mean length of the prompt to generate
    prompt_tokens_mean: int = DEFAULT_PROMPT_TOKENS_MEAN

    # prompt_tokens_stddev:
    #       The standard deviation of the length of the prompt to generate
    prompt_tokens_stddev: int = DEFAULT_PROMPT_TOKENS_STDDEV

    # random_seed:
    #       Seed used to generate random values
    random_seed: int = DEFAULT_RANDOM_SEED

    # tokenizer:
    #      The tokenizer to use when generating synthetic prompts
    tokenizer: Tokenizer = get_tokenizer(DEFAULT_TOKENIZER)
