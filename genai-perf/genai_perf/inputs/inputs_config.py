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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from genai_perf.inputs import input_constants as ic
from genai_perf.inputs.retrievers import AudioFormat, ImageFormat
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

    # The number of audio inputs per request
    batch_size_audio: int = ic.DEFAULT_BATCH_SIZE

    # The number of image inputs per request
    batch_size_image: int = ic.DEFAULT_BATCH_SIZE

    # The number of text inputs per request
    batch_size_text: int = ic.DEFAULT_BATCH_SIZE

    # If provided, append these inputs to every request
    extra_inputs: Dict[str, Any] = field(default_factory=dict)

    # The filename where the input data is available
    input_filename: Optional[Path] = Path("")

    # The filenames used for synthetic data generation
    synthetic_input_filenames: Optional[List[str]] = field(default_factory=list)

    # The filename where payload input data is available
    payload_input_filename: Optional[Path] = Path("")

    # Specify how the input is received
    input_type: ic.PromptSource = ic.PromptSource.SYNTHETIC

    # Number of entries to gather
    length: int = ic.DEFAULT_LENGTH

    # The model name
    model_name: List[str] = field(default_factory=list)

    # The strategy to use when selecting models when multiple models are provided
    model_selection_strategy: ic.ModelSelectionStrategy = (
        ic.ModelSelectionStrategy.ROUND_ROBIN
    )

    # Specify the output format
    output_format: ic.OutputFormat = ic.OutputFormat.TENSORRTLLM

    # If true, the output tokens will set the minimum and maximum tokens to be equivalent.
    output_tokens_deterministic: bool = False

    # The mean length of the output to generate. If not using fixed output lengths, this should be set to -1.
    output_tokens_mean: int = ic.DEFAULT_OUTPUT_TOKENS_MEAN

    # The standard deviation of the length of the output to generate. This is only used if output_tokens_mean is provided.
    output_tokens_stddev: int = ic.DEFAULT_OUTPUT_TOKENS_STDDEV

    # Offset from within the list to start gathering inputs
    starting_index: int = ic.DEFAULT_STARTING_INDEX

    # The directory where all arifacts are saved
    output_dir: Path = Path("")

    ########################################
    # Synthetic Prompt Generation Parameters
    ########################################

    # The number of dataset entries to generate and use as the payload pool
    num_dataset_entries: int = ic.DEFAULT_NUM_DATASET_ENTRIES

    # The mean length of the prompt to generate
    prompt_tokens_mean: int = ic.DEFAULT_PROMPT_TOKENS_MEAN

    # The standard deviation of the length of the prompt to generate
    prompt_tokens_stddev: int = ic.DEFAULT_PROMPT_TOKENS_STDDEV

    # Seed used to generate random values
    random_seed: int = ic.DEFAULT_RANDOM_SEED

    # The number of prefix prompts to generate and pool from
    num_prefix_prompts: int = ic.DEFAULT_NUM_PREFIX_PROMPTS

    # The length of the prefix prompts to generate
    prefix_prompt_length: int = ic.DEFAULT_PREFIX_PROMPT_LENGTH

    # The number of sessions to generate
    num_sessions: int = ic.DEFAULT_NUM_SESSIONS

    # The mean number of turns per session
    session_turns_mean: int = ic.DEFAULT_SESSION_TURNS_MEAN

    # The standard deviation of the number of turns per session
    session_turns_stddev: int = ic.DEFAULT_SESSION_TURNS_STDDEV

    # The mean delay between turns in a session
    session_turn_delay_mean: int = ic.DEFAULT_SESSION_TURN_DELAY_MEAN_MS

    # The standard deviation of the delay between turns in a session
    session_turn_delay_stddev: int = ic.DEFAULT_SESSION_TURN_DELAY_STDDEV_MS

    #######################################
    # Synthetic Audio Generation Parameters
    #######################################

    # The mean length of the audio to generate in seconds.
    audio_length_mean: float = ic.DEFAULT_AUDIO_LENGTH_MEAN

    # The standard deviation of the length of the audio to generate in seconds.
    audio_length_stddev: float = ic.DEFAULT_AUDIO_LENGTH_STDDEV

    # The sampling rates of the audio to generate in kHz.
    audio_sample_rates: List[float] = field(default_factory=list)

    # The bit depths of the audio to generate.
    audio_depths: List[int] = field(default_factory=list)

    # The number of channels of the audio to generate.
    audio_num_channels: int = ic.DEFAULT_AUDIO_NUM_CHANNELS

    # The format of the audio to generate.
    audio_format: AudioFormat = AudioFormat.WAV

    #######################################
    # Synthetic Image Generation Parameters
    #######################################

    # The compression format of the images.
    image_format: ImageFormat = ImageFormat.PNG

    # The mean height of images when generating synthetic image data.
    image_height_mean: int = ic.DEFAULT_IMAGE_HEIGHT_MEAN

    # The standard deviation of height of images when generating synthetic image data.
    image_height_stddev: int = ic.DEFAULT_IMAGE_HEIGHT_STDDEV

    # The mean width of images when generating synthetic image data.
    image_width_mean: int = ic.DEFAULT_IMAGE_WIDTH_MEAN

    # The standard deviation of width of images when generating synthetic image data.
    image_width_stddev: int = ic.DEFAULT_IMAGE_WIDTH_STDDEV
