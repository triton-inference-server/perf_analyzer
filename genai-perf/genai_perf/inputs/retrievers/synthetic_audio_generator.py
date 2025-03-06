# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import base64
import io
from typing import Dict, List, Literal, TypedDict

import numpy as np
import soundfile as sf

AudioFormat = Literal["wav", "mp3"]

# MP3 supported sample rates in Hz
MP3_SUPPORTED_SAMPLE_RATES = {
    8000,
    11025,
    12000,
    16000,
    22050,
    24000,
    32000,
    44100,
    48000,
}


class InputAudio(TypedDict):
    data: str
    format: AudioFormat


class AudioResponse(TypedDict):
    type: Literal["input_audio"]
    input_audio: InputAudio


class SyntheticAudioGenerator:
    @staticmethod
    def _sample_positive_normal(
        mean: float, stddev: float, min_value: float = 0.1
    ) -> float:
        """
        Sample from a normal distribution ensuring positive values without distorting the distribution.
        Uses rejection sampling to maintain the proper shape of the distribution.

        Args:
            mean: Mean value for the normal distribution
            stddev: Standard deviation for the normal distribution
            min_value: Minimum acceptable value

        Returns:
            A positive sample from the normal distribution
        """
        if mean < min_value:
            raise ValueError(
                f"Mean value ({mean}) must be greater than min_value ({min_value})"
            )

        while True:
            sample = np.random.normal(mean, stddev)
            if sample >= min_value:
                return sample

    @staticmethod
    def _validate_sampling_rate(sampling_rate: int, output_format: AudioFormat) -> None:
        """
        Validate sampling rate for the given output format.

        Args:
            sampling_rate: Sampling rate in Hz
            output_format: Output audio format

        Raises:
            ValueError: If sampling rate is not supported for the given format
        """
        if (
            output_format.lower() == "mp3"
            and sampling_rate not in MP3_SUPPORTED_SAMPLE_RATES
        ):
            supported_rates = sorted(MP3_SUPPORTED_SAMPLE_RATES)
            raise ValueError(
                f"MP3 format only supports the following sample rates (in Hz): {supported_rates}. "
                f"Got {sampling_rate} Hz. Please choose a supported rate from the list."
            )

    @staticmethod
    def create_synthetic_audio(
        audio_length_mean: float,
        audio_length_stddev: float,
        sampling_rates_khz: List[int],
        bit_depths: List[int],
        output_format: AudioFormat = "wav",
    ) -> AudioResponse:
        """
        Generate synthetic audio data with specified parameters.

        Args:
            audio_length_mean: Mean audio length in seconds (must be >= 0.1)
            audio_length_stddev: Standard deviation of audio length in seconds
            sampling_rates_khz: List of allowed sampling rates in kHz
            bit_depths: List of allowed bit depths in bits
            output_format: Output audio format, either "wav" or "mp3"

        Returns:
            Dictionary compatible with OpenAI API input_audio format:
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "<base64-encoded audio>",
                    "format": "<wav or mp3>"
                }
            }

        Raises:
            ValueError: If sampling rate is not supported for the given format
        """
        # Sample audio length (in seconds) using rejection sampling
        audio_length = SyntheticAudioGenerator._sample_positive_normal(
            audio_length_mean, audio_length_stddev
        )

        # Randomly select sampling rate and bit depth
        sampling_rate = int(
            np.random.choice(sampling_rates_khz) * 1000
        )  # Convert kHz to Hz
        bit_depth = np.random.choice(bit_depths)

        # Validate sampling rate for the output format
        SyntheticAudioGenerator._validate_sampling_rate(sampling_rate, output_format)

        # Generate synthetic audio data (gaussian noise)
        num_samples = int(audio_length * sampling_rate)
        audio_data = np.random.normal(0, 0.3, num_samples)

        # Ensure the signal is within [-1, 1] range
        audio_data = np.clip(audio_data, -1, 1)

        # Scale to the appropriate bit depth range
        max_val = 2 ** (bit_depth - 1) - 1
        audio_data = (audio_data * max_val).astype(
            {
                8: np.int8,
                16: np.int16,
                24: np.int32,  # soundfile can handle 24-bit as 32-bit
                32: np.int32,
            }[bit_depth]
        )

        # Write audio using soundfile
        output_buffer = io.BytesIO()

        # Select appropriate subtype based on format
        if output_format.upper() == "MP3":
            subtype = "MPEG_LAYER_III"
        else:
            subtype = {8: "PCM_S8", 16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}[bit_depth]

        sf.write(
            output_buffer,
            audio_data,
            sampling_rate,
            format=output_format.upper(),
            subtype=subtype,
        )
        audio_bytes = output_buffer.getvalue()

        # Encode to base64
        base64_data = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "type": "input_audio",
            "input_audio": {"data": base64_data, "format": output_format},
        }
