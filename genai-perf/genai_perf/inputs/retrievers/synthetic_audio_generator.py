# Copyright 2025, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import soundfile as sf
from genai_perf.config.input.config_input import ConfigAudio
from genai_perf.inputs.input_constants import AudioFormat

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

# Supported bit depths and their corresponding numpy types
SUPPORTED_BIT_DEPTHS = {
    8: (np.int8, "PCM_S8"),
    16: (np.int16, "PCM_16"),
    24: (np.int32, "PCM_24"),  # soundfile handles 24-bit as 32-bit
    32: (np.int32, "PCM_32"),
}


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
    def _validate_sampling_rate(sampling_rate: int, audio_format: AudioFormat) -> None:
        """
        Validate sampling rate for the given output format.

        Args:
            sampling_rate: Sampling rate in Hz
            audio_format: Audio format

        Raises:
            ValueError: If sampling rate is not supported for the given format
        """
        if (
            audio_format == AudioFormat.MP3
            and sampling_rate not in MP3_SUPPORTED_SAMPLE_RATES
        ):
            supported_rates = sorted(MP3_SUPPORTED_SAMPLE_RATES)
            raise ValueError(
                f"MP3 format only supports the following sample rates (in Hz): {supported_rates}. "
                f"Got {sampling_rate} Hz. Please choose a supported rate from the list."
            )

    @staticmethod
    def _validate_bit_depth(bit_depth: int) -> None:
        """
        Validate bit depth is supported.

        Args:
            bit_depth: Bit depth in bits

        Raises:
            ValueError: If bit depth is not supported
        """
        if bit_depth not in SUPPORTED_BIT_DEPTHS:
            supported_depths = sorted(SUPPORTED_BIT_DEPTHS.keys())
            raise ValueError(
                f"Unsupported bit depth: {bit_depth}. "
                f"Supported bit depths are: {supported_depths}"
            )

    @staticmethod
    def create_synthetic_audio(config: ConfigAudio) -> str:
        """
        Generate synthetic audio data with specified parameters.

        Args:
            config: ConfigAudio object containing audio generation parameters

        Returns:
            Data URI containing base64-encoded audio data with format specification

        Raises:
            ValueError: If any of the following conditions are met:
                - audio_length_mean is less than 0.1 seconds
                - channels is not 1 (mono) or 2 (stereo)
                - sampling rate is not supported for MP3 format
                - bit depth is not supported (must be 8, 16, 24, or 32)
                - audio format is not supported (must be 'wav' or 'mp3')
        """
        if config.num_channels not in (1, 2):
            raise ValueError("Only mono (1) and stereo (2) channels are supported")

        # Sample audio length (in seconds) using rejection sampling
        audio_length = SyntheticAudioGenerator._sample_positive_normal(
            config.length.mean, config.length.stddev
        )

        # Randomly select sampling rate and bit depth
        sampling_rate = int(
            np.random.choice(config.sample_rates) * 1000
        )  # Convert kHz to Hz
        bit_depth = np.random.choice(config.depths)

        # Validate sampling rate and bit depth
        SyntheticAudioGenerator._validate_sampling_rate(sampling_rate, config.format)
        SyntheticAudioGenerator._validate_bit_depth(bit_depth)

        # Generate synthetic audio data (gaussian noise)
        num_samples = int(audio_length * sampling_rate)
        audio_data = np.random.normal(
            0,
            0.3,
            (
                (num_samples, config.num_channels)
                if config.num_channels > 1
                else num_samples
            ),
        )

        # Ensure the signal is within [-1, 1] range
        audio_data = np.clip(audio_data, -1, 1)

        # Scale to the appropriate bit depth range
        max_val = 2 ** (bit_depth - 1) - 1
        numpy_type, _ = SUPPORTED_BIT_DEPTHS[bit_depth]
        audio_data = (audio_data * max_val).astype(numpy_type)

        # Write audio using soundfile
        output_buffer = io.BytesIO()

        # Select appropriate subtype based on format
        if config.format == AudioFormat.MP3:
            subtype = "MPEG_LAYER_III"
        elif config.format == AudioFormat.WAV:
            _, subtype = SUPPORTED_BIT_DEPTHS[bit_depth]
        else:
            raise ValueError(
                f"Unsupported audio format: {config.format.name}. "
                f"Supported formats are: {AudioFormat.WAV.name}, {AudioFormat.MP3.name}"
            )

        sf.write(
            output_buffer,
            audio_data,
            sampling_rate,
            format=config.format.name,
            subtype=subtype,
        )
        audio_bytes = output_buffer.getvalue()

        # Encode to base64 with data URI scheme: "{format},{data}"
        base64_data = base64.b64encode(audio_bytes).decode("utf-8")
        return f"{config.format.name.lower()},{base64_data}"
