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

import base64
import io
import random

import numpy as np
import pytest
import soundfile as sf
from genai_perf.config.input.config_input import ConfigAudio
from genai_perf.inputs.input_constants import AudioFormat
from genai_perf.inputs.retrievers.synthetic_audio_generator import (
    SyntheticAudioGenerator,
)


def decode_audio(data_uri: str) -> tuple[np.ndarray, int]:
    """Helper function to decode audio from data URI format.

    Args:
        data_uri: Data URI string in format "format,b64_data"

    Returns:
        Tuple of (audio_data: np.ndarray, sample_rate: int)
    """
    # Parse data URI
    _, b64_data = data_uri.split(",")
    decoded_data = base64.b64decode(b64_data)

    # Load audio using soundfile - format is auto-detected from content
    audio_data, sample_rate = sf.read(io.BytesIO(decoded_data))
    return audio_data, sample_rate


@pytest.mark.parametrize(
    "expected_audio_length",
    [
        1.0,
        2.0,
    ],
)
def test_different_audio_length(expected_audio_length):
    config_audio = ConfigAudio()
    config_audio.length.mean = expected_audio_length
    config_audio.length.stddev = 0
    config_audio.sample_rates = [44]  # Fixed sampling rate for test
    config_audio.depths = [16]  # Fixed bit depth for test
    config_audio.format = AudioFormat.WAV

    data_uri = SyntheticAudioGenerator.create_synthetic_audio(config_audio)

    audio_data, sample_rate = decode_audio(data_uri)
    actual_length = len(audio_data) / sample_rate
    assert (
        abs(actual_length - expected_audio_length) < 0.1
    ), "audio length not as expected"


def test_negative_length_raises_error():
    with pytest.raises(ValueError):
        config_audio = ConfigAudio()
        config_audio.length.mean = 0.05  # Below minimum of 0.1
        config_audio.length.stddev = 0
        config_audio.sample_rates = [44]
        config_audio.depths = [16]
        config_audio.format = AudioFormat.WAV

        SyntheticAudioGenerator.create_synthetic_audio(config_audio)


@pytest.mark.parametrize(
    "length_mean, length_stddev, sampling_rate_khz, bit_depth",
    [
        (1.0, 0.1, 44, 16),
        (2.0, 0.2, 48, 24),
    ],
)
def test_generator_deterministic(
    length_mean, length_stddev, sampling_rate_khz, bit_depth
):
    np.random.seed(123)
    random.seed(123)

    config_audio = ConfigAudio()
    config_audio.length.mean = length_mean
    config_audio.length.stddev = length_stddev
    config_audio.sample_rates = [sampling_rate_khz]
    config_audio.depths = [bit_depth]
    config_audio.format = AudioFormat.WAV

    data_uri1 = SyntheticAudioGenerator.create_synthetic_audio(config_audio)

    np.random.seed(123)
    random.seed(123)
    data_uri2 = SyntheticAudioGenerator.create_synthetic_audio(config_audio)

    # Compare the actual audio data
    audio_data1, _ = decode_audio(data_uri1)
    audio_data2, _ = decode_audio(data_uri2)
    assert np.array_equal(audio_data1, audio_data2), "generator is nondeterministic"


@pytest.mark.parametrize("audio_format", [AudioFormat.WAV, AudioFormat.MP3])
def test_audio_format(audio_format):
    # use sample rate supported by all formats (44.1kHz)
    config_audio = ConfigAudio()
    config_audio.length.mean = 1.0
    config_audio.length.stddev = 0
    config_audio.sample_rates = [44.1]
    config_audio.depths = [16]  # Fixed bit depth for test
    config_audio.format = audio_format

    data_uri = SyntheticAudioGenerator.create_synthetic_audio(config_audio)

    # Check data URI format
    assert data_uri.startswith(
        f"{audio_format.name.lower()},"
    ), "incorrect data URI format"

    # Verify the audio can be decoded
    audio_data, _ = decode_audio(data_uri)
    assert len(audio_data) > 0, "audio data is empty"


def test_unsupported_bit_depth():
    with pytest.raises(ValueError) as exc_info:
        config_audio = ConfigAudio()
        config_audio.length.mean = 1.0
        config_audio.length.stddev = 0
        config_audio.sample_rates = [44]
        config_audio.depths = [12]  # Unsupported bit depth
        config_audio.format = AudioFormat.WAV

        SyntheticAudioGenerator.create_synthetic_audio(config_audio)

    assert "Supported bit depths are:" in str(exc_info.value)


@pytest.mark.parametrize("channels", [1, 2])
def test_channels(channels):
    config_audio = ConfigAudio()
    config_audio.length.mean = 1.0
    config_audio.length.stddev = 0
    config_audio.sample_rates = [44]
    config_audio.depths = [16]
    config_audio.format = AudioFormat.WAV
    config_audio.num_channels = channels

    data_uri = SyntheticAudioGenerator.create_synthetic_audio(config_audio)

    audio_data, _ = decode_audio(data_uri)
    if channels == 1:
        assert len(audio_data.shape) == 1, "mono audio should be 1D array"
    else:
        assert len(audio_data.shape) == 2, "stereo audio should be 2D array"
        assert audio_data.shape[1] == 2, "stereo audio should have 2 channels"


@pytest.mark.parametrize(
    "sampling_rate_khz, bit_depth",
    [
        (44.1, 16),  # Common CD quality
        (48, 24),  # Studio quality
        (96, 32),  # High-res audio
    ],
)
def test_audio_parameters(sampling_rate_khz, bit_depth):
    config_audio = ConfigAudio()
    config_audio.length.mean = 1.0
    config_audio.length.stddev = 0
    config_audio.sample_rates = [sampling_rate_khz]
    config_audio.depths = [bit_depth]
    config_audio.format = AudioFormat.WAV

    data_uri = SyntheticAudioGenerator.create_synthetic_audio(config_audio)

    _, sample_rate = decode_audio(data_uri)
    assert sample_rate == sampling_rate_khz * 1000, "unexpected sampling rate"


def test_mp3_unsupported_sampling_rate():
    with pytest.raises(ValueError) as exc_info:
        config_audio = ConfigAudio()
        config_audio.length.mean = 1.0
        config_audio.length.stddev = 0
        config_audio.sample_rates = [96]  # 96kHz is not supported for MP3
        config_audio.depths = [16]
        config_audio.format = AudioFormat.MP3

        SyntheticAudioGenerator.create_synthetic_audio(config_audio)
    assert "MP3 format only supports" in str(
        exc_info.value
    ), "error message should mention supported rates"


def test_positive_normal_sampling():
    mean = 1.0
    stddev = 0.2
    min_value = 0.1
    samples = [
        SyntheticAudioGenerator._sample_positive_normal(mean, stddev, min_value)
        for _ in range(1000)
    ]

    assert all(s >= min_value for s in samples), "samples below minimum value"
    assert (
        abs(np.mean(samples) - mean) < 0.1
    ), "mean significantly different from expected"
