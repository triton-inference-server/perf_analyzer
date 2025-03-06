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

import base64
import random
import io
import pytest
import numpy as np
import soundfile as sf
from genai_perf.inputs.retrievers.synthetic_audio_generator import (
    AudioFormat,
    SyntheticAudioGenerator,
)


def decode_audio(audio_dict):
    """Helper function to decode audio from the OpenAI API format"""
    assert audio_dict["type"] == "input_audio"
    audio_data = audio_dict["input_audio"]
    decoded_data = base64.b64decode(audio_data["data"])
    
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
    audio_dict = SyntheticAudioGenerator.create_synthetic_audio(
        audio_length_mean=expected_audio_length,
        audio_length_stddev=0,
        sampling_rates_khz=[44],  # Fixed sampling rate for test
        bit_depths=[16],  # Fixed bit depth for test
        output_format="wav",
    )

    audio_data, sample_rate = decode_audio(audio_dict)
    actual_length = len(audio_data) / sample_rate
    assert abs(actual_length - expected_audio_length) < 0.1, "audio length not as expected"


def test_negative_length_raises_error():
    with pytest.raises(ValueError):
        SyntheticAudioGenerator.create_synthetic_audio(
            audio_length_mean=0.05,  # Below minimum of 0.1
            audio_length_stddev=0,
            sampling_rates_khz=[44],
            bit_depths=[16],
            output_format="wav",
        )


@pytest.mark.parametrize(
    "length_mean, length_stddev, sampling_rate_khz, bit_depth",
    [
        (1.0, 0.1, 44, 16),
        (2.0, 0.2, 48, 24),
    ],
)
def test_generator_deterministic(length_mean, length_stddev, sampling_rate_khz, bit_depth):
    np.random.seed(123)
    random.seed(123)
    audio1 = SyntheticAudioGenerator.create_synthetic_audio(
        audio_length_mean=length_mean,
        audio_length_stddev=length_stddev,
        sampling_rates_khz=[sampling_rate_khz],
        bit_depths=[bit_depth],
        output_format="wav",
    )

    np.random.seed(123)
    random.seed(123)
    audio2 = SyntheticAudioGenerator.create_synthetic_audio(
        audio_length_mean=length_mean,
        audio_length_stddev=length_stddev,
        sampling_rates_khz=[sampling_rate_khz],
        bit_depths=[bit_depth],
        output_format="wav",
    )

    # Compare the actual audio data
    audio_data1, _ = decode_audio(audio1)
    audio_data2, _ = decode_audio(audio2)
    assert np.array_equal(audio_data1, audio_data2), "generator is nondeterministic"


@pytest.mark.parametrize("output_format", ["wav", "mp3"])
def test_audio_format(output_format):
    # use sample rate supported by all formats (44.1kHz)
    sampling_rate = 44.1
    audio_dict = SyntheticAudioGenerator.create_synthetic_audio(
        audio_length_mean=1.0,
        audio_length_stddev=0,
        sampling_rates_khz=[sampling_rate],
        bit_depths=[16],
        output_format=output_format,
    )

    # Check dictionary structure
    assert audio_dict["type"] == "input_audio"
    assert audio_dict["input_audio"]["format"] == output_format
    
    # Verify the audio can be decoded
    audio_data, _ = decode_audio(audio_dict)
    assert len(audio_data) > 0, "audio data is empty"


@pytest.mark.parametrize(
    "sampling_rate_khz, bit_depth",
    [
        (44, 16),
        (48, 24),
        (96, 32),
    ],
)
def test_audio_parameters(sampling_rate_khz, bit_depth):
    audio_dict = SyntheticAudioGenerator.create_synthetic_audio(
        audio_length_mean=1.0,
        audio_length_stddev=0,
        sampling_rates_khz=[sampling_rate_khz],
        bit_depths=[bit_depth],
        output_format="wav",
    )

    _, sample_rate = decode_audio(audio_dict)
    assert sample_rate == sampling_rate_khz * 1000, "unexpected sampling rate"


def test_mp3_unsupported_sampling_rate():
    with pytest.raises(ValueError) as exc_info:
        SyntheticAudioGenerator.create_synthetic_audio(
            audio_length_mean=1.0,
            audio_length_stddev=0,
            sampling_rates_khz=[96],  # 96kHz is not supported for MP3
            bit_depths=[16],
            output_format="mp3",
        )
    assert "MP3 format only supports" in str(exc_info.value), "error message should mention supported rates"


def test_positive_normal_sampling():
    mean = 1.0
    stddev = 0.2
    min_value = 0.1
    samples = [
        SyntheticAudioGenerator._sample_positive_normal(mean, stddev, min_value)
        for _ in range(1000)
    ]
    
    assert all(s >= min_value for s in samples), "samples below minimum value"
    assert abs(np.mean(samples) - mean) < 0.1, "mean significantly different from expected" 