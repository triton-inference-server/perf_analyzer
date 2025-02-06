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

from pathlib import Path

import pytest
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters.dynamic_grpc_converter import DynamicGRPCConverter
from genai_perf.inputs.input_constants import DEFAULT_BATCH_SIZE
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.tokenizer import get_empty_tokenizer


@pytest.fixture
def sample_dataset():
    return GenericDataset(
        files_data={
            "file1": FileData(
                rows=[
                    DataRow(texts=["first input"]),
                    DataRow(texts=["second input"]),
                ],
            )
        }
    )


@pytest.fixture
def valid_config():
    return InputsConfig(
        tokenizer=get_empty_tokenizer(),
        batch_size_text=DEFAULT_BATCH_SIZE,
        input_filename=Path("test_input.txt"),
    )


@pytest.fixture
def converter():
    return DynamicGRPCConverter()


def test_convert(converter, sample_dataset, valid_config):
    result = converter.convert(sample_dataset, valid_config)

    expected_result = {
        "data": [
            {"ipc_stream": "first input"},
            {"ipc_stream": "second input"},
        ]
    }

    assert result == expected_result


def test_convert_with_extra_inputs(converter, sample_dataset, valid_config, mocker):
    valid_config.extra_inputs = {"extra_key": "extra_value"}
    mock_add_request_params = mocker.patch.object(
        converter, "_add_request_params", wraps=converter._add_request_params
    )

    result = converter.convert(sample_dataset, valid_config)

    assert (
        mock_add_request_params.call_count == 2
    ), "Expected _add_request_params to be called for each input row."

    expected_result = {
        "data": [
            {"ipc_stream": "first input", "extra_key": "extra_value"},
            {"ipc_stream": "second input", "extra_key": "extra_value"},
        ]
    }
    assert result == expected_result


@pytest.mark.parametrize("input_texts", [["hello"], ["long sentence example"], [""]])
def test_convert_varied_inputs(converter, valid_config, input_texts):
    dataset = GenericDataset(
        files_data={"file1": FileData(rows=[DataRow(texts=input_texts)])}
    )

    result = converter.convert(dataset, valid_config)

    expected_result = {"data": [{"ipc_stream": text} for text in input_texts]}

    assert result == expected_result


def test_convert_large_dataset(converter, valid_config):
    """Test behavior with a large dataset."""
    large_dataset = GenericDataset(
        files_data={
            "file1": FileData(rows=[DataRow(texts=[f"text {i}"]) for i in range(1000)])
        }
    )

    result = converter.convert(large_dataset, valid_config)

    assert len(result["data"]) == 1000, "Large dataset should generate 1000 entries."
    assert result["data"][0] == {
        "ipc_stream": "text 0"
    }, "First entry does not match expected."
    assert result["data"][-1] == {
        "ipc_stream": "text 999"
    }, "Last entry does not match expected."


def test_check_config_valid(converter, valid_config):
    try:
        converter.check_config(valid_config)
    except GenAIPerfException as e:
        pytest.fail(f"check_config() raised an unexpected GenAIPerfException: {e}")


def test_check_config_invalid_batch_size(converter, valid_config):
    valid_config.batch_size_text = [2]
    with pytest.raises(
        GenAIPerfException, match="The --batch-size-text flag is not supported"
    ):
        converter.check_config(valid_config)


def test_check_config_missing_input_filename(converter, valid_config):
    valid_config.input_filename = ""
    with pytest.raises(
        GenAIPerfException,
        match="The dynamic GRPC converter only supports the input file path.",
    ):
        converter.check_config(valid_config)
