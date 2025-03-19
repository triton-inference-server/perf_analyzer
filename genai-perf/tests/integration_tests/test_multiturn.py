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
import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.inputs.input_constants import PromptSource
from genai_perf.inputs.inputs import Inputs
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.payload_input_retriever import PayloadInputRetriever
from genai_perf.tokenizer import get_empty_tokenizer


class TestIntegrationMultiTurn:
    @pytest.fixture
    def mock_config(self):
        class MockConfig(InputsConfig):
            def __init__(self):
                self.tokenizer = get_empty_tokenizer()
                self.config = ConfigCommand({"model_name": "test_model"})
                self.output_directory = Path("test_output")

        return MockConfig()

    @pytest.fixture
    def retriever(self, mock_config):
        return PayloadInputRetriever(mock_config)

    def test_multi_turn_input_generation(self, monkeypatch, mock_config):
        input_data = (
            '{"text": "Turn 1", "session_id": "abc", "delay": 2}\n'
            '{"text": "Turn 2", "session_id": "abc", "delay": 3}\n'
            '{"text": "Turn 3", "session_id": "abc", "delay": 5000}\n'
        )
        mock_file_open = mock_open(read_data=input_data)

        inputs = Inputs(mock_config)
        output_file_path = mock_config.output_directory / "inputs.json"

        with patch("builtins.open", mock_file_open) as mocked_open, patch(
            "pathlib.Path.exists", return_value=True
        ):
            inputs.create_inputs()

            mocked_open.assert_called_with(str(output_file_path), "w")

            handle = mocked_open()
            written_content = "".join(
                call.args[0] for call in handle.write.call_args_list
            )
            output_data = json.loads(written_content)

            assert isinstance(
                output_data, dict
            ), f"Expected dict, got: {type(output_data)}"
            assert "data" in output_data, f"Unexpected JSON structure: {output_data}"

            for entry in output_data["data"]:
                assert "session_id" in entry, "Session ID missing in output entry"
                assert "delay" in entry, "Delay must be present in output entry"
