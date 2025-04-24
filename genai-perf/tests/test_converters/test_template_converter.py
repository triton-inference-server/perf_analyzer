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

from unittest.mock import mock_open, patch

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters import TemplateConverter
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)


class TestTemplateConverter:
    @staticmethod
    def create_generic_dataset(rows) -> GenericDataset:
        return GenericDataset(files_data={"file1": FileData(rows)})

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='[{"custom_key": {{ texts|tojson }} }]',
    )
    def test_check_config_with_file_template(self, mock_open_fn):
        fake_template_path = "/fake/path/template.jinja2"

        with patch("os.path.isfile", return_value=True):
            config = ConfigCommand({"model_name": "test_model"})
            config.endpoint.output_format = OutputFormat.TEMPLATE
            config.input.extra = {"payload_template": fake_template_path}

            converter = TemplateConverter(config)
            converter.check_config()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{ "invalid_format": {{ texts|tojson ]] }',
    )
    def test_check_config_invalid_template(self, mock_open_fn):
        fake_template_path = "/fake/path/invalid_template.jinja2"

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.output_format = OutputFormat.TEMPLATE
        config.input.extra = {"payload_template": fake_template_path}

        template_converter = TemplateConverter(config)

        with patch("os.path.isfile", return_value=True):
            with pytest.raises(GenAIPerfException, match="unexpected ']'"):
                template_converter.check_config()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{ "invalid_format": {{ texts|tojson }} }',
    )
    def test_check_config_invalid_type_conversion(self, mock_open_fn):
        fake_template_path = "/fake/path/invalid_template.jinja2"

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.output_format = OutputFormat.TEMPLATE
        config.input.extra = {"payload_template": fake_template_path}

        converter = TemplateConverter(config)

        with patch("os.path.isfile", return_value=True):
            with pytest.raises(
                GenAIPerfException,
                match="Template does not render a list of strings to a list of items",
            ):
                converter.check_config()

    def test_check_config_missing_payload_template(self):
        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.output_format = OutputFormat.TEMPLATE

        converter = TemplateConverter(config)

        with pytest.raises(
            GenAIPerfException,
            match="The template converter requires the extra input payload_template",
        ):
            converter.check_config()

    def test_check_config_with_named_template(self):
        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.output_format = OutputFormat.TEMPLATE
        config.input.extra = {"payload_template": "nv-embedqa"}

        converter = TemplateConverter(config)
        converter.check_config()

    def test_check_config_with_nonexistent_template_file(self):
        fake_template_path = "/nonexistent/path/template.jinja2"

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.output_format = OutputFormat.TEMPLATE
        config.input.extra = {"payload_template": fake_template_path}

        converter = TemplateConverter(config)

        with pytest.raises(
            GenAIPerfException,
            match=f"Template file not found: {fake_template_path}",
        ):
            with patch("os.path.isfile", return_value=False):  # Simulate missing file
                converter.check_config()

    @patch("builtins.open", side_effect=IOError("File read error"))
    def test_check_config_with_unreadable_template_file(self, mock_open_fn):
        fake_template_path = "/fake/path/template.jinja2"

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.output_format = OutputFormat.TEMPLATE
        config.input.extra = {"payload_template": fake_template_path}

        converter = TemplateConverter(config)

        with pytest.raises(
            GenAIPerfException, match="Error reading template file: File read error"
        ):
            with patch("os.path.isfile", return_value=True):  # Simulate file exists
                converter.check_config()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{ "invalid_format": {{ texts|tojson }} }',
    )
    def test_check_config_with_invalid_extra_inputs(self, mock_open_fn):
        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.output_format = OutputFormat.TEMPLATE
        config.input.extra = {
            "invalid_key": "value",
            "payload_template": "template.jinja2",
        }

        converter = TemplateConverter(config)

        with pytest.raises(
            GenAIPerfException,
            match="Template only supports the extra input 'payload_template'",
        ):
            converter.check_config()

    @patch("builtins.open", new_callable=mock_open)
    def test_convert_custom_template(self, mock_open_fn):
        template_content = """[{ "dummy": {{ texts|tojson }} }]"""
        mock_open_fn.return_value.read.return_value = template_content

        with patch("os.path.isfile", return_value=True):
            fake_template_path = "/fake/path/template.jinja2"

            generic_dataset = self.create_generic_dataset(
                [
                    DataRow(texts=["sample_prompt_1"]),
                    DataRow(texts=["sample_prompt_2", "sample_prompt_3"]),
                ]
            )

            config = ConfigCommand({"model_name": "test_model"})
            config.endpoint.output_format = OutputFormat.TEMPLATE
            config.input.extra = {"payload_template": fake_template_path}

            converter = TemplateConverter(config)
            result = converter.convert(generic_dataset)

            expected_result = {
                "data": [
                    {"dummy": ["sample_prompt_1"]},
                    {"dummy": ["sample_prompt_2", "sample_prompt_3"]},
                ]
            }

            assert (
                result == expected_result
            ), f"Expected {expected_result}, but got {result}"

    def test_convert_named_template_nvembedqa(self):
        generic_dataset = self.create_generic_dataset(
            [
                DataRow(texts=["sample_prompt_1"]),
                DataRow(texts=["sample_prompt_2", "sample_prompt_3"]),
            ]
        )

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.output_format = OutputFormat.TEMPLATE
        config.input.extra = {"payload_template": "nv-embedqa"}

        converter = TemplateConverter(config)
        result = converter.convert(generic_dataset)
        expected_result = {
            "data": [
                {"text": ["sample_prompt_1"]},
                {"text": ["sample_prompt_2"]},
                {"text": ["sample_prompt_3"]},
            ]
        }
        assert result == expected_result

    @patch("builtins.open", new_callable=mock_open)
    def test_render_template_file(self, mock_open_fn):

        fake_template_content = '[{"custom_key": {{ texts|tojson }} }]'
        with patch("builtins.open", mock_open(read_data=fake_template_content)), patch(
            "os.path.isfile", return_value=True
        ):
            converter = TemplateConverter(ConfigCommand({"model_names": "test_model"}))
            template = converter.resolve_template("/path/to/template.jinja2")
            assert template.render(texts=["sample"]) == '[{"custom_key": ["sample"] }]'
