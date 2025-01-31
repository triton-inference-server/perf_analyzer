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

import sys
from unittest.mock import patch

import pytest
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters import TemplateConverter
from genai_perf.inputs.converters.template_converter import NAMED_TEMPLATES
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.tokenizer import get_empty_tokenizer


class TestTemplateConverter:
    @staticmethod
    def create_generic_dataset(rows) -> GenericDataset:
        return GenericDataset(files_data={"file1": FileData(rows)})

    def test_jinja2_not_installed(self):
        with patch.dict(sys.modules, {"jinja2": None}, clear=True):
            with pytest.raises(ImportError):
                TemplateConverter().resolve_template("dummy")

    @pytest.mark.parametrize(
        "template", (*NAMED_TEMPLATES.keys(), """[{ "dummy": {{ texts|tojson }} }]""")
    )
    def test_check_config(self, template):
        config = InputsConfig(
            output_format=OutputFormat.TEMPLATE,
            output_template=template,
            tokenizer=get_empty_tokenizer(),
        )

        template_converter = TemplateConverter()
        template_converter.check_config(config)

    @pytest.mark.parametrize("template", ("unknown", """[] {{ texts }} }]"""))
    def test_check_config_invalid_template(self, template):
        config = InputsConfig(
            output_format=OutputFormat.TEMPLATE,
            tokenizer=get_empty_tokenizer(),
            output_template=template,
        )

        template_converter = TemplateConverter()

        with pytest.raises(GenAIPerfException):
            template_converter.check_config(config)

    def test_convert_named_template_nvembedqa(self):
        generic_dataset = self.create_generic_dataset(
            [
                DataRow(texts=["sample_prompt_1"]),
                DataRow(texts=["sample_prompt_2", "sample_prompt_3"]),
            ]
        )
        config = InputsConfig(
            output_format=OutputFormat.TEMPLATE,
            tokenizer=get_empty_tokenizer(),
            output_template="nv-embedqa",
        )
        converter = TemplateConverter()
        result = converter.convert(generic_dataset, config)
        expected_result = {
            "data": [
                {"text": ["sample_prompt_1"]},
                {"text": ["sample_prompt_2"]},
                {"text": ["sample_prompt_3"]},
            ]
        }
        assert result == expected_result

    def test_convert_custom_template(self):
        output_template = """[{ "dummy": {{ texts|tojson }} }]"""
        generic_dataset = self.create_generic_dataset(
            [
                DataRow(texts=["sample_prompt_1"]),
                DataRow(texts=["sample_prompt_2", "sample_prompt_3"]),
            ]
        )
        config = InputsConfig(
            output_format=OutputFormat.TEMPLATE,
            tokenizer=get_empty_tokenizer(),
            output_template=output_template,
        )
        converter = TemplateConverter()
        result = converter.convert(generic_dataset, config)
        expected_result = {
            "data": [
                {"dummy": ["sample_prompt_1"]},
                {"dummy": ["sample_prompt_2", "sample_prompt_3"]},
            ]
        }
        assert result == expected_result
