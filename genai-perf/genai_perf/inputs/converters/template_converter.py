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

import os
from typing import Any, Dict, cast

import jinja2
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.retrievers.generic_dataset import GenericDataset
from genai_perf.utils import load_json_str

NAMED_TEMPLATES = {
    "nv-embedqa": """[
        {% for item in texts %}
            {"text": [{{ item|tojson }}]}{% if not loop.last %},{% endif %}
        {% endfor %}
    ]""",
}


class TemplateConverter(BaseConverter):
    """
    Convert a GenericDataset to a request body.

    The template should render a list of text strings, and return a list of payloads.
    """

    def resolve_template(self, template_name_or_content: str):

        if template_name_or_content in NAMED_TEMPLATES:
            environment = jinja2.Environment(
                autoescape=True,
                loader=jinja2.DictLoader(NAMED_TEMPLATES),
            )
            return environment.get_template(template_name_or_content)

        if os.path.isfile(template_name_or_content):
            try:
                with open(template_name_or_content, "r", encoding="utf-8") as f:
                    template_content = f.read()
            except Exception as e:
                raise GenAIPerfException(f"Error reading template file: {e}")
        else:
            raise GenAIPerfException(
                f"Template file not found: {template_name_or_content}"
            )

        environment = jinja2.Environment(autoescape=True)
        return environment.from_string(template_content)

    def check_config(self) -> None:
        if self.config.input.extra:
            for key, value in self.config.input.extra.items():
                if key != "payload_template":
                    raise GenAIPerfException(
                        "Template only supports the extra input 'payload_template'. "
                    )

        payload_template = (
            self.config.input.extra.get("payload_template")
            if self.config.input.extra
            else None
        )
        if not payload_template:
            keys = (
                list(self.config.input.extra.keys()) if self.config.input.extra else []
            )
            raise GenAIPerfException(
                "The template converter requires the "
                "extra input payload_template, only "
                "detected the following --extra-inputs: "
                f"{keys}."
            )
        try:
            template = self.resolve_template(payload_template)
            test_texts = ["test1", "test2"]
            payloads_json = template.render(texts=test_texts)
            payloads = load_json_str(payloads_json)
            if not isinstance(payloads, list):
                raise ValueError(
                    "Template does not render a list of strings to a list of items. "
                    f"For example, {test_texts} is rendered into {payloads}."
                )
        except Exception as e:
            raise GenAIPerfException(e)

    def convert(
        self,
        generic_dataset: GenericDataset,
    ) -> Dict[Any, Any]:
        payload_template = self.config.input.extra.get("payload_template")
        payload_template = cast(str, payload_template)
        template = self.resolve_template(payload_template)

        request_body: Dict[str, Any] = {"data": []}

        for file_data in generic_dataset.files_data.values():
            for _, row in enumerate(file_data.rows):
                payloads_json = template.render(texts=row.texts)
                request_body["data"] += load_json_str(payloads_json)
        return request_body
