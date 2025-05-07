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
import subprocess  # nosec
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import genai_perf.logging as logging
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.inputs import Inputs
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.metrics.telemetry_metrics import TelemetryMetrics
from genai_perf.profile_data_parser import (
    ImageRetrievalProfileDataParser,
    LLMProfileDataParser,
    ProfileDataParser,
)
from genai_perf.telemetry_data.triton_telemetry_data_collector import (
    TelemetryDataCollector,
    TritonTelemetryDataCollector,
)
from genai_perf.tokenizer import Tokenizer
from genai_perf.utils import load_json_str, remove_file

logger = logging.getLogger(__name__)


"""
Contains methods that are used by multiple subcommands
"""


def get_extra_inputs_as_dict(args: Namespace) -> Dict[str, Any]:
    request_inputs: Dict[str, Any] = {}
    if args.extra_inputs:
        for input_str in args.extra_inputs:
            semicolon_count = input_str.count(":")
            if input_str.startswith("{") and input_str.endswith("}"):
                request_inputs.update(load_json_str(input_str))
            elif semicolon_count == 0:  # extra input as a flag
                request_inputs[input_str] = None
            elif semicolon_count == 1:
                input_name, value = input_str.split(":", 1)

                if not input_name or not value:
                    raise ValueError(
                        f"Input name or value is empty in --extra-inputs: "
                        f"{input_str}\nExpected input format: 'input_name' or "
                        "'input_name:value'"
                    )

                is_bool = value.lower() in ["true", "false"]
                is_int = value.isdigit()
                is_float = value.count(".") == 1 and (
                    value[0] == "." or value.replace(".", "").isdigit()
                )

                if is_bool:
                    value = value.lower() == "true"
                elif is_int:
                    value = int(value)
                elif is_float:
                    value = float(value)

                if input_name in request_inputs:
                    raise ValueError(
                        f"Input name already exists in request_inputs "
                        f"dictionary: {input_name}"
                    )
                request_inputs[input_name] = value
            else:
                raise ValueError(
                    f"Invalid input format for --extra-inputs: {input_str}\n"
                    "Expected input format: 'input_name' or 'input_name:value'"
                )

    return request_inputs
