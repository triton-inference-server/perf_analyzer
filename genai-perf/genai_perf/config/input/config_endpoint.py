# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict

from genai_perf.config.endpoint_config import endpoint_type_map
from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_defaults import EndPointDefaults
from genai_perf.config.input.config_field import ConfigField
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat


class ConfigEndPoint(BaseConfig):
    """
    Describes the configuration for an endpoint
    """

    def __init__(self):
        super().__init__()
        self.model_selection_strategy: Any = ConfigField(
            default=EndPointDefaults.MODEL_SELECTION_STRATEGY,
            choices=ModelSelectionStrategy,
        )
        self.backend: Any = ConfigField(
            default=EndPointDefaults.BACKEND, choices=OutputFormat
        )
        self.custom: Any = ConfigField(default=EndPointDefaults.CUSTOM)
        self.type: Any = ConfigField(
            default=EndPointDefaults.TYPE,
            choices=list(endpoint_type_map.keys()),
        )
        self.service_kind: Any = ConfigField(
            default=EndPointDefaults.SERVICE_KIND,
            choices=["triton", "openai", "tensorrtllm_engine"],
        )
        self.streaming: Any = ConfigField(default=EndPointDefaults.STREAMING)
        self.server_metrics_url: Any = ConfigField(
            default=EndPointDefaults.SERVER_METRICS_URL
        )
        self.url: Any = ConfigField(default=EndPointDefaults.URL)

    def parse(self, endpoint: Dict[str, Any]) -> None:
        for key, value in endpoint.items():
            if key == "model_selection_strategy":
                self.model_selection_strategy = ModelSelectionStrategy(value.upper())
            elif key == "backend":
                self.backend = OutputFormat(value.upper())
            elif key == "custom":
                self.custom = value
            elif key == "type":
                self.type = value
            elif key == "service_kind":
                self.service_kind = value
            elif key == "streaming":
                self.streaming = value
            elif key == "server_metrics_url":
                self.server_metrics_url = value
            elif key == "url":
                self.url = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid endpoint parameter"
                )
