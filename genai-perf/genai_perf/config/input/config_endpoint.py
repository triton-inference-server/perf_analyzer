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

from typing import Any, Dict, List

from genai_perf.config.endpoint_config import EndpointConfig, endpoint_type_map
from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_defaults import EndPointDefaults
from genai_perf.config.input.config_field import ConfigField
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat


class ConfigEndPoint(BaseConfig):
    """
    Describes the configuration for an endpoint
    """

    def __init__(self) -> None:
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
        self.server_metrics_urls: Any = ConfigField(
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
            elif key == "server_metrics_url" or key == "server_metrics_urls":
                self.server_metrics_url = self._parse_server_metrics_url(value)
            elif key == "url":
                self.url = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid endpoint parameter"
                )

    def _parse_server_metrics_url(self, value: Any) -> List[str]:
        if type(value) is str:
            return [value]
        elif type(value) is list:
            return value
        else:
            raise ValueError(
                "User Config: server_metrics_url(s) must be a string or list"
            )

    ###########################################################################
    # Illegal Combination Methods
    ###########################################################################
    def check_for_illegal_combinations(self) -> None:
        self._check_service_kind_and_type()

    def _check_service_kind_and_type(self) -> None:
        if not self.type:
            if (
                not self.service_kind == "triton"
                or not self.service_kind == "tensorrtllm_engine"
            ):
                raise ValueError(
                    f"User Config: service_kind {self.service_kind} requires a type to be specified"
                )

    ###########################################################################
    # Infer Methods
    ###########################################################################
    def infer_settings(self, model_name: str) -> None:
        # IMPORTANT: There is a nasty dependency ordering between the infer methods
        # For readability they are broken up into separate methods, that daisy chain
        # their calls
        self.infer_type(model_name)

    def infer_type(self, model_name: str) -> None:
        if self.service_kind == "triton" and not self.type:
            self.type = "kserve"

        if self.service_kind == "tensorrtllm_engine" and not self.type:
            self.type = "tensorrtllm_engine"

        self._check_inferred_type()
        self.infer_service_kind(model_name)

    def infer_service_kind(self, model_name: str) -> None:
        if self.service_kind == "triton" and self.type == "generate":
            self.service_kind = "openai"

        self._check_inferred_backend()
        self.infer_output_format(model_name)

    def infer_output_format(self, model_name: str) -> None:
        if self.service_kind == "triton" and self.type == "kserve":
            self.output_format = self.backend
        elif self.service_kind == "tensorrtllm_engine":
            self.output_format = OutputFormat.TENSORRTLLM_ENGINE
        else:
            endpoint_config = endpoint_type_map[self.type]
            self.output_format = endpoint_config.output_format
            self._check_inferred_service_kind(endpoint_config)

        self.infer_custom(model_name)

    def infer_custom(self, model_name: str) -> None:
        endpoint_config = endpoint_type_map[self.type]

        if self.custom:
            self.custom = self.custom.lstrip(" /")
        elif endpoint_config.endpoint:
            self.custom = endpoint_config.endpoint.format(MODEL_NAME=model_name)

    ###########################################################################
    # Check Methods - these must be done after inferencing
    ###########################################################################
    def _check_inferred_type(self) -> None:
        if self.type and self.type not in endpoint_type_map:
            raise ValueError(f"User Config: {self.type} is not a valid endpoint type")

    def _check_inferred_service_kind(self, endpoint_config: EndpointConfig) -> None:
        # This is inferred and cannot be checked by the endpoint_config
        if self.service_kind == "openai" and self.type == "generate":
            return

        if self.service_kind != endpoint_config.service_kind:
            raise ValueError(
                f"Invalid service-kind '{self.service_kind}' for endpoint-type '{self.type}'"
            )

    def _check_inferred_backend(self) -> None:
        if self.get_field("backend").is_set_by_user:
            if self.service_kind == "triton" and self.type == "kserve":
                return
            else:
                raise ValueError(
                    f"The backend should only be used with the following combination: 'service_kind: triton' & 'type: kserve'"
                )
