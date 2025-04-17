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

import logging
from typing import Any, Dict
from urllib.parse import urlparse

from genai_perf.config.endpoint_config import endpoint_type_map
from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_defaults import EndPointDefaults
from genai_perf.config.input.config_field import ConfigField
from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.utils import split_and_strip_whitespace

logger = logging.getLogger(__name__)


class ConfigEndPoint(BaseConfig):
    """
    Describes the configuration for an endpoint
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_selection_strategy: Any = ConfigField(
            default=EndPointDefaults.MODEL_SELECTION_STRATEGY,
            choices=ModelSelectionStrategy,
            verbose_template_comment="When multiple model are specified, this is how a specific model should be assigned to a prompt.\
            \nround_robin: nth prompt in the list gets assigned to n-mod len(models).\
            \nrandom: assignment is uniformly random",
        )
        self.backend: Any = ConfigField(
            default=EndPointDefaults.BACKEND,
            choices=list(OutputFormat)[:2],
            verbose_template_comment="When benchmarking Triton, this is the backend of the model.",
        )
        self.custom: Any = ConfigField(
            default=EndPointDefaults.CUSTOM,
            verbose_template_comment="Set a custom endpoint that differs from the OpenAI defaults.",
        )
        self.type: Any = ConfigField(
            default=EndPointDefaults.TYPE,
            choices=list(endpoint_type_map.keys()),
            verbose_template_comment="The type to send requests to on the server.",
        )
        self.streaming: Any = ConfigField(
            default=EndPointDefaults.STREAMING,
            verbose_template_comment="An option to enable the use of the streaming API.",
        )
        self.server_metrics_urls: Any = ConfigField(
            default=EndPointDefaults.SERVER_METRICS_URLS,
            verbose_template_comment="The list of Triton server metrics URLs.\
                \nThese are used for Telemetry metric reporting with Triton.",
        )
        self.url: Any = ConfigField(
            default=EndPointDefaults.URL,
            verbose_template_comment="URL of the endpoint to target for benchmarking.",
        )
        self.grpc_method: Any = ConfigField(
            default=EndPointDefaults.GRPC_METHOD,
            verbose_template_comment="A fully-qualified gRPC method name in "
            "'<package>.<service>/<method>' format."
            "\nThe option is only supported by dynamic gRPC service kind and is"
            "\nrequired to identify the RPC to use when sending requests to the server.",
        )

    def parse(self, endpoint: Dict[str, Any]) -> None:
        for key, value in endpoint.items():
            if key == "model_selection_strategy":
                self.model_selection_strategy = ModelSelectionStrategy(value.upper())
            elif key == "backend":
                self.backend = OutputFormat(value.upper()) if value else None
            elif key == "custom":
                self.custom = value
            elif key == "type":
                self.type = value
            elif key == "streaming":
                self.streaming = value
            elif key == "server_metrics_url" or key == "server_metrics_urls":
                self._parse_server_metrics_url(value)
            elif key == "url":
                self.url = value
            elif key == "grpc_method":
                self.grpc_method = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid endpoint parameter"
                )

    def _parse_server_metrics_url(self, server_metrics_urls: Any) -> None:
        if type(server_metrics_urls) is str:
            self.server_metrics_urls = split_and_strip_whitespace(server_metrics_urls)
        elif type(server_metrics_urls) is list:
            self.server_metrics_urls = server_metrics_urls
        else:
            raise ValueError(
                "User Config: server_metrics_url(s) must be a string or list"
            )

    ###########################################################################
    # Illegal Combination Methods
    ###########################################################################
    def check_for_illegal_combinations(self) -> None:
        self._check_server_metrics_url()

    def _check_server_metrics_url(self) -> None:
        if self.service_kind == "triton" and self.server_metrics_urls:
            for url in self.server_metrics_urls:
                self._check_for_valid_url(url)

    def _check_for_valid_url(self, url: str) -> None:
        """
        Validates a URL to ensure it meets the following criteria:
        - The scheme must be 'http' or 'https'.
        - The netloc (domain) must be present OR the URL must be a valid localhost
        address.
        - The path must contain '/metrics'.
        - The port must be specified.

        Raises:
            ValueError if the URL is invalid.

        The URL structure is expected to follow:
        <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
        """
        parsed_url = urlparse(url)

        if parsed_url.scheme not in ["http", "https"]:
            raise ValueError(
                f"Invalid scheme '{parsed_url.scheme}' in URL: {url}. Use 'http' "
                "or 'https'."
            )

        valid_localhost = parsed_url.hostname in ["localhost", "127.0.0.1"]

        if not parsed_url.netloc and not valid_localhost:
            raise ValueError(
                f"Invalid domain in URL: {url}. Use a valid hostname or " "'localhost'."
            )

        if "/metrics" not in parsed_url.path:
            raise ValueError(
                f"Invalid URL path '{parsed_url.path}' in {url}. The path must "
                "include '/metrics'."
            )

        if parsed_url.port is None:
            raise ValueError(
                f"Port missing in URL: {url}. A port number is required "
                "(e.g., ':8002')."
            )

    ###########################################################################
    # Infer Methods
    ###########################################################################
    def infer_settings(self, model_name: str) -> None:
        # IMPORTANT: There is a nasty dependency ordering between the infer methods
        # For readability they are broken up into separate methods, that daisy chain
        # their calls
        self.infer_service_kind(model_name)

    def infer_service_kind(self, model_name: str) -> None:
        self.service_kind: Any = ConfigField(
            default=EndPointDefaults.SERVICE_KIND,
            choices=["dynamic_grpc", "openai", "tensorrtllm_engine", "triton"],
        )
        if self.get_field("type").is_set_by_user:
            endpoint_config = endpoint_type_map[self.type]
            self.service_kind = endpoint_config.service_kind
        else:
            self.service_kind = "triton"

        self._check_inferred_type()
        self._check_inferred_backend()
        self.infer_output_format(model_name)

    def infer_output_format(self, model_name: str) -> None:
        self.output_format: Any = ConfigField(default=None, add_to_template=False)

        if self.service_kind == "triton" and self.type in ["kserve", "template"]:
            self.output_format = self.backend
        elif self.service_kind == "tensorrtllm_engine":
            self.output_format = OutputFormat.TENSORRTLLM_ENGINE
        else:
            endpoint_config = endpoint_type_map[self.type]
            self.output_format = endpoint_config.output_format

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

    def _check_inferred_backend(self) -> None:
        if self.get_field("backend").is_set_by_user:
            if self.service_kind == "triton" and self.type == "kserve":
                return
            else:
                raise ValueError(
                    f"The backend should only be used with the following combination: 'service_kind: triton' & 'type: kserve'"
                )
