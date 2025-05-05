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

from unittest.mock import MagicMock, patch

import genai_perf.logging as logging
import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.constants import DEFAULT_DCGM_METRICS_URL
from genai_perf.subcommand.subcommand import Subcommand
from genai_perf.telemetry_data.dcgm_telemetry_data_collector import (
    DCGMTelemetryDataCollector,
)
from requests import codes as http_codes


class TestCreateTelemetryDataCollectors:

    @pytest.mark.parametrize(
        "server_metrics_urls, expected_url",
        [
            (["http://dcgmhost:9400/metrics"], "http://dcgmhost:9400/metrics"),
            (None, DEFAULT_DCGM_METRICS_URL),
        ],
    )
    @patch("requests.get")
    def test_creates_telemetry_data_collectors_success(
        self, mock_requests_get, server_metrics_urls, expected_url
    ):
        mock_requests_get.return_value = MagicMock(status_code=http_codes.ok)

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.service_kind = "openai"
        config.endpoint.server_metrics_urls = server_metrics_urls

        subcommand = Subcommand(config)
        telemetry_collectors = subcommand._create_telemetry_data_collectors()

        assert len(telemetry_collectors) > 0
        assert telemetry_collectors[0].metrics_url == expected_url

    @pytest.mark.parametrize(
        "server_metrics_urls, expected_urls",
        [
            (
                [
                    "http://dcgmmetrics1.com:9400/metrics",
                    "http://dcgmmetrics2.com:9090/metrics",
                ],
                [
                    "http://dcgmmetrics1.com:9400/metrics",
                    "http://dcgmmetrics2.com:9090/metrics",
                ],
            ),
            ([], [DEFAULT_DCGM_METRICS_URL]),
        ],
    )
    @patch("requests.get")
    def test_creates_multiple_telemetry_collectors(
        self, mock_requests_get, server_metrics_urls, expected_urls
    ):
        mock_requests_get.return_value = MagicMock(status_code=http_codes.ok)

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.service_kind = "openai"
        config.endpoint.server_metrics_urls = server_metrics_urls

        subcommand = Subcommand(config)
        collectors = subcommand._create_telemetry_data_collectors()

        assert len(collectors) == len(expected_urls)
        for collector, expected_url in zip(collectors, expected_urls):
            assert isinstance(collector, DCGMTelemetryDataCollector)
            assert collector.metrics_url == expected_url

    @patch("requests.get")
    def test_telemetry_data_collectors_unreachable_url(self, mock_requests_get):
        mock_requests_get.return_value = MagicMock(status_code=http_codes.not_found)

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.service_kind = "openai"
        config.endpoint.server_metrics_urls = ["http://dcgmhost:9410/metrics"]

        subcommand = Subcommand(config)
        collectors = subcommand._create_telemetry_data_collectors()

        assert collectors == []

    @patch("requests.get")
    def test_logs_warning_when_service_kind_triton(self, mock_requests_get):
        expected_warning_message = (
            "GPU metrics are no longer collected from Triton's /metrics endpoint.\n"
            "Telemetry is now collected exclusively from the DCGM-Exporter /metrics endpoint.\n"
            "If you're using Triton, please ensure DCGM-Exporter is running.\n"
        )

        mock_requests_get.return_value = MagicMock(status_code=http_codes.ok)
        logging.init_logging()
        logger = logging.getLogger("genai_perf.subcommand.subcommand")

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.service_kind = "triton"

        subcommand = Subcommand(config)

        with patch.object(logger, "warning") as mock_logger:
            _ = subcommand._create_telemetry_data_collectors()

        mock_logger.assert_any_call(expected_warning_message)
