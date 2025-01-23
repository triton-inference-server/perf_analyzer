#!/usr/bin/env python3
#
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from genai_perf.constants import DEFAULT_TRITON_METRICS_URL
from genai_perf.subcommand.common import create_telemetry_data_collectors
from genai_perf.telemetry_data import TritonTelemetryDataCollector
from requests import codes as http_codes


class MockArgs:
    def __init__(self, service_kind, server_metrics_url):
        self.service_kind = service_kind
        self.server_metrics_url = server_metrics_url


class TestCreateTelemetryDataCollector:
    test_triton_metrics_url = "http://tritonmetrics.com:8080/metrics"

    @pytest.mark.parametrize(
        "server_metrics_url, expected_url",
        [
            (test_triton_metrics_url, test_triton_metrics_url),
            (None, DEFAULT_TRITON_METRICS_URL),
        ],
    )
    @patch("requests.get")
    def test_creates_telemetry_data_collector_success(
        self, mock_requests_get, server_metrics_url, expected_url
    ):
        mock_requests_get.return_value = MagicMock(status_code=http_codes.ok)

        mock_args = MockArgs(
            service_kind="triton", server_metrics_url=server_metrics_url
        )
        telemetry_data_collector = create_telemetry_data_collectors(mock_args)[0]

        assert isinstance(telemetry_data_collector, TritonTelemetryDataCollector)
        assert telemetry_data_collector.metrics_url == expected_url

    @pytest.mark.parametrize(
        "server_metrics_url",
        [
            test_triton_metrics_url,
            None,
        ],
    )
    @patch("requests.get")
    def test_create_telemetry_data_collector_unreachable_url(
        self, mock_requests_get, server_metrics_url
    ):
        mock_requests_get.return_value = MagicMock(status_code=http_codes.not_found)

        mock_args = MockArgs(
            service_kind="triton", server_metrics_url=server_metrics_url
        )
        telemetry_data_collector = create_telemetry_data_collectors(mock_args)[0]

        assert telemetry_data_collector is None

    @patch("genai_perf.subcommand.common.TritonTelemetryDataCollector")
    @patch("requests.get")
    def test_create_telemetry_data_collector_service_kind_not_triton(
        self,
        mock_requests_get,
        mock_telemetry_collector,
    ):
        mock_requests_get.return_value = MagicMock(status_code=http_codes.ok)
        mock_telemetry_collector.return_value = MagicMock()

        mock_args = MockArgs(
            service_kind="openai", server_metrics_url=self.test_triton_metrics_url
        )
        telemetry_data_collector = create_telemetry_data_collectors(mock_args)[0]

        assert telemetry_data_collector is None
