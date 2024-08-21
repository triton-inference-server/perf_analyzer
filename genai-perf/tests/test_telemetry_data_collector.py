#!/usr/bin/env python3

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
import requests
from genai_perf.telemetry_data.telemetry_data_collector import TelemetryDataCollector


class MockTelemetryDataCollector(TelemetryDataCollector):
    def _process_and_update_metrics(self, metrics_data: str) -> None:
        pass


class TestTelemetryDataCollector:

    TEST_SERVER_URL = "http://testserver:8080/metrics"

    triton_metrics_response = """\
            nv_gpu_power_usage{gpu="0",uuid="GPU-1234"} 123.45
            nv_gpu_power_usage{gpu="1",uuid="GPU-5678"} 234.56
            nv_gpu_utilization{gpu="0",uuid="GPU-1234"} 76.3
            nv_gpu_utilization{gpu="1",uuid="GPU-5678"} 88.1
            nv_gpu_memory_total_bytes{gpu="0",uuid="GPU-1234"} 8589934592
            nv_gpu_memory_total_bytes{gpu="1",uuid="GPU-5678"} 8589934592
            nv_gpu_memory_used_bytes{gpu="0",uuid="GPU-1234"} 2147483648
            nv_gpu_memory_used_bytes{gpu="1",uuid="GPU-5678"} 3221225472
            """

    @pytest.fixture
    def collector(self) -> MockTelemetryDataCollector:
        return MockTelemetryDataCollector(self.TEST_SERVER_URL)

    @patch("genai_perf.telemetry_data.telemetry_data_collector.Thread")
    def test_start(
        self, mock_thread_class: MagicMock, collector: MockTelemetryDataCollector
    ) -> None:
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance

        collector.start()

        assert collector._thread is not None
        assert collector._thread.is_alive()
        mock_thread_class.assert_called_once_with(target=collector._collect_metrics)
        mock_thread_instance.start.assert_called_once()

    @patch("genai_perf.telemetry_data.telemetry_data_collector.Thread")
    def test_stop(
        self, mock_thread_class: MagicMock, collector: MockTelemetryDataCollector
    ) -> None:
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance

        collector.start()

        assert collector._thread is not None
        assert collector._thread.is_alive()

        collector.stop()

        assert collector._stop_event.is_set()
        mock_thread_instance.join.assert_called_once()

        mock_thread_instance.is_alive.return_value = False
        assert not collector._thread.is_alive()

    @patch("requests.get")
    def test_fetch_metrics_success(
        self, mock_requests_get: MagicMock, collector: MockTelemetryDataCollector
    ) -> None:

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = self.triton_metrics_response
        mock_requests_get.return_value = mock_response

        result = collector._fetch_metrics()

        mock_requests_get.assert_called_once_with(self.TEST_SERVER_URL)

        assert result == self.triton_metrics_response

    @patch("requests.get")
    def test_fetch_metrics_failure(
        self, mock_requests_get: MagicMock, collector: MockTelemetryDataCollector
    ) -> None:
        mock_requests_get.side_effect = requests.exceptions.HTTPError("Not Found")

        with pytest.raises(requests.exceptions.HTTPError):
            collector._fetch_metrics()

    @patch.object(MockTelemetryDataCollector, "_fetch_metrics")
    @patch("genai_perf.telemetry_data.telemetry_data_collector.time.sleep")
    def test_collect_metrics(
        self,
        mock_sleep: MagicMock,
        mock_fetch_metrics: MagicMock,
        collector: MockTelemetryDataCollector,
    ) -> None:

        mock_fetch_metrics.return_value = self.triton_metrics_response

        with patch.object(
            collector, "_process_and_update_metrics", new_callable=MagicMock
        ) as mock_process_and_update_metrics:
            # Mock _stop_event.is_set
            collector._stop_event = MagicMock()
            collector._stop_event.is_set = MagicMock(
                side_effect=[False, True]
            )  # Ensure loop exits immediately

            collector._collect_metrics()

            mock_fetch_metrics.assert_called_once()
            mock_process_and_update_metrics.assert_called_once_with(
                self.triton_metrics_response
            )
            mock_sleep.assert_called_once()

    @patch("requests.get")
    def test_url_reachability_check_success(
        self,
        mock_get: MagicMock,
        collector: MockTelemetryDataCollector,
    ) -> None:
        mock_get.return_value.status_code = requests.codes.ok
        assert collector.is_url_reachable() is True

    @patch("requests.get")
    def test_url_reachability_check_failure(
        self, mock_get: MagicMock, collector: MockTelemetryDataCollector
    ) -> None:
        # Simulate a 404 Not Found error
        mock_get.return_value.status_code = requests.codes.not_found
        assert collector.is_url_reachable() is False

        # Simulate a 500 Internal Server Error
        mock_get.return_value.status_code = requests.codes.server_error
        assert collector.is_url_reachable() is False

        # Simulate a 403 Forbidden error
        mock_get.return_value.status_code = requests.codes.forbidden
        assert collector.is_url_reachable() is False

        # Simulate a timeout exception
        mock_get.side_effect = requests.exceptions.Timeout
        assert collector.is_url_reachable() is False

        # Simulate a connection error
        mock_get.side_effect = requests.exceptions.ConnectionError
        assert collector.is_url_reachable() is False

        # Simulate too many redirects
        mock_get.side_effect = requests.exceptions.TooManyRedirects
        assert collector.is_url_reachable() is False

        # Simulate a generic request exception
        mock_get.side_effect = requests.exceptions.RequestException
        assert collector.is_url_reachable() is False
