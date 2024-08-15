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

from unittest.mock import patch

import pytest
from genai_perf.constants import DEFAULT_TRITON_METRICS_URL
from genai_perf.parser import profile_handler
from genai_perf.telemetry_data.triton_telemetry_data_collector import (
    TritonTelemetryDataCollector,
)

TRITON_TEST_METRICS_URL = "http://tritonmetrics.com:8080/metrics"


# Parameterized fixture
@pytest.fixture(
    params=[
        {"service_kind": "triton", "server_metrics_url": TRITON_TEST_METRICS_URL},
        {"service_kind": "triton", "server_metrics_url": None},  # Default URL case
    ]
)
def mock_args(request):
    params = request.param

    class MockArgs:
        def __init__(self, service_kind, server_metrics_url):
            self.service_kind = service_kind
            self.server_metrics_url = server_metrics_url

    return MockArgs(params["service_kind"], params["server_metrics_url"])


@patch("genai_perf.wrapper.Profiler.run")
def test_profile_handler_creates_telemetry_collector_triton(
    mock_profiler_run, mock_args
):
    profile_handler(mock_args, extra_args={})
    mock_profiler_run.assert_called_once()

    args, kwargs = mock_profiler_run.call_args
    telemetry_collector = kwargs.get("telemetry_data_collector")

    expected_url = (
        TRITON_TEST_METRICS_URL
        if mock_args.server_metrics_url == TRITON_TEST_METRICS_URL
        else DEFAULT_TRITON_METRICS_URL
    )

    assert isinstance(telemetry_collector, TritonTelemetryDataCollector)
    assert telemetry_collector.metrics_url == expected_url
