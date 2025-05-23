# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import MagicMock

import pytest
from genai_perf.metrics.telemetry_metrics import TelemetryMetricName, TelemetryMetrics
from genai_perf.telemetry_data.dcgm_telemetry_data_collector import (
    DCGMTelemetryDataCollector,
)


@pytest.fixture
def collector():
    collector = DCGMTelemetryDataCollector(
        server_metrics_url="http://localhost:9400/metrics"
    )
    collector._metrics = MagicMock(spec=TelemetryMetrics)
    collector._metrics.TELEMETRY_METRICS = [
        MagicMock(name=metric.value) for metric in TelemetryMetricName
    ]
    return collector


def test_process_and_update_metrics_success(collector):
    sample_metrics = """
    DCGM_FI_DEV_POWER_USAGE{gpu="0"} 25.0
    DCGM_FI_DEV_GPU_UTIL{gpu="0"} 88.5
    DCGM_FI_DEV_FB_USED{gpu="0"} 4096
    DCGM_FI_DEV_FB_TOTAL{gpu="0"} 49152
    DCGM_FI_DEV_FB_FREE{gpu="0"} 8192
    DCGM_FI_DEV_MEMORY_TEMP{gpu="0"} 60
    DCGM_FI_DEV_GPU_TEMP{gpu="0"} 70
    DCGM_FI_DEV_SM_CLOCK{gpu="0"} 1530
    DCGM_FI_DEV_MEM_CLOCK{gpu="0"} 5050
    DCGM_FI_PROF_SM_ACTIVE{gpu="0"} 65
    DCGM_FI_DEV_MEM_COPY_UTIL{gpu="0"} 55
    DCGM_FI_DEV_ENC_UTIL{gpu="0"} 22
    DCGM_FI_DEV_DEC_UTIL{gpu="0"} 17
    DCGM_FI_DEV_ECC_SBE_VOL_TOTAL{gpu="0"} 1
    DCGM_FI_DEV_ECC_DBE_VOL_TOTAL{gpu="0"} 2
    DCGM_FI_DEV_ECC_SBE_AGG_TOTAL{gpu="0"} 3
    DCGM_FI_DEV_ECC_DBE_AGG_TOTAL{gpu="0"} 4
    DCGM_FI_DEV_XID_ERRORS{gpu="0"} 5
    DCGM_FI_DEV_POWER_VIOLATION{gpu="0"} 6
    DCGM_FI_DEV_THERMAL_VIOLATION{gpu="0"} 7
    DCGM_FI_DEV_RETIRED_SBE{gpu="0"} 8
    DCGM_FI_DEV_RETIRED_DBE{gpu="0"} 9
    DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL{gpu="0"} 10
    DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL{gpu="0"} 11
    DCGM_FI_PROF_PCIE_TX_BYTES{gpu="0"} 1048576
    DCGM_FI_PROF_PCIE_RX_BYTES{gpu="0"} 2097152
    DCGM_FI_DEV_PCIE_REPLAY_COUNTER{gpu="0"} 3

    """

    collector._process_and_update_metrics(sample_metrics)
    update_call = collector.metrics.update_metrics.call_args[0][0]

    assert update_call["gpu_power_usage"]["0"] == [25.0]
    assert update_call["gpu_utilization"]["0"] == [88.5]
    assert update_call["gpu_memory_used"]["0"] == [4096.0]
    assert update_call["total_gpu_memory"]["0"] == [49152.0]
    assert update_call["gpu_memory_free"]["0"] == [8192.0]
    assert update_call["gpu_memory_temperature"]["0"] == [60.0]
    assert update_call["gpu_temperature"]["0"] == [70.0]
    assert update_call["gpu_clock_sm"]["0"] == [1530.0]
    assert update_call["gpu_clock_memory"]["0"] == [5050.0]
    assert update_call["sm_utilization"]["0"] == [65.0]
    assert update_call["memory_copy_utilization"]["0"] == [55.0]
    assert update_call["video_encoder_utilization"]["0"] == [22.0]
    assert update_call["video_decoder_utilization"]["0"] == [17.0]
    assert update_call["total_ecc_sbe_volatile"]["0"] == [1.0]
    assert update_call["total_ecc_dbe_volatile"]["0"] == [2.0]
    assert update_call["total_ecc_sbe_aggregate"]["0"] == [3.0]
    assert update_call["total_ecc_dbe_aggregate"]["0"] == [4.0]
    assert update_call["xid_last_error"]["0"] == [5.0]
    assert update_call["power_throttle_duration"]["0"] == [6.0]
    assert update_call["thermal_throttle_duration"]["0"] == [7.0]
    assert update_call["retired_pages_sbe"]["0"] == [8.0]
    assert update_call["retired_pages_dbe"]["0"] == [9.0]
    assert update_call["total_nvlink_crc_flit_errors"]["0"] == [10.0]
    assert update_call["total_nvlink_crc_data_errors"]["0"] == [11.0]
    assert update_call["pcie_transmit_throughput"]["0"] == [1048576.0]
    assert update_call["pcie_receive_throughput"]["0"] == [2097152.0]
    assert update_call["pcie_replay_counter"]["0"] == [3.0]


def test_process_and_update_metrics_ignores_unmapped_metrics(collector):
    sample_metrics = """
    DCGM_FI_DEV_UNRELATED{gpu="0"} 100.0
    DCGM_FI_DEV_FB_TOTAL{gpu="0"} 49152
    """

    collector._process_and_update_metrics(sample_metrics)
    update_call = collector.metrics.update_metrics.call_args[0][0]

    assert "total_gpu_memory" in update_call
    assert update_call["total_gpu_memory"]["0"] == [49152.0]
    assert "unrelated" not in update_call


def test_parse_metric_line_success(collector):
    line = 'DCGM_FI_DEV_GPU_UTIL{gpu="0"} 76.5'
    metric_full_name, value = collector._parse_metric_line(line)
    assert metric_full_name == 'DCGM_FI_DEV_GPU_UTIL{gpu="0"}'
    assert value == 76.5


def test_parse_metric_line_failure(collector):
    assert collector._parse_metric_line("bad_line_without_value") is None


def test_extract_gpu_label_success(collector):
    line = 'DCGM_FI_DEV_POWER_USAGE{gpu="1",UUID="abc"}'
    assert collector._extract_gpu_label(line) == "1"


def test_extract_gpu_label_missing(collector):
    line = 'DCGM_FI_DEV_POWER_USAGE{UUID="abc"}'
    assert collector._extract_gpu_label(line) is None


def test_append_metric_value_new_gpu(collector):
    metric_data = {}
    collector._append_metric_value(metric_data, "gpu_utilization", "2", 90.0)
    assert metric_data["gpu_utilization"]["2"] == [90.0]
