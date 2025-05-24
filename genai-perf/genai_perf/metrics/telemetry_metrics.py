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

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional

from genai_perf.metrics.metrics import MetricMetadata


class TelemetryMetricName(str, Enum):
    GPU_UTILIZATION = "gpu_utilization"
    SM_UTILIZATION = "sm_utilization"
    MEMORY_COPY_UTILIZATION = "memory_copy_utilization"
    VIDEO_ENCODER_UTILIZATION = "video_encoder_utilization"
    VIDEO_DECODER_UTILIZATION = "video_decoder_utilization"
    GPU_POWER_USAGE = "gpu_power_usage"
    GPU_POWER_LIMIT = "gpu_power_limit"
    ENERGY_CONSUMPTION = "energy_consumption"
    TOTAL_GPU_MEMORY = "total_gpu_memory"
    GPU_MEMORY_USED = "gpu_memory_used"
    GPU_MEMORY_FREE = "gpu_memory_free"
    GPU_MEMORY_TEMPERATURE = "gpu_memory_temperature"
    GPU_TEMPERATURE = "gpu_temperature"
    GPU_CLOCK_SM = "gpu_clock_sm"
    GPU_CLOCK_MEMORY = "gpu_clock_memory"
    ECC_SBE_VOLATILE_TOTAL = "total_ecc_sbe_volatile"
    ECC_DBE_VOLATILE_TOTAL = "total_ecc_dbe_volatile"
    ECC_SBE_AGGREGATE_TOTAL = "total_ecc_sbe_aggregate"
    ECC_DBE_AGGREGATE_TOTAL = "total_ecc_dbe_aggregate"
    XID_LAST_ERROR = "xid_last_error"
    POWER_THROTTLE_DURATION = "power_throttle_duration"
    THERMAL_THROTTLE_DURATION = "thermal_throttle_duration"
    RETIRED_PAGES_SBE = "retired_pages_sbe"
    RETIRED_PAGES_DBE = "retired_pages_dbe"
    NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL = "total_nvlink_crc_flit_errors"
    NVLINK_CRC_DATA_ERROR_COUNT_TOTAL = "total_nvlink_crc_data_errors"
    PCIE_TRANSMIT_THROUGHPUT = "pcie_transmit_throughput"
    PCIE_RECEIVE_THROUGHPUT = "pcie_receive_throughput"
    PCIE_REPLAY_COUNTER = "pcie_replay_counter"

    @classmethod
    def values(cls) -> List["TelemetryMetricName"]:
        return list(cls)


class TelemetryMetrics:
    """
    A class that contains common telemetry metrics.
    Metrics are stored as
        'gpu_power_usage': {
            'gpu0': [27.01]
        },
        'gpu_utilization': {
            'gpu0': [75.5]
        },
        'energy_consumption': {
            'gpu0': [123.56]
        }
    """

    UTILIZATION_METRICS = [
        MetricMetadata(TelemetryMetricName.GPU_UTILIZATION.value, "%"),
        MetricMetadata(TelemetryMetricName.SM_UTILIZATION.value, "%"),
        MetricMetadata(TelemetryMetricName.MEMORY_COPY_UTILIZATION.value, "%"),
        MetricMetadata(TelemetryMetricName.VIDEO_ENCODER_UTILIZATION.value, "%"),
        MetricMetadata(TelemetryMetricName.VIDEO_DECODER_UTILIZATION.value, "%"),
    ]

    POWER_METRICS = [
        MetricMetadata(TelemetryMetricName.GPU_POWER_USAGE.value, "W"),
        MetricMetadata(TelemetryMetricName.GPU_POWER_LIMIT.value, "W"),
        MetricMetadata(TelemetryMetricName.ENERGY_CONSUMPTION.value, "MJ"),
    ]

    CLOCK_METRICS = [
        MetricMetadata(TelemetryMetricName.GPU_CLOCK_SM.value, "MHz"),
        MetricMetadata(TelemetryMetricName.GPU_CLOCK_MEMORY.value, "MHz"),
    ]

    MEMORY_METRICS = [
        MetricMetadata(TelemetryMetricName.TOTAL_GPU_MEMORY.value, "GB"),
        MetricMetadata(TelemetryMetricName.GPU_MEMORY_USED.value, "GB"),
        MetricMetadata(TelemetryMetricName.GPU_MEMORY_FREE.value, "GB"),
    ]

    TEMPERATURE_METRICS = [
        MetricMetadata(TelemetryMetricName.GPU_TEMPERATURE.value, "C"),
        MetricMetadata(TelemetryMetricName.GPU_MEMORY_TEMPERATURE.value, "C"),
    ]

    ECC_AND_ERROR_METRICS = [
        MetricMetadata(TelemetryMetricName.ECC_SBE_VOLATILE_TOTAL.value, "errors"),
        MetricMetadata(TelemetryMetricName.ECC_DBE_VOLATILE_TOTAL.value, "errors"),
        MetricMetadata(TelemetryMetricName.ECC_SBE_AGGREGATE_TOTAL.value, "errors"),
        MetricMetadata(TelemetryMetricName.ECC_DBE_AGGREGATE_TOTAL.value, "errors"),
        MetricMetadata(TelemetryMetricName.XID_LAST_ERROR.value, "errors"),
        MetricMetadata(TelemetryMetricName.POWER_THROTTLE_DURATION.value, "us"),
        MetricMetadata(TelemetryMetricName.THERMAL_THROTTLE_DURATION.value, "us"),
        MetricMetadata(TelemetryMetricName.RETIRED_PAGES_SBE.value, "pages"),
        MetricMetadata(TelemetryMetricName.RETIRED_PAGES_DBE.value, "pages"),
        MetricMetadata(
            TelemetryMetricName.NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL.value, "errors"
        ),
        MetricMetadata(
            TelemetryMetricName.NVLINK_CRC_DATA_ERROR_COUNT_TOTAL.value, "errors"
        ),
    ]

    PCIE_METRICS = [
        MetricMetadata(TelemetryMetricName.PCIE_TRANSMIT_THROUGHPUT.value, "KB/s"),
        MetricMetadata(TelemetryMetricName.PCIE_RECEIVE_THROUGHPUT.value, "KB/s"),
        MetricMetadata(TelemetryMetricName.PCIE_REPLAY_COUNTER.value, "retries"),
    ]

    TELEMETRY_METRICS = (
        POWER_METRICS
        + MEMORY_METRICS
        + UTILIZATION_METRICS
        + CLOCK_METRICS
        + TEMPERATURE_METRICS
        + ECC_AND_ERROR_METRICS
        + PCIE_METRICS
    )

    def __init__(
        self,
        gpu_power_usage: Optional[Dict[str, List[float]]] = None,
        gpu_power_limit: Optional[Dict[str, List[float]]] = None,
        energy_consumption: Optional[Dict[str, List[float]]] = None,
        gpu_utilization: Optional[Dict[str, List[float]]] = None,
        sm_utilization: Optional[Dict[str, List[float]]] = None,
        memory_copy_utilization: Optional[Dict[str, List[float]]] = None,
        video_encoder_utilization: Optional[Dict[str, List[float]]] = None,
        video_decoder_utilization: Optional[Dict[str, List[float]]] = None,
        total_gpu_memory: Optional[Dict[str, List[float]]] = None,
        gpu_memory_used: Optional[Dict[str, List[float]]] = None,
        gpu_memory_free: Optional[Dict[str, List[float]]] = None,
        gpu_memory_temperature: Optional[Dict[str, List[float]]] = None,
        gpu_temperature: Optional[Dict[str, List[float]]] = None,
        gpu_clock_sm: Optional[Dict[str, List[float]]] = None,
        gpu_clock_memory: Optional[Dict[str, List[float]]] = None,
        total_ecc_sbe_volatile: Optional[Dict[str, List[float]]] = None,
        total_ecc_dbe_volatile: Optional[Dict[str, List[float]]] = None,
        total_ecc_sbe_aggregate: Optional[Dict[str, List[float]]] = None,
        total_ecc_dbe_aggregate: Optional[Dict[str, List[float]]] = None,
        total_nvlink_crc_flit_errors: Optional[Dict[str, List[float]]] = None,
        total_nvlink_crc_data_errors: Optional[Dict[str, List[float]]] = None,
        xid_last_error: Optional[Dict[str, List[float]]] = None,
        power_throttle_duration: Optional[Dict[str, List[float]]] = None,
        thermal_throttle_duration: Optional[Dict[str, List[float]]] = None,
        retired_pages_sbe: Optional[Dict[str, List[float]]] = None,
        retired_pages_dbe: Optional[Dict[str, List[float]]] = None,
        pcie_transmit_throughput: Optional[Dict[str, List[float]]] = None,
        pcie_receive_throughput: Optional[Dict[str, List[float]]] = None,
        pcie_replay_counter: Optional[Dict[str, List[float]]] = None,
    ):
        self.gpu_power_usage = defaultdict(list, gpu_power_usage or {})
        self.gpu_power_limit = defaultdict(list, gpu_power_limit or {})
        self.energy_consumption = defaultdict(list, energy_consumption or {})
        self.gpu_utilization = defaultdict(list, gpu_utilization or {})
        self.sm_utilization = defaultdict(list, sm_utilization or {})
        self.memory_copy_utilization = defaultdict(list, memory_copy_utilization or {})
        self.video_encoder_utilization = defaultdict(
            list, video_encoder_utilization or {}
        )
        self.video_decoder_utilization = defaultdict(
            list, video_decoder_utilization or {}
        )
        self.total_gpu_memory = defaultdict(list, total_gpu_memory or {})
        self.gpu_memory_used = defaultdict(list, gpu_memory_used or {})
        self.gpu_memory_free = defaultdict(list, gpu_memory_free or {})
        self.gpu_memory_temperature = defaultdict(list, gpu_memory_temperature or {})
        self.gpu_temperature = defaultdict(list, gpu_temperature or {})
        self.gpu_clock_sm = defaultdict(list, gpu_clock_sm or {})
        self.gpu_clock_memory = defaultdict(list, gpu_clock_memory or {})
        self.total_ecc_sbe_volatile = defaultdict(list, total_ecc_sbe_volatile or {})
        self.total_ecc_dbe_volatile = defaultdict(list, total_ecc_dbe_volatile or {})
        self.total_ecc_sbe_aggregate = defaultdict(list, total_ecc_sbe_aggregate or {})
        self.total_ecc_dbe_aggregate = defaultdict(list, total_ecc_dbe_aggregate or {})
        self.xid_last_error = defaultdict(list, xid_last_error or {})
        self.power_throttle_duration = defaultdict(list, power_throttle_duration or {})
        self.thermal_throttle_duration = defaultdict(
            list, thermal_throttle_duration or {}
        )
        self.retired_pages_sbe = defaultdict(list, retired_pages_sbe or {})
        self.retired_pages_dbe = defaultdict(list, retired_pages_dbe or {})
        self.total_nvlink_crc_flit_errors = defaultdict(
            list, total_nvlink_crc_flit_errors or {}
        )
        self.total_nvlink_crc_data_errors = defaultdict(
            list, total_nvlink_crc_data_errors or {}
        )
        self.pcie_transmit_throughput = defaultdict(
            list, pcie_transmit_throughput or {}
        )
        self.pcie_receive_throughput = defaultdict(list, pcie_receive_throughput or {})
        self.pcie_replay_counter = defaultdict(list, pcie_replay_counter or {})

    def update_metrics(self, measurement_data: dict) -> None:
        for metric in self.TELEMETRY_METRICS:
            metric_key = metric.name
            if metric_key in measurement_data:
                metric_data = measurement_data[metric_key]
                for gpu_name, values in metric_data.items():
                    getattr(self, metric_key)[gpu_name].extend(values)

    def __repr__(self):
        attr_strs = []
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                attr_strs.append(f"{k}={v}")
        return f"TelemetryMetrics({','.join(attr_strs)})"

    @property
    def telemetry_metrics(self) -> List[MetricMetadata]:
        return self.TELEMETRY_METRICS

    @property
    def data(self) -> dict:
        """Returns all the metrics."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
