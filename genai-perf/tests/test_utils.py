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

from typing import Union

import pytest
from genai_perf.measurements.run_config_measurement import RunConfigMeasurement
from genai_perf.metrics.statistics import Statistics
from genai_perf.record.types.gpu_power_usage import GPUPowerUsage
from genai_perf.record.types.gpu_utilization import GPUUtilization
from genai_perf.record.types.perf_latency_p99 import PerfLatencyP99
from genai_perf.record.types.perf_throughput import PerfThroughput
from genai_perf.types import GpuId, PerfRecords


def ns_to_sec(ns: int) -> Union[int, float]:
    """Convert from nanosecond to second."""
    return ns / 1e9


def check_statistics(s1: Statistics, s2: Statistics) -> None:
    s1_dict = s1.stats_dict
    s2_dict = s2.stats_dict
    for metric in s1_dict.keys():
        for stat_name, value in s1_dict[metric].items():
            if stat_name != "unit":
                assert s2_dict[metric][stat_name] == pytest.approx(value)


###########################################################################
# Perf Metrics Constructor
###########################################################################
def create_perf_metrics(throughput: int, latency: int) -> PerfRecords:
    throughput_record = PerfThroughput(throughput)
    latency_record = PerfLatencyP99(latency)

    perf_metrics = {
        PerfThroughput.tag: throughput_record,
        PerfLatencyP99.tag: latency_record,
    }

    return perf_metrics


###########################################################################
# RCM Constructor
###########################################################################
def create_run_config_measurement(
    gpu_power: int, gpu_utilization: int, gpu_id: GpuId = "0"
) -> RunConfigMeasurement:
    gpu_power_record = GPUPowerUsage(gpu_power)
    gpu_util_record = GPUUtilization(gpu_utilization)

    gpu_metrics = {
        gpu_id: {
            GPUPowerUsage.tag: gpu_power_record,
            GPUUtilization.tag: gpu_util_record,
        }
    }

    return RunConfigMeasurement(gpu_metrics)
