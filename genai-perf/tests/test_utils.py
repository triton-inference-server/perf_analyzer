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

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytest
from genai_perf import utils
from genai_perf.config.generate.genai_perf_config import GenAIPerfConfig
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.run.run_config import RunConfig
from genai_perf.export_data.exporter_config import ExporterConfig
from genai_perf.measurements.run_config_measurement import RunConfigMeasurement
from genai_perf.metrics.metrics import Metrics
from genai_perf.metrics.statistics import Statistics
from genai_perf.record.types.gpu_power_usage_p99 import GPUPowerUsageP99
from genai_perf.record.types.gpu_utilization_p99 import GPUUtilizationP99
from genai_perf.record.types.input_sequence_length_p99 import InputSequenceLengthP99
from genai_perf.record.types.output_sequence_length_p99 import OutputSequenceLengthP99
from genai_perf.record.types.request_latency_p99 import RequestLatencyP99
from genai_perf.record.types.request_throughput_avg import RequestThroughputAvg
from genai_perf.types import GpuId, ModelObjectiveParameters, PerfRecords


###########################################################
# Library of utility functions for other unit test scripts.
###########################################################
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
# ExporterConfig Constructor
###########################################################################
def create_default_exporter_config(
    stats: Optional[Dict[Any, Any]] = None,
    metrics: Optional[Metrics] = None,
    config: Optional[ConfigCommand] = None,
    telemetry_stats: Dict[str, Any] = {},
    session_stats: Dict[str, Any] = {},
) -> ExporterConfig:
    if not config:
        config = ConfigCommand({"model_name": "test_model"})

    perf_analyzer_config = PerfAnalyzerConfig(config=config)

    return ExporterConfig(
        stats=stats or {},
        metrics=metrics or Metrics(),
        config=config,
        perf_analyzer_config=perf_analyzer_config,
        extra_inputs=config.input.extra,
        telemetry_stats=telemetry_stats,
        session_stats=session_stats,
    )


###########################################################################
# Perf Metrics Constructor
###########################################################################
def create_perf_metrics(
    throughput: Optional[int] = None,
    latency: Optional[int] = None,
    input_seq_length: Optional[int] = None,
    output_seq_length: Optional[int] = None,
) -> PerfRecords:
    perf_metrics: PerfRecords = {}
    if throughput:
        perf_metrics[RequestThroughputAvg.tag] = RequestThroughputAvg(throughput)
    if latency:
        perf_metrics[RequestLatencyP99.tag] = RequestLatencyP99(latency)
    if input_seq_length:
        perf_metrics[InputSequenceLengthP99.tag] = InputSequenceLengthP99(
            input_seq_length
        )
    if output_seq_length:
        perf_metrics[OutputSequenceLengthP99.tag] = OutputSequenceLengthP99(
            output_seq_length
        )

    return perf_metrics


###########################################################################
# RCM Constructor
###########################################################################
def create_run_config_measurement(
    gpu_power: int, gpu_utilization: int, gpu_id: GpuId = "0"
) -> RunConfigMeasurement:
    gpu_power_record = GPUPowerUsageP99(gpu_power)
    gpu_util_record = GPUUtilizationP99(gpu_utilization)

    gpu_metrics = {
        gpu_id: {
            GPUPowerUsageP99.tag: gpu_power_record,
            GPUUtilizationP99.tag: gpu_util_record,
        }
    }

    return RunConfigMeasurement(gpu_metrics)


###########################################################################
# RunConfig Constructor
###########################################################################
def create_run_config(
    run_config_name: str,
    model_name: str = "test_model",
    model_objective_parameters: ModelObjectiveParameters = {},
    gpu_power: int = 0,
    gpu_utilization: int = 0,
    gpu_id: GpuId = "0",
    throughput: int = 0,
    latency: int = 0,
    input_seq_length: int = 0,
    output_seq_length: int = 0,
) -> RunConfig:
    config = ConfigCommand(user_config={"model_name": "test_model"})
    config.model_names = [model_name]
    genai_perf_config = GenAIPerfConfig(
        config=config, model_objective_parameters=model_objective_parameters
    )
    perf_analyzer_config = PerfAnalyzerConfig(
        config=config,
        model_objective_parameters=model_objective_parameters,
    )

    rcm = create_run_config_measurement(gpu_power, gpu_utilization, gpu_id)

    perf_metrics = create_perf_metrics(
        throughput=throughput,
        latency=latency,
        input_seq_length=input_seq_length,
        output_seq_length=output_seq_length,
    )
    rcm.add_perf_metrics(model_name=model_name, perf_metrics=perf_metrics)

    run_config = RunConfig(
        name=run_config_name,
        genai_perf_config=genai_perf_config,
        perf_analyzer_config=perf_analyzer_config,
        measurement=rcm,
    )

    return run_config


class TestUtils:
    """
    Unit test for genai_perf/utils.py utility functions
    """

    def test_sample_bounded_normal(self):
        # lower bounded by -10
        n = utils.sample_bounded_normal(mean=-1000, stddev=0, lower=-10)
        assert n == -10

        # upper bounded by 10
        n = utils.sample_bounded_normal(mean=1000, stddev=0, upper=10)
        assert n == 10
