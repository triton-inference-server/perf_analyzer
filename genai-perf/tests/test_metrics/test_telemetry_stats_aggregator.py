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

import pytest
from genai_perf.metrics import TelemetryStatistics
from genai_perf.metrics.telemetry_stats_aggregator import TelemetryStatsAggregator


class TestTelemetryStatsAggregator:

    @pytest.fixture
    def telemetry_dicts(self):
        return [
            {
                "gpu_power_usage": {
                    "gpu0": {
                        "avg": 22.502,
                        "p25": 22.502,
                        "p50": 22.502,
                        "p75": 22.502,
                        "p90": 22.502,
                        "p95": 22.502,
                        "p99": 22.502,
                        "min": 22.502,
                        "max": 22.502,
                        "std": 0.0,
                    }
                },
                "gpu_power_limit": {"gpu0": {"avg": 300.0}},
                "energy_consumption": {
                    "gpu0": {
                        "avg": 6.287884382000105,
                        "p25": 6.287884382000105,
                        "p50": 6.287884382000105,
                        "p75": 6.287884382000105,
                        "p90": 6.287884382000105,
                        "p95": 6.287884382000105,
                        "p99": 6.287884382000105,
                        "min": 6.287884382000105,
                        "max": 6.287884382000105,
                        "std": 0.0,
                    }
                },
                "gpu_utilization": {
                    "gpu0": {
                        "avg": 0.0,
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                        "p99": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "std": 0.0,
                    }
                },
                "total_gpu_memory": {"gpu0": {"avg": 51.52702464}},
                "gpu_memory_used": {
                    "gpu0": {
                        "avg": 25.856835584000002,
                        "p25": 25.856835584000002,
                        "p50": 25.856835584000002,
                        "p75": 25.856835584000002,
                        "p90": 25.856835584000002,
                        "p95": 25.856835584000002,
                        "p99": 25.856835584000002,
                        "min": 25.856835584000002,
                        "max": 25.856835584000002,
                        "std": 0.0,
                    }
                },
            },
            {
                "gpu_power_usage": {
                    "gpu0": {
                        "avg": 22.334,
                        "p25": 22.334,
                        "p50": 22.334,
                        "p75": 22.334,
                        "p90": 22.334,
                        "p95": 22.334,
                        "p99": 22.334,
                        "min": 22.334,
                        "max": 22.334,
                        "std": 0.0,
                    }
                },
                "gpu_power_limit": {"gpu0": {"avg": 300.0}},
                "energy_consumption": {
                    "gpu0": {
                        "avg": 6.289814182000104,
                        "p25": 6.289814182000104,
                        "p50": 6.289814182000104,
                        "p75": 6.289814182000104,
                        "p90": 6.289814182000104,
                        "p95": 6.289814182000104,
                        "p99": 6.289814182000104,
                        "min": 6.289814182000104,
                        "max": 6.289814182000104,
                        "std": 0.0,
                    }
                },
                "gpu_utilization": {
                    "gpu0": {
                        "avg": 0.0,
                        "p25": 0.0,
                        "p50": 0.0,
                        "p75": 0.0,
                        "p90": 0.0,
                        "p95": 0.0,
                        "p99": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "std": 0.0,
                    }
                },
                "total_gpu_memory": {"gpu0": {"avg": 51.52702464}},
                "gpu_memory_used": {
                    "gpu0": {
                        "avg": 25.856835584000002,
                        "p25": 25.856835584000002,
                        "p50": 25.856835584000002,
                        "p75": 25.856835584000002,
                        "p90": 25.856835584000002,
                        "p95": 25.856835584000002,
                        "p99": 25.856835584000002,
                        "min": 25.856835584000002,
                        "max": 25.856835584000002,
                        "std": 0.0,
                    }
                },
            },
        ]

    @pytest.fixture
    def aggregator(self, telemetry_dicts):
        return TelemetryStatsAggregator(telemetry_dicts)

    def test_aggregate_telemetry_stats(self, aggregator):
        telemetry_stats = aggregator.get_telemetry_stats()

        gpu_power_usage = telemetry_stats.stats_dict["gpu_power_usage"]
        assert gpu_power_usage["gpu0"]["avg"] == 22.418

        gpu_power_limit = telemetry_stats.stats_dict["gpu_power_limit"]
        assert gpu_power_limit["gpu0"]["avg"] == 300.0

        energy_consumption = telemetry_stats.stats_dict["energy_consumption"]
        assert energy_consumption["gpu0"]["avg"] == 6.288849282000104

        gpu_utilization = telemetry_stats.stats_dict["gpu_utilization"]
        assert gpu_utilization["gpu0"]["avg"] == 0.0

        total_gpu_memory = telemetry_stats.stats_dict["total_gpu_memory"]
        assert total_gpu_memory["gpu0"]["avg"] == 51.52702464

        gpu_memory_used = telemetry_stats.stats_dict["gpu_memory_used"]
        assert gpu_memory_used["gpu0"]["avg"] == 25.856835584000002

    def test_get_telemetry_stats(self, aggregator):
        telemetry_stats = aggregator.get_telemetry_stats()

        assert isinstance(telemetry_stats, TelemetryStatistics)
        assert telemetry_stats.stats_dict["gpu_power_usage"]["gpu0"]["avg"] == 22.418
        assert (
            telemetry_stats.stats_dict["gpu_memory_used"]["gpu0"]["avg"]
            == 25.856835584000002
        )
        assert (
            telemetry_stats.stats_dict["total_gpu_memory"]["gpu0"]["avg"] == 51.52702464
        )
        assert telemetry_stats.stats_dict["gpu_power_limit"]["gpu0"]["avg"] == 300.0
        assert (
            telemetry_stats.stats_dict["energy_consumption"]["gpu0"]["avg"]
            == 6.288849282000104
        )
        assert telemetry_stats.stats_dict["gpu_utilization"]["gpu0"]["avg"] == 0.0
