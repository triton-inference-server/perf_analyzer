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

import pytest
from genai_perf.metrics import ImageRetrievalMetrics, LLMMetrics, Metrics, Statistics

BASE_METRIC_UNITS = {
    "request_throughput": "requests/sec",
    "request_latency": "ms",
    "request_goodput": "requests/sec",
}


class TestStatistics:

    @pytest.mark.parametrize(
        "metrics, expected_units",
        [
            (
                Metrics(
                    request_throughputs=[1, 2, 3],
                    request_latencies=[4, 5, 6],
                    request_goodputs=[7, 8, 9],
                ),
                BASE_METRIC_UNITS,
            ),
            (
                LLMMetrics(
                    request_throughputs=[1, 2, 3],
                    request_latencies=[4, 5, 6],
                    time_to_first_tokens=[7, 8, 9],
                    inter_token_latencies=[1, 2, 3],
                    output_token_throughputs=[4, 5, 6],
                    output_token_throughputs_per_user=[7, 8, 9],
                    output_sequence_lengths=[1, 2, 3],
                    input_sequence_lengths=[4, 5, 6],
                    request_goodputs=[7, 8, 9],
                ),
                {
                    **BASE_METRIC_UNITS,
                    "time_to_first_token": "ms",
                    "inter_token_latency": "ms",
                    "output_token_throughput": "tokens/sec",
                    "output_token_throughput_per_user": "tokens/sec/user",
                    "output_sequence_length": "tokens",
                    "input_sequence_length": "tokens",
                },
            ),
            (
                ImageRetrievalMetrics(
                    request_throughputs=[1, 2, 3],
                    request_latencies=[4, 5, 6],
                    image_throughputs=[7, 8, 9],
                    image_latencies=[1, 2, 3],
                    request_goodputs=[4, 5, 6],
                ),
                {
                    **BASE_METRIC_UNITS,
                    "image_throughput": "pages/sec",
                    "image_latency": "ms",
                },
            ),
        ],
    )
    def test_metric_units(self, metrics, expected_units):
        stats = Statistics(metrics=metrics).stats_dict
        for metric, unit in expected_units.items():
            assert stats[metric]["unit"] == unit
