#!/usr/bin/env python3

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

from typing import Dict, List, Union

from genai_perf.metrics.metrics import MetricMetadata, Metrics


class LLMMetrics(Metrics):
    """A simple dataclass that holds core LLM performance metrics."""

    LLM_REQUEST_TIME_METRICS = [
        MetricMetadata("time_to_first_token", "ms"),
        MetricMetadata("time_to_second_token", "ms"),
        MetricMetadata("inter_token_latency", "ms"),
    ]

    LLM_REQUEST_THROUGHPUT_METRICS = [
        MetricMetadata("output_token_throughput_per_user", "tokens/sec/user"),
    ]

    LLM_REQUEST_OTHER_METRICS = [
        MetricMetadata("output_sequence_length", "tokens"),
        MetricMetadata("input_sequence_length", "tokens"),
    ]

    LLM_REQUEST_METRICS = (
        LLM_REQUEST_TIME_METRICS
        + LLM_REQUEST_THROUGHPUT_METRICS
        + LLM_REQUEST_OTHER_METRICS
    )

    LLM_SYSTEM_METRICS = [
        # (TMA-1977) Make the unit consistent with statistics dict (e.g. tokens/sec)
        MetricMetadata("output_token_throughput", "tokens/sec"),
    ]

    def __init__(
        self,
        request_throughputs: List[float] = [],
        request_latencies: List[int] = [],
        time_to_first_tokens: List[int] = [],
        time_to_second_tokens: List[int] = [],
        inter_token_latencies: List[int] = [],
        output_token_throughputs: List[float] = [],
        output_token_throughputs_per_user: List[float] = [],
        output_sequence_lengths: List[int] = [],
        input_sequence_lengths: List[int] = [],
        chunked_inter_token_latencies: List[List[int]] = [[]],
        request_goodputs: Union[List[float], None] = [],
    ) -> None:
        super().__init__(request_throughputs, request_latencies, request_goodputs)
        self.time_to_first_tokens = time_to_first_tokens
        self.time_to_second_tokens = time_to_second_tokens
        self.inter_token_latencies = inter_token_latencies
        self.output_token_throughputs = output_token_throughputs
        self.output_token_throughputs_per_user = output_token_throughputs_per_user
        self.output_sequence_lengths = output_sequence_lengths
        self.input_sequence_lengths = input_sequence_lengths

        # Keeping chunked ITL (old) as a WAR to preserve visualization.
        # Excluded from data.
        self._chunked_inter_token_latencies = chunked_inter_token_latencies

        # add base name mapping
        self._base_names["time_to_first_tokens"] = "time_to_first_token"
        self._base_names["time_to_second_tokens"] = "time_to_second_token"
        self._base_names["inter_token_latencies"] = "inter_token_latency"  # nosec
        self._base_names["output_token_throughputs"] = (  # nosec
            "output_token_throughput"
        )
        self._base_names["output_token_throughputs_per_user"] = (  # nosec
            "output_token_throughput_per_user"
        )
        self._base_names["output_sequence_lengths"] = "output_sequence_length"
        self._base_names["input_sequence_lengths"] = "input_sequence_length"

    @classmethod
    def from_dict(cls, data: Dict[str, List[float | int]]) -> "LLMMetrics":
        """Create LLMMetrics from a dictionary, converting types as needed."""
        return cls(
            request_throughputs=[float(x) for x in data["request_throughputs"]],
            request_latencies=[int(x) for x in data["request_latencies"]],
            time_to_first_tokens=[int(x) for x in data["time_to_first_tokens"]],
            time_to_second_tokens=[int(x) for x in data["time_to_second_tokens"]],
            inter_token_latencies=[int(x) for x in data["inter_token_latencies"]],
            output_token_throughputs=[
                float(x) for x in data["output_token_throughputs"]
            ],
            output_token_throughputs_per_user=[
                float(x) for x in data["output_token_throughputs_per_user"]
            ],
            output_sequence_lengths=[int(x) for x in data["output_sequence_lengths"]],
            input_sequence_lengths=[int(x) for x in data["input_sequence_lengths"]],
        )

    @property
    def request_metrics(self) -> List[MetricMetadata]:
        base_metrics = super().request_metrics  # base metrics

        # (TMA-1975) The order is hardcoded as below to avoid introducing any
        # breaking changes to the users who might be parsing the outputs. However,
        # we would eventually want to impose some consistent order such as a
        # base metrics first and then task specific metrics. Uncomment the below
        # line to enable this order:
        # return base_metrics + self.LLM_REQUEST_METRICS
        return (
            self.LLM_REQUEST_METRICS[:2] + base_metrics + self.LLM_REQUEST_METRICS[2:]
        )

    @property
    def system_metrics(self) -> List[MetricMetadata]:
        base_metrics = super().system_metrics  # base metrics

        # (TMA-1975) The order is hardcoded as below to avoid introducing any
        # breaking changes to the users who might be parsing the outputs. However,
        # we would eventually want to impose some consistent order such as a
        # base metrics first and then task specific metrics. Uncomment the below
        # line to enable this order:
        # return base_metrics + self.LLM_SYSTEM_METRICS
        return self.LLM_SYSTEM_METRICS + base_metrics

    @property
    def request_time_metrics(self) -> List[MetricMetadata]:
        base_metrics = super().request_time_metrics
        return self.LLM_REQUEST_TIME_METRICS + base_metrics

    @property
    def request_throughput_metrics(self) -> List[MetricMetadata]:
        base_metrics = super().request_throughput_metrics
        return self.LLM_REQUEST_THROUGHPUT_METRICS + base_metrics
