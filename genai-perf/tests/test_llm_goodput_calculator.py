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
from genai_perf.goodput_calculator.llm_goodput_calculator import LLMGoodputCalculator
from genai_perf.metrics.llm_metrics import LLMMetrics


def ns_to_sec(ns: int) -> Union[int, float]:
    """Convert from nanosecond to second."""
    return ns / 1e9


class TestLLMGoodputCalculator:

    TEST_METRIC = LLMMetrics(
        request_throughputs=[10.12, 11.33],
        request_latencies=[3, 44],
        request_goodputs=[9.88, 10.23],
    )

    TEST_BENCHMARK_DURATION = 10

    TEST_GOODPUT_CONSTRAINTS = {"request_latency": 10e-6}  # ms

    def test_goodput_property(self) -> None:
        gc = LLMGoodputCalculator(
            goodput_constraints=self.TEST_GOODPUT_CONSTRAINTS,
            metric=self.TEST_METRIC,
            benchmark_duration=self.TEST_BENCHMARK_DURATION,
        )

        assert gc.goodput is None
        gc.compute()
        assert gc.goodput == [0.1]

    def test_get_slo_name(self) -> None:
        gc = LLMGoodputCalculator(
            goodput_constraints=self.TEST_GOODPUT_CONSTRAINTS,
            metric=self.TEST_METRIC,
            benchmark_duration=self.TEST_BENCHMARK_DURATION,
        )

        assert gc.get_slo_name("request_latency") == "request_latencies"
        assert gc.get_slo_name("time_to_first_token") == "time_to_first_tokens"
        assert gc.get_slo_name("inter_token_latency") == "inter_token_latencies"
        assert gc.get_slo_name("output_token_throughput_per_request") == (
            "output_token_throughputs_per_request"
        )
        with pytest.raises(KeyError):
            gc.get_slo_name("hello1234")

    def test_compute(self) -> None:
        """
        Goodput constraints for experiment 1 and 2:
        * time_to_first_token: 2.5e-6 ms
        * inter_token_latency: 2.5e-6 ms
        * output_token_throughput_per_request: 0.5e9 s

        Benchmark duration for experiment 1 and 2: 10 s

        LLMMetrics
        * time to first tokens
            - experiment 1: [3 - 1, 4 - 2] = [2, 2]
            - experiment 2: [7 - 5, 6 - 3] = [2, 3]
        * inter token latencies
            - experiment 1: [((8 - 1) - 2)/(3 - 1), ((11 - 2) - 2)/(6 - 1)]
                          : [2.5, 1.4]
                          : [2, 1]  # rounded
            - experiment 2: [((18 - 5) - 2)/(4 - 1), ((11 - 3) - 3)/(6 - 1)]
                          : [11/3, 1]
                          : [4, 1]  # rounded
        * output token throughputs per request
            - experiment 1: [3/(8 - 1), 6/(11 - 2)] = [3/7, 6/9]
            - experiment 2: [4/(18 - 5), 6/(11 - 3)] = [4/13, 6/8]

        Request good counts according to constraints:
            - experiment 1: 1
            - experiment 2: 0

        Request goodputs
            - experiment 1: [1 / 10] = [0.1]
            - experiment 2: [0 / 10] = [0]
        """
        test_goodput_constraints = {
            "time_to_first_token": 2.5e-6,  # ms
            "inter_token_latency": 1.5e-6,  # ms
            "output_token_throughput_per_request": 0.5e9,  # s
        }

        # experiment 1
        test_llm_metrics_1 = LLMMetrics(
            time_to_first_tokens=[2, 2],
            inter_token_latencies=[2, 1],
            output_token_throughputs_per_request=[3 / ns_to_sec(7), 6 / ns_to_sec(9)],
        )

        gc_1 = LLMGoodputCalculator(
            goodput_constraints=test_goodput_constraints,
            metric=test_llm_metrics_1,
            benchmark_duration=self.TEST_BENCHMARK_DURATION,
        )
        assert gc_1.goodput is None
        gc_1.compute()
        assert gc_1.goodput == [0.1]

        # experiment 2
        test_llm_metrics_2 = LLMMetrics(
            time_to_first_tokens=[2, 3],
            inter_token_latencies=[4, 1],
            output_token_throughputs_per_request=[4 / ns_to_sec(13), 6 / ns_to_sec(8)],
        )

        gc_2 = LLMGoodputCalculator(
            goodput_constraints=test_goodput_constraints,
            metric=test_llm_metrics_2,
            benchmark_duration=self.TEST_BENCHMARK_DURATION,
        )
        assert gc_2.goodput is None
        gc_2.compute()
        assert gc_2.goodput == [0.0]