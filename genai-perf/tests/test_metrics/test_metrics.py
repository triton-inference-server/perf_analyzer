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

import pytest
from genai_perf.metrics import Metrics


class TestMetrics:

    def test_metric_request_metrics(self) -> None:
        """Test request_metrics property."""
        m = Metrics(
            request_throughputs=[10.12, 11.33],
            request_latencies=[3, 44],
            request_goodputs=[9.88, 10.22],
        )
        req_metrics = m.request_metrics
        assert len(req_metrics) == 1
        assert req_metrics[0].name == "request_latency"
        assert req_metrics[0].unit == "ms"

    def test_metric_system_metrics(self) -> None:
        """Test system_metrics property."""
        m = Metrics(
            request_throughputs=[10.12, 11.33],
            request_latencies=[3, 44],
            request_goodputs=[9.88, 10.22],
        )
        sys_metrics = m.system_metrics
        assert len(sys_metrics) == 3
        assert sys_metrics[0].name == "request_throughput"
        assert sys_metrics[0].unit == "per sec"
        assert sys_metrics[1].name == "request_goodput"
        assert sys_metrics[1].unit == "per sec"
        assert sys_metrics[2].name == "request_count"
        assert sys_metrics[2].unit == "count"
        assert m.data.get("request_count") == [2]

    def test_metrics_get_base_name(self) -> None:
        """Test get_base_name method in Metrics class."""
        metrics = Metrics(
            request_throughputs=[10.12, 11.33],
            request_latencies=[3, 44],
            request_goodputs=[9.88, 10.22],
        )
        assert metrics.get_base_name("request_throughputs") == "request_throughput"
        assert metrics.get_base_name("request_latencies") == "request_latency"
        assert metrics.get_base_name("request_goodputs") == "request_goodput"
        assert metrics.get_base_name("request_count") == "request_count"
        with pytest.raises(KeyError):
            metrics.get_base_name("hello1234")
