#!/usr/bin/env python3

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

from typing import List, Union

from genai_perf.metrics.metrics import MetricMetadata, Metrics


class ImageRetrievalMetrics(Metrics):
    """A simple dataclass that holds core Image Retrieval performance metrics."""

    IMAGE_RETRIEVAL_REQUEST_TIME_METRICS = [
        MetricMetadata("image_latency", "ms/image"),
    ]

    IMAGE_RETRIEVAL_REQUEST_THOUGHPUT_METRICS = [
        MetricMetadata("image_throughput", "images/sec"),
    ]

    IMAGE_RETRIEVAL_REQUEST_METRICS = (
        IMAGE_RETRIEVAL_REQUEST_THOUGHPUT_METRICS + IMAGE_RETRIEVAL_REQUEST_TIME_METRICS
    )

    def __init__(
        self,
        request_throughputs: List[float] = [],
        request_latencies: List[int] = [],
        image_throughputs: List[int] = [],
        image_latencies: List[int] = [],
        request_goodputs: Union[List[float], None] = [],
    ) -> None:
        super().__init__(request_throughputs, request_latencies, request_goodputs)
        self.image_throughputs = image_throughputs
        self.image_latencies = image_latencies

        # add base name mapping
        self._base_names["image_throughputs"] = "image_throughput"
        self._base_names["image_latencies"] = "image_latency"

    @property
    def request_metrics(self) -> List[MetricMetadata]:
        base_metrics = super().request_metrics  # base metrics
        return base_metrics + self.IMAGE_RETRIEVAL_REQUEST_METRICS

    @property
    def request_time_metrics(self) -> List[MetricMetadata]:
        base_metrics = super().request_time_metrics
        return self.IMAGE_RETRIEVAL_REQUEST_TIME_METRICS + base_metrics

    @property
    def request_throughput_metrics(self) -> List[MetricMetadata]:
        base_metrics = super().request_throughput_metrics
        return self.IMAGE_RETRIEVAL_REQUEST_THOUGHPUT_METRICS + base_metrics
