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

from pathlib import Path
from typing import Dict

from genai_perf.metrics import ImageRetrievalMetrics
from genai_perf.profile_data_parser.profile_data_parser import ProfileDataParser
from genai_perf.utils import load_json_str


class ImageRetrievalProfileDataParser(ProfileDataParser):
    """Calculate and aggregate all the Image Retrieval performance statistics
    across the Perf Analyzer profile results.
    """

    def __init__(
        self,
        filename: Path,
        goodput_constraints: Dict[str, float] = {},
    ) -> None:
        super().__init__(filename, goodput_constraints)

    def _parse_requests(self, requests: dict) -> ImageRetrievalMetrics:
        """Parse each request in profile data to extract core metrics."""
        min_req_timestamp, max_res_timestamp = float("inf"), 0
        request_latencies = []
        image_throughputs = []
        image_latencies = []

        for request in requests:
            req_timestamp = request["timestamp"]
            res_timestamps = request["response_timestamps"]
            req_inputs = request["request_inputs"]

            # track entire benchmark duration
            min_req_timestamp = min(min_req_timestamp, req_timestamp)
            max_res_timestamp = max(max_res_timestamp, res_timestamps[-1])

            # request latencies
            req_latency_ns = res_timestamps[-1] - req_timestamp
            request_latencies.append(req_latency_ns)

            payload = load_json_str(req_inputs["payload"])
            contents = payload["messages"][0]["content"]
            num_images = len([c for c in contents if c["type"] == "image_url"])

            # image throughput
            req_latency_s = req_latency_ns / 1e9  # to seconds
            image_throughputs.append(num_images / req_latency_s)

            # image latencies
            image_latencies.append(req_latency_ns / num_images)

        # request throughput
        benchmark_duration = (max_res_timestamp - min_req_timestamp) / 1e9  # to seconds
        request_throughputs = [len(requests) / benchmark_duration]

        image_metric = ImageRetrievalMetrics(
            request_throughputs,
            request_latencies,
            image_throughputs,
            image_latencies,
        )

        if self._goodput_constraints:
            goodput_val = self._calculate_goodput(benchmark_duration, image_metric)
            image_metric.request_goodputs = goodput_val

        return image_metric
