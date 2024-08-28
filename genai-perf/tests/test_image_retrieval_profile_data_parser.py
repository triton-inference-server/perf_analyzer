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

import json
from pathlib import Path
from typing import cast
from unittest.mock import mock_open, patch

import pytest
from genai_perf.metrics import ImageRetrievalMetrics, Statistics
from genai_perf.profile_data_parser import ImageRetrievalProfileDataParser

from .test_utils import check_statistics, ns_to_sec


def check_image_retrieval_metrics(
    m1: ImageRetrievalMetrics, m2: ImageRetrievalMetrics
) -> None:
    assert m1.request_latencies == m2.request_latencies
    assert m1.request_throughputs == pytest.approx(m2.request_throughputs)
    assert m1.image_latencies == m2.image_latencies
    assert m1.image_throughputs == pytest.approx(m2.image_throughputs)
    assert m1.request_goodputs == m2.request_goodputs


class TestImageRetrievalProfileDataParser:

    image_retrieval_profile_data = {
        "experiments": [
            {
                "experiment": {"mode": "concurrency", "value": 10},
                "requests": [
                    {
                        "timestamp": 1,
                        "request_inputs": {
                            "payload": '{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"image1"}},{"type":"image_url","image_url":{"url":"image2"}}]}],"model":"yolox"}'
                        },
                        "response_timestamps": [3],
                        "response_outputs": [
                            {
                                "response": '{"object":"list","data":[],"model":"yolox","usage":null}'
                            }
                        ],
                    },
                    {
                        "timestamp": 3,
                        "request_inputs": {
                            "payload": '{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"image1"}},{"type":"image_url","image_url":{"url":"image2"}},{"type":"image_url","image_url":{"url":"image3"}}]}],"model":"yolox"}'
                        },
                        "response_timestamps": [7],
                        "response_outputs": [
                            {
                                "response": '{"object":"list","data":[],"model":"yolox","usage":null}'
                            }
                        ],
                    },
                ],
            }
        ],
        "version": "",
        "service_kind": "openai",
        "endpoint": "v1/infer",
    }

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps(image_retrieval_profile_data),
    )
    @pytest.mark.parametrize(
        "infer_mode, load_level, expected_metrics",
        [
            (
                "concurrency",
                "10",
                {
                    "request_throughputs": [1 / ns_to_sec(3)],
                    "request_latencies": [2, 4],
                    "image_throughputs": [1 / ns_to_sec(1), 3 / ns_to_sec(4)],
                    "image_latencies": [1, 4 / 3],
                },
            ),
        ],
    )
    def test_image_retrieval_profile_data(
        self,
        mock_exists,
        mock_file,
        infer_mode,
        load_level,
        expected_metrics,
    ) -> None:
        """Collect image retrieval metrics from profile export data and check values.

        Metrics
        * request throughputs
            - [2 / (7 - 1)] = [1 / ns_to_sec(3)]
        * request latencies
            - [3 - 1, 7 - 3] = [2, 4]
        * image throughputs
            - [2 / (3 - 1), 3 / (7 - 3)] = [1 / ns_to_sec(1), 3 / ns_to_sec(4)]
        * image latencies
            - [(3 - 1) / 2, (7 - 3) / 3] = [1, 4/3]

        """
        pd = ImageRetrievalProfileDataParser(
            filename=Path("image_retrieval_profile_export.json")
        )

        statistics = pd.get_statistics(infer_mode="concurrency", load_level="10")
        metrics = cast(ImageRetrievalMetrics, statistics.metrics)

        expected_metrics = ImageRetrievalMetrics(**expected_metrics)
        expected_statistics = Statistics(expected_metrics)

        check_image_retrieval_metrics(metrics, expected_metrics)
        check_statistics(statistics, expected_statistics)

    @patch("pathlib.Path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data=json.dumps(image_retrieval_profile_data),
    )
    @pytest.mark.parametrize(
        "infer_mode, load_level, expected_metrics",
        [
            (
                "concurrency",
                "10",
                {
                    "request_throughputs": [1 / ns_to_sec(3)],
                    "request_latencies": [2, 4],
                    "image_throughputs": [1 / ns_to_sec(1), 3 / ns_to_sec(4)],
                    "image_latencies": [1, 4 / 3],
                    "request_goodputs": [1 / ns_to_sec(6)],
                },
            ),
        ],
    )
    def test_image_retrieval_profile_data_with_goodput(
        self,
        mock_exists,
        mock_file,
        infer_mode,
        load_level,
        expected_metrics,
    ) -> None:
        """Collect image retrieval metrics from profile export data and check values.

        Goodput Constraints
        * image latency: 1.25e-6 ms
        * request latency: 2.5e-6 ms

        Metrics
        * request throughputs
            - [2 / (7 - 1)] = [1 / ns_to_sec(3)]
        * request latencies
            - [3 - 1, 7 - 3] = [2, 4]
        * image throughputs
            - [2 / (3 - 1), 3 / (7 - 3)] = [1 / ns_to_sec(1), 3 / ns_to_sec(4)]
        * image latencies
            - [(3 - 1) / 2, (7 - 3) / 3] = [1, 4/3]
        * request goodput
            - number of good requests
                image latency: [1 < 1.25, 4/3 > 1.25]
                request latency: [2 < 2.5, 4 > 2.5]
                1 request is good under such constraints
            - [1 / (7 - 1)] = [1 / ns_to_sec(6)]

        """
        test_goodput_constraints = {
            "image_latency": 1.25e-6,  # ms
            "request_latency": 2.5e-6,  # ms
        }

        pd = ImageRetrievalProfileDataParser(
            filename=Path("image_retrieval_profile_export.json"),
            goodput_constraints=test_goodput_constraints,
        )

        statistics = pd.get_statistics(infer_mode="concurrency", load_level="10")
        metrics = cast(ImageRetrievalMetrics, statistics.metrics)

        expected_metrics = ImageRetrievalMetrics(**expected_metrics)
        expected_statistics = Statistics(expected_metrics)

        check_image_retrieval_metrics(metrics, expected_metrics)
        check_statistics(statistics, expected_statistics)
