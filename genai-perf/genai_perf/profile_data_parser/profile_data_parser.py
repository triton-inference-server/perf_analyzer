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

from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from genai_perf.goodput_calculator.llm_goodput_calculator import LLMGoodputCalculator
from genai_perf.logging import logging
from genai_perf.metrics import Metrics, Statistics
from genai_perf.utils import load_json

logger = logging.getLogger(__name__)


class ResponseFormat(Enum):
    HUGGINGFACE_GENERATE = auto()
    HUGGINGFACE_RANKINGS = auto()
    OPENAI_CHAT_COMPLETIONS = auto()
    OPENAI_COMPLETIONS = auto()
    OPENAI_EMBEDDINGS = auto()
    OPENAI_MULTIMODAL = auto()
    RANKINGS = auto()
    IMAGE_RETRIEVAL = auto()
    TRITON = auto()
    TRITON_GENERATE = auto()


class ProfileDataParser:
    """Base profile data parser class that reads the profile data JSON file to
    extract core metrics and calculate various performance statistics.
    """

    def __init__(
        self,
        filename: Path,
        goodput_constraints: Dict[str, float] = {},
    ) -> None:
        self._goodput_constraints = goodput_constraints
        self._session_statistics: Dict[str, Statistics] = {}
        logger.info("Loading response data from '%s'", str(filename))
        data = load_json(filename)
        self._get_profile_metadata(data)
        self._parse_profile_data(data)

    def _get_profile_metadata(self, data: dict) -> None:
        self._service_kind = data["service_kind"]
        if self._service_kind == "openai":
            if data["endpoint"] == "rerank":
                self._response_format = ResponseFormat.HUGGINGFACE_RANKINGS
            elif data["endpoint"] == "v1/chat/completions":
                # (TPA-66) add PA metadata to deduce the response format instead
                # of parsing the request input payload in profile export json
                # file.
                request = data["experiments"][0]["requests"][0]
                request_input = request["request_inputs"]["payload"]
                if "image_url" in request_input or "input_audio" in request_input:
                    self._response_format = ResponseFormat.OPENAI_MULTIMODAL
                else:
                    self._response_format = ResponseFormat.OPENAI_CHAT_COMPLETIONS
            elif data["endpoint"] == "v1/completions":
                self._response_format = ResponseFormat.OPENAI_COMPLETIONS
            elif data["endpoint"] == "v1/embeddings":
                self._response_format = ResponseFormat.OPENAI_EMBEDDINGS
            elif data["endpoint"] == "v1/ranking":
                self._response_format = ResponseFormat.RANKINGS
            elif data["endpoint"] == "v1/infer":
                self._response_format = ResponseFormat.IMAGE_RETRIEVAL
            elif "huggingface/generate" in data["endpoint"].lower():
                self._response_format = ResponseFormat.HUGGINGFACE_GENERATE
            else:
                # (TPA-66) add PA metadata to handle this case
                # When endpoint field is either empty or custom endpoint, fall
                # back to parsing the response to extract the response format.
                request = data["experiments"][0]["requests"][0]
                request_input = request["request_inputs"]["payload"]
                response = request["response_outputs"][0]["response"]
                if "chat.completion" in response:
                    if "image_url" in request_input or "input_audio" in request_input:
                        self._response_format = ResponseFormat.OPENAI_MULTIMODAL
                    else:
                        self._response_format = ResponseFormat.OPENAI_CHAT_COMPLETIONS
                elif "text_completion" in response:
                    self._response_format = ResponseFormat.OPENAI_COMPLETIONS
                elif "embedding" in response:
                    self._response_format = ResponseFormat.OPENAI_EMBEDDINGS
                elif "ranking" in response:
                    self._response_format = ResponseFormat.RANKINGS
                elif "image_retrieval" in response:
                    self._response_format = ResponseFormat.IMAGE_RETRIEVAL
                elif "generated_text" in response:
                    self._response_format = ResponseFormat.HUGGINGFACE_GENERATE
                elif "/generate" in data["endpoint"]:
                    self._response_format = ResponseFormat.TRITON_GENERATE
                else:
                    raise RuntimeError("Unknown OpenAI response format.")

        elif self._service_kind in ["dynamic_grpc", "triton"]:
            self._response_format = ResponseFormat.TRITON
        elif self._service_kind == "triton_c_api":
            pass  # ignore
        else:
            raise ValueError(f"Unknown service kind: {self._service_kind}")

    def _parse_profile_data(self, data: dict) -> None:
        """Parse through the entire profile data to collect statistics."""
        self._profile_results = {}
        for experiment in data["experiments"]:
            infer_mode = experiment["experiment"]["mode"]
            load_level = experiment["experiment"]["value"]
            requests = experiment["requests"]

            metrics = self._parse_requests(requests)

            # aggregate and calculate statistics
            statistics = Statistics(metrics)
            self._profile_results[(infer_mode, str(load_level))] = statistics

    def _parse_requests(self, requests: dict) -> Metrics:
        """Parse each request in profile data to extract core metrics."""
        min_req_timestamp, max_res_timestamp = float("inf"), 0
        request_latencies = []

        for request in requests:
            req_timestamp = request["timestamp"]
            res_timestamps = request["response_timestamps"]

            # track entire benchmark duration
            min_req_timestamp = min(min_req_timestamp, req_timestamp)
            max_res_timestamp = max(max_res_timestamp, res_timestamps[-1])

            # request latencies
            req_latency = res_timestamps[-1] - req_timestamp
            request_latencies.append(req_latency)

        # request throughput
        benchmark_duration = (max_res_timestamp - min_req_timestamp) / 1e9  # to seconds
        request_throughputs = [len(requests) / benchmark_duration]

        metric = Metrics(
            request_throughputs,
            request_latencies,
        )

        if self._goodput_constraints:
            goodput_val = self._calculate_goodput(benchmark_duration, metric)
            metric.request_goodputs = goodput_val

        return metric

    def _calculate_goodput(
        self,
        benchmark_duration: float,
        metric: Metrics,
    ) -> Optional[List[float]]:
        llm_goodput_calculator = LLMGoodputCalculator(
            self._goodput_constraints,
            metric,
            benchmark_duration,
        )

        llm_goodput_calculator.compute()
        return llm_goodput_calculator.goodput

    def get_statistics(self, infer_mode: str, load_level: str) -> Statistics:
        """Return profile statistics if it exists."""
        if (infer_mode, load_level) not in self._profile_results:
            raise KeyError(f"Profile with {infer_mode}={load_level} does not exist.")
        return self._profile_results[(infer_mode, load_level)]

    def get_session_statistics(self) -> Dict[str, Statistics]:
        """Return session statistics."""
        return self._session_statistics

    def get_profile_load_info(self) -> List[Tuple[str, str]]:
        """Return available (infer_mode, load_level) tuple keys."""
        return [k for k, _ in self._profile_results.items()]
