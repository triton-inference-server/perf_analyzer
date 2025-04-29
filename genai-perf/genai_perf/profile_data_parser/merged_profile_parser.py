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

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, TypeAlias

from genai_perf.logging import logging
from genai_perf.profile_data_parser.llm_profile_data_parser import LLMProfileDataParser
from genai_perf.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

SessionMetrics: TypeAlias = Dict[str, Dict[str, List[float | int]]]


class MergedProfileParser(LLMProfileDataParser):
    """
    A class that parses a merged profile data file produced from multiple distributed runs
    and calculates aggregated metrics for throughput and goodput.

    Parameters:
    - filename (Path): The path to the merged profile data file.
    - tokenizer (Tokenizer): The tokenizer used for processing text data.
    - throughput_metrics_dict (Dict[str, List[float]]): A dictionary containing throughput metrics for requests and outputs.
    - goodput_constraints (Dict[str, float], optional): Constraints for goodput calculation. Defaults to an empty dictionary.
    """

    def __init__(
        self,
        filename: Path,
        tokenizer: Tokenizer,
        throughput_metrics_dict: Dict[str, List[float]],
        goodput_constraints: Dict[str, float] = {},
    ) -> None:
        self._throughput_metrics_dict = throughput_metrics_dict
        self._session_metrics: SessionMetrics = defaultdict(lambda: defaultdict(list))
        super().__init__(filename, tokenizer, goodput_constraints)

    def _calculate_throughput_metrics(
        self,
        requests: dict,
        output_sequence_lengths: List[int],
        benchmark_duration: float,
    ) -> Tuple[List[float], List[float]]:
        """Calculate request throughput and output token throughput."""

        request_throughputs = [
            sum(self._throughput_metrics_dict.get("request_throughput", []))
        ]
        output_token_throughputs = [
            sum(self._throughput_metrics_dict.get("output_token_throughput", []))
        ]

        return request_throughputs, output_token_throughputs
