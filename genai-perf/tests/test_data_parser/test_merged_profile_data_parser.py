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

from unittest.mock import MagicMock, patch

from genai_perf.profile_data_parser.merged_profile_parser import MergedProfileParser


class TestMergedProfileParser:

    def test_calculate_throughput_metrics_valid(self):
        mock_throughput_metrics_dict = {
            "request_throughput": [100.0, 200.0, 300.0],
            "output_token_throughput": [10.0, 20.0, 30.0],
        }

        mock_tokenizer = MagicMock()

        with patch.object(
            MergedProfileParser,
            "__init__",
            lambda x, filename, tokenizer, throughput_metrics_dict, goodput_constraints=None: None,
        ):
            parser = MergedProfileParser(
                filename="mock_merged_profile.json",
                tokenizer=mock_tokenizer,
                throughput_metrics_dict=mock_throughput_metrics_dict,
            )

        parser._throughput_metrics_dict = mock_throughput_metrics_dict
        parser._session_metrics = {}

        requests = {}
        output_sequence_lengths = [10, 20, 30]
        benchmark_duration = 1000.0

        request_throughputs, output_token_throughputs = (
            parser._calculate_throughput_metrics(
                requests, output_sequence_lengths, benchmark_duration
            )
        )

        assert request_throughputs == [600.0]
        assert output_token_throughputs == [60.0]

    def test_calculate_throughput_metrics_missing_data(self):
        mock_throughput_metrics_dict = {}

        mock_tokenizer = MagicMock()

        with patch.object(
            MergedProfileParser,
            "__init__",
            lambda x, filename, tokenizer, throughput_metrics_dict, goodput_constraints=None: None,
        ):
            parser = MergedProfileParser(
                filename="mock_merged_profile.json",
                tokenizer=mock_tokenizer,
                throughput_metrics_dict=mock_throughput_metrics_dict,
            )

        parser._throughput_metrics_dict = mock_throughput_metrics_dict
        parser._session_metrics = {}

        requests = {}
        output_sequence_lengths = [10, 20, 30]
        benchmark_duration = 1000.0

        request_throughputs, output_token_throughputs = (
            parser._calculate_throughput_metrics(
                requests, output_sequence_lengths, benchmark_duration
            )
        )

        assert request_throughputs == [0]
        assert output_token_throughputs == [0]
