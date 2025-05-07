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

ABBREVIATIONS = ["gpu"]
DEFAULT_LRU_CACHE_SIZE = 100_000
DEFAULT_DCGM_METRICS_URL = "http://localhost:9400/metrics"
EMPTY_RESPONSE_TOKEN = 0

# These map to the various fields that can be set for PA and model configs
# See github.com/triton-inference-server/model_analyzer/blob/main/docs/config.md
exponential_range_parameters = [
    "model_batch_size",
    "runtime_batch_size",
    "concurrency",
    "request_rate",
    "input_sequence_length",
]

linear_range_parameters = ["instance_count", "num_dataset_entries"]


runtime_pa_parameters = ["runtime_batch_size", "concurrency", "request_rate"]

runtime_gap_parameters = ["num_dataset_entries", "input_sequence_length"]

all_parameters = runtime_pa_parameters + runtime_gap_parameters
