# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import total_ordering

from genai_perf.record.types.inter_token_latency_base import InterTokenLatencyBase


@total_ordering
class InterTokenLatencyP75(InterTokenLatencyBase):
    """
    A record for p75 Inter token latency metric
    """

    tag = InterTokenLatencyBase.base_tag + "_p75"

    def __init__(self, value, timestamp=0):
        super().__init__(value, timestamp)

    @classmethod
    def header(cls, aggregation_tag=False) -> str:
        return "p75 Inter Token Latency (ms)"
