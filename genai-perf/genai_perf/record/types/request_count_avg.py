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

from genai_perf.record.record import IncreasingRecord, ReductionFactor


@total_ordering
class RequestCountAvg(IncreasingRecord):
    """
    A record avg request count metric
    """

    tag = "request_count_avg"
    reduction_factor = ReductionFactor.NONE

    def __init__(self, value, timestamp=0):
        super().__init__(value, timestamp)

    @staticmethod
    def value_function():
        return sum

    @staticmethod
    def header(aggregation_tag=False) -> str:
        return "Request Count (requests/sec)"

    def __eq__(self, other: "RequestCountAvg") -> bool:  # type: ignore
        return self.value() == other.value()

    def __lt__(self, other: "RequestCountAvg") -> bool:
        return self.value() < other.value()

    def __add__(self, other: "RequestCountAvg") -> "RequestCountAvg":
        return self.__class__(value=(self.value() + other.value()))

    def __sub__(self, other: "RequestCountAvg") -> "RequestCountAvg":
        return self.__class__(value=(self.value() - other.value()))
