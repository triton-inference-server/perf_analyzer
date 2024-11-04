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

from genai_perf.record.record import IncreasingRecord
from genai_perf.types import RecordValue


@total_ordering
class OutputTokenThroughput(IncreasingRecord):
    """
    A record for Output token throughput
    """

    tag = "output_token_throughput"

    def __init__(self, value: RecordValue, timestamp: int = 0) -> None:
        super().__init__(value, timestamp)

    @staticmethod
    def value_function():
        """
        Returns the total value from a list

        Returns
        -------
        Total value of the list
        """
        return sum

    @staticmethod
    def header(aggregation_tag=False) -> str:
        return "Output Token Throughput (infer/sec)"

    def __eq__(self, other: "OutputTokenThroughput") -> bool:  # type: ignore
        return self.value() == other.value()

    def __lt__(self, other: "OutputTokenThroughput") -> bool:
        return self.value() < other.value()

    def __add__(self, other: "OutputTokenThroughput") -> "OutputTokenThroughput":
        """
        Allows adding two records together
        to produce a brand new record.
        """

        return self.__class__(value=(self.value() + other.value()))

    def __sub__(self, other: "OutputTokenThroughput") -> "OutputTokenThroughput":
        """
        Allows subtracting two records together
        to produce a brand new record.
        """

        return self.__class__(value=(self.value() - other.value()))
