# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from genai_perf.record.gpu_record import DecreasingGPURecord
from genai_perf.record.record import ReductionFactor


@total_ordering
class PCieReplayCounterBase(DecreasingGPURecord):
    """
    A base class for the PCie Replay Counter metric
    """

    base_tag = "pcie_replay_counter"
    reduction_factor = ReductionFactor.NONE

    def __init__(self, value, device_uuid=None, timestamp=0):
        super().__init__(value, device_uuid, timestamp)

    @staticmethod
    def header(aggregation_tag=False):
        return ("Max " if aggregation_tag else "") + "PCIe Replay Counter"

    def __eq__(self, other: "PCieReplayCounterBase") -> bool:  # type: ignore
        return self.value() == other.value()

    def __lt__(self, other: "PCieReplayCounterBase") -> bool:
        return other.value() < self.value()

    def __add__(self, other: "PCieReplayCounterBase") -> "PCieReplayCounterBase":
        return self.__class__(device_uuid=None, value=(self.value() + other.value()))

    def __sub__(self, other: "PCieReplayCounterBase") -> "PCieReplayCounterBase":
        return self.__class__(device_uuid=None, value=(other.value() - self.value()))
