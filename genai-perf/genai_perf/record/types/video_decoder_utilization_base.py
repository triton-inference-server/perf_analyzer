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

from genai_perf.record.gpu_record import IncreasingGPURecord
from genai_perf.record.record import ReductionFactor


@total_ordering
class VideoDecoderUtilizationBase(IncreasingGPURecord):
    """
    A base class for the Video Decoder utilization percentage metric
    """

    base_tag = "video_decoder_utilization"
    reduction_factor = ReductionFactor.NONE

    def __init__(self, value, device_uuid=None, timestamp=0):
        super().__init__(value, device_uuid, timestamp)

    @staticmethod
    def aggregation_function():
        def average(seq):
            return sum(seq[1:], start=seq[0]) / len(seq)

        return average

    @staticmethod
    def header(aggregation_tag=False):
        return ("Average " if aggregation_tag else "") + "Video Decoder Utilization (%)"

    def __eq__(self, other: "VideoDecoderUtilizationBase") -> bool:  # type: ignore
        return self.value() == other.value()

    def __lt__(self, other: "VideoDecoderUtilizationBase") -> bool:
        return self.value() < other.value()

    def __add__(
        self, other: "VideoDecoderUtilizationBase"
    ) -> "VideoDecoderUtilizationBase":
        return self.__class__(device_uuid=None, value=(self.value() + other.value()))

    def __sub__(
        self, other: "VideoDecoderUtilizationBase"
    ) -> "VideoDecoderUtilizationBase":
        return self.__class__(device_uuid=None, value=(self.value() - other.value()))
