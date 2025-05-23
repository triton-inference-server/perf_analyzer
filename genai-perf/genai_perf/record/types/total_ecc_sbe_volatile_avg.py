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

from genai_perf.record.types.total_ecc_sbe_volatile_base import ECCSBEVolatileTotalBase


@total_ordering
class ECCSBEVolatileTotalAvg(ECCSBEVolatileTotalBase):
    """
    A record for avg total ECC single bit volatile errors
    """

    tag = ECCSBEVolatileTotalBase.base_tag + "_avg"

    def __init__(self, value, device_uuid=None, timestamp=0):
        super().__init__(value, device_uuid, timestamp)

    @classmethod
    def header(cls, aggregation_tag=False) -> str:
        return "Avg. Total ECC SBE Volatile Errors"
