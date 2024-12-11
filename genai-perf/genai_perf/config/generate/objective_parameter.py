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

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, TypeAlias

from genai_perf.config.generate.search_parameter import SearchUsage
from genai_perf.exceptions import GenAIPerfException


class ObjectiveCategory(Enum):
    INTEGER = auto()
    EXPONENTIAL = auto()
    STR = auto()


ObjectiveParameters: TypeAlias = Dict[str, "ObjectiveParameter"]


@dataclass
class ObjectiveParameter:
    """
    A dataclass that holds information about a configuration's objective parameter
    """

    usage: SearchUsage
    category: ObjectiveCategory
    value: Any

    def get_value_based_on_category(self) -> Any:
        if (
            self.category == ObjectiveCategory.INTEGER
            or self.category == ObjectiveCategory.STR
        ):
            return self.value
        elif self.category == ObjectiveCategory.EXPONENTIAL:
            return 2**self.value

        raise GenAIPerfException(f"{self.category} is not a known ObjectiveCategory")
