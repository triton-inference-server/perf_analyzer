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
from typing import Any, List, Optional, TypeAlias, Union

from genai_perf.exceptions import GenAIPerfException

ParameterList: TypeAlias = List[Union[int, str]]


class SearchUsage(Enum):
    MODEL = auto()
    RUNTIME_PA = auto()
    RUNTIME_GAP = auto()
    BUILD = auto()


class SearchCategory(Enum):
    INTEGER = auto()
    EXPONENTIAL = auto()
    STR_LIST = auto()
    INT_LIST = auto()


@dataclass
class SearchParameter:
    """
    A dataclass that holds information about a configuration's search parameter
    """

    usage: SearchUsage
    category: SearchCategory

    # This is only applicable to the LIST categories
    enumerated_list: Optional[List[Any]] = None

    # These are only applicable to INTEGER and EXPONENTIAL categories
    min_range: Optional[int] = None
    max_range: Optional[int] = None

    def get_list(self) -> List[Any]:
        """
        Returns the list of all possible parameter values
        """
        if (
            self.category == SearchCategory.STR_LIST
            or self.category == SearchCategory.INT_LIST
        ):
            return self.enumerated_list  # type: ignore
        elif (
            self.category == SearchCategory.INTEGER
            or self.category == SearchCategory.EXPONENTIAL
        ):
            return [value for value in range(self.min_range, self.max_range + 1)]  # type: ignore

        raise GenAIPerfException(f"{self.category} is not a known SearchCategory")
