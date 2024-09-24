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
from typing import Dict, Optional, Tuple, TypeAlias, Union

ConstraintName: TypeAlias = str
ConstraintValue: TypeAlias = Union[float, int]

Constraint: TypeAlias = Tuple[ConstraintName, ConstraintValue]
Constraints: TypeAlias = Dict[ConstraintName, ConstraintValue]


@dataclass
class ModelConstraints:
    """
    A dataclass that specifies the constraints used for a single model
    """

    constraints: Optional[Constraints] = None

    def has_constraint(self, constraint_name: ConstraintName) -> bool:
        """
        Checks if a given constraint is present
        """
        if self.constraints and constraint_name in self.constraints:
            return True
        else:
            return False
