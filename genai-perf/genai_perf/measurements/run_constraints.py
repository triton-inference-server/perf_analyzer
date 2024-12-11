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
from typing import Dict, Optional

from genai_perf.measurements.model_constraints import ModelConstraints
from genai_perf.types import ConstraintName, ModelName


@dataclass
class RunConstraints:
    """
    A dataclass that specifies the constraints used for a single run
    """

    constraints: Optional[Dict[ModelName, ModelConstraints]] = None

    def has_constraint(
        self, model_name: ModelName, constraint_name: ConstraintName
    ) -> bool:
        """
        Checks if a given constraint is present for a model
        """
        if self.constraints and model_name in self.constraints:
            return self.constraints[model_name].has_constraint(constraint_name)
        else:
            return False
