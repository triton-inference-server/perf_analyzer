# Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Dict, List, Optional

# TODO: OPTIMIZE
from genai_perf.measurements.model_constraints import ModelConstraints


@dataclass
class ModelSpec:
    """
    A dataclass that specifies the various ways
    a model is configured among PA/GAP/Triton
    """

    # Model information/parameters
    model_name: str
    cpu_only: bool = False
    objectives: Optional[List] = None
    constraints: Optional[ModelConstraints] = None
    model_config_parameters: Optional[Dict] = None

    # PA/GAP flags/parameters
    perf_analyzer_parameters: Optional[Dict] = None
    perf_analyzer_flags: Optional[Dict] = None
    genai_perf_flags: Optional[Dict] = None

    # Triton flags/args
    triton_server_flags: Optional[Dict] = None
    triton_server_args: Optional[Dict] = None
    triton_docker_args: Optional[Dict] = None
