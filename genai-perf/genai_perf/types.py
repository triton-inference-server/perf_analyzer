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

from typing import Dict, Tuple, TypeAlias, Union

from genai_perf.record.gpu_record import GPURecord
from genai_perf.record.record import Record

###########################################################################
# Model
###########################################################################
ModelName: TypeAlias = str
ModelWeights: TypeAlias = Dict[ModelName, Union[int, float]]

###########################################################################
# GPU
###########################################################################
GpuId: TypeAlias = str

###########################################################################
# Records
###########################################################################
TelemetryRecords: TypeAlias = Dict[str, GPURecord]
GpuRecords: TypeAlias = Dict[GpuId, TelemetryRecords]
PerfRecords: TypeAlias = Dict[str, Record]

###########################################################################
# Constraints
###########################################################################
ConstraintName: TypeAlias = str
ConstraintValue: TypeAlias = Union[float, int]

Constraint: TypeAlias = Tuple[ConstraintName, ConstraintValue]
Constraints: TypeAlias = Dict[ConstraintName, ConstraintValue]

###########################################################################
# Objectives
###########################################################################
MetricObjectives: TypeAlias = Dict[str, float]
