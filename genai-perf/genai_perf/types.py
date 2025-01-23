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

from typing import Any, Dict, Tuple, TypeAlias, Union

# NOTE: Any classes used must be declared as "<class_name>" and use `#type: ignore`
# this is to prevent circular import issues, while still allowing us to keep any
# TypeAlias used across multiple files in a single location

###########################################################################
# Model
###########################################################################
ModelName: TypeAlias = str
ModelWeights: TypeAlias = Dict[ModelName, Union[int, float]]
ModelSearchParameters: TypeAlias = Dict[ModelName, "SearchParameters"]  # type: ignore
ModelObjectiveParameters: TypeAlias = Dict[ModelName, "ObjectiveParameters"]  # type: ignore

###########################################################################
# GPU
###########################################################################
GpuId: TypeAlias = str

###########################################################################
# Records
###########################################################################
TelemetryRecords: TypeAlias = Dict[str, "GPURecord"]  # type: ignore
GpuRecords: TypeAlias = Dict[GpuId, TelemetryRecords]
PerfRecords: TypeAlias = Dict[str, "Record"]  # type: ignore
PerfMetricName: TypeAlias = str
RecordValue = Union[float, int]

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
GpuMetricObjectives: TypeAlias = Dict[ModelName, MetricObjectives]
PerfMetricObjectives: TypeAlias = Dict[ModelName, MetricObjectives]

###########################################################################
# Parameters
###########################################################################
Parameters: TypeAlias = Dict[str, Any]

###########################################################################
# Run Config
###########################################################################
RunConfigName: TypeAlias = str

###########################################################################
# Checkpoint
###########################################################################
CheckpointObject: TypeAlias = Dict[str, Any]
