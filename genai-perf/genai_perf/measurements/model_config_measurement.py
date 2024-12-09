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


from copy import deepcopy
from dataclasses import dataclass
from functools import total_ordering
from statistics import mean
from typing import Any, Dict, Optional, TypeAlias

from genai_perf.record.record import Record
from genai_perf.record.types.request_throughput_avg import RequestThroughputAvg
from genai_perf.types import (
    CheckpointObject,
    MetricObjectives,
    ModelName,
    PerfMetricName,
    PerfRecords,
)

###########################################################################
# Typing
###########################################################################
ModelConfigMeasurements: TypeAlias = Dict[ModelName, "ModelConfigMeasurement"]


###########################################################################
# Defaults
###########################################################################
@dataclass(frozen=True)
class ModelConfigMeasurementDefaults:
    METRIC_OBJECTIVE = {RequestThroughputAvg.tag: 1.0}

    SELF_IS_BETTER = 1
    OTHER_IS_BETTER = -1
    EQUALIVILENT = 0

    COMPARISON_SCORE_THRESHOLD = 0


@total_ordering
class ModelConfigMeasurement:
    """
    Encapsulates the set of performance metrics (measurements) obtained when profiling a model
    """

    def __init__(self, perf_metrics: PerfRecords):
        """
        perf_metrics:
            Metrics (stored in the Record class) that are associated with how the model
            performed. Examples include throughput and latency.
        """

        self._perf_metrics = perf_metrics

        # Set a default metric objective
        self._metric_objectives = ModelConfigMeasurementDefaults.METRIC_OBJECTIVE

    ###########################################################################
    # Accessor Methods
    ###########################################################################
    def get_perf_metrics(self) -> PerfRecords:
        return self._perf_metrics

    def get_perf_metric(self, name: PerfMetricName) -> Optional[Record]:
        return self._perf_metrics[name] if name in self._perf_metrics else None

    def get_perf_metric_value(self, name: PerfMetricName, return_value: int = 0) -> Any:
        metric = self.get_perf_metric(name)
        return (
            metric.value() / (10**metric.reduction_factor) if metric else return_value
        )

    def get_weighted_score(self, other: "ModelConfigMeasurement") -> float:
        """
        Returns the weighted score between this MCM and the
        provided MCM
        """
        return self._calculate_weighted_score(other)

    def set_metric_objectives(self, metric_objectives: MetricObjectives) -> None:
        """
        Sets metric weighting for this measurement based
        on the objectives
        """

        # Each individual weighting is based on it's percentage of the total
        # weighting. Example: {A: 1, B: 3} would be stored as {A: 0.25, B: 0.75}
        self._metric_objectives = {
            objective: (value / sum(metric_objectives.values()))
            for objective, value in metric_objectives.items()
        }

    ###########################################################################
    # Checkpoint Methods
    ###########################################################################
    def create_checkpoint_object(self) -> CheckpointObject:
        """
        Converts the class data into a dictionary that can be written to
        the checkpoint file
        """
        mcm_dict = deepcopy(self.__dict__)

        # Values based solely on user/config settings (that can vary from run to run)
        # are not stored in the checkpoint
        del mcm_dict["_metric_objectives"]

        return mcm_dict

    @classmethod
    def create_class_from_checkpoint(
        cls, mcm_dict: CheckpointObject
    ) -> "ModelConfigMeasurement":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of a MCM
        """
        perf_metrics = cls._read_perf_metrics_from_checkpoint(mcm_dict["_perf_metrics"])

        mcm = ModelConfigMeasurement(perf_metrics)

        return mcm

    @classmethod
    def _read_perf_metrics_from_checkpoint(
        cls, perf_metrics_dict: Dict[str, Any]
    ) -> PerfRecords:
        perf_metrics: PerfRecords = {}

        for [tag, record_dict] in perf_metrics_dict.values():
            record = Record.get(tag)
            record = record.create_class_from_checkpoint(record_dict)  # type: ignore
            perf_metrics[tag] = record  # type: ignore

        return perf_metrics

    ###########################################################################
    # Comparison Methods
    ###########################################################################
    def is_better_than(self, other: "ModelConfigMeasurement") -> bool:
        return (
            self._compare_measurements(other)
            == ModelConfigMeasurementDefaults.SELF_IS_BETTER
        )

    def __lt__(self, other: "ModelConfigMeasurement") -> bool:
        return (
            self._compare_measurements(other)
            == ModelConfigMeasurementDefaults.OTHER_IS_BETTER
        )

    def __gt__(self, other: "ModelConfigMeasurement") -> bool:
        return (
            self._compare_measurements(other)
            == ModelConfigMeasurementDefaults.SELF_IS_BETTER
        )

    def __eq__(self, other: "ModelConfigMeasurement") -> bool:  # type: ignore
        return (
            self._compare_measurements(other)
            == ModelConfigMeasurementDefaults.EQUALIVILENT
        )

    def _compare_measurements(self, other: "ModelConfigMeasurement") -> int:
        """
        Compares two MCMs
        based on the weighted metric objectives
        """
        weighted_score = self._calculate_weighted_score(other)

        if weighted_score > ModelConfigMeasurementDefaults.COMPARISON_SCORE_THRESHOLD:
            return ModelConfigMeasurementDefaults.SELF_IS_BETTER
        elif weighted_score < ModelConfigMeasurementDefaults.COMPARISON_SCORE_THRESHOLD:
            return ModelConfigMeasurementDefaults.OTHER_IS_BETTER
        else:
            return ModelConfigMeasurementDefaults.EQUALIVILENT

    ###########################################################################
    # Calculation Methods
    ###########################################################################
    def _calculate_weighted_score(self, other: "ModelConfigMeasurement") -> float:
        """
        Calculates the weighted score between two
        ModelConfig measurements based on the weighted
        metric objectives

        A positive value indicates this MCM is better than the other
        """

        weighted_score = 0.0
        for objective, weight in self._metric_objectives.items():
            self_metric = self.get_perf_metric(objective)
            other_metric = other.get_perf_metric(objective)

            # This handles the case where metric(s) do not exist
            if self_metric and other_metric is None:
                return ModelConfigMeasurementDefaults.SELF_IS_BETTER
            elif other_metric and self_metric is None:
                return ModelConfigMeasurementDefaults.OTHER_IS_BETTER
            elif self_metric is None and other_metric is None:
                return ModelConfigMeasurementDefaults.EQUALIVILENT

            metric_diff = self_metric - other_metric  # type: ignore
            average = mean([self_metric.value(), other_metric.value()])  # type: ignore
            weighted_score += weight * (metric_diff.value() / average)

        return weighted_score

    def calculate_weighted_percentage_gain(
        self, other: "ModelConfigMeasurement"
    ) -> float:
        """
        Calculates the weighted percentage between two
        ModelConfig measurements based on the weighted
        metric objectives

        The weighted percentage gain. A positive value indicates
        this MCM is better than the other
        """

        weighted_pct = 0.0
        for objective, weight in self._metric_objectives.items():
            self_metric = self.get_perf_metric(objective)
            other_metric = other.get_perf_metric(objective)

            # This handles the case where metric(s) do not exist
            if self_metric and other_metric is None:
                return 100 * ModelConfigMeasurementDefaults.SELF_IS_BETTER
            elif other_metric and self_metric is None:
                return 100 * ModelConfigMeasurementDefaults.OTHER_IS_BETTER
            elif self_metric is None and other_metric is None:
                return 100 * ModelConfigMeasurementDefaults.EQUALIVILENT

            metric_pct = self_metric.calculate_percentage_gain(other_metric)  # type: ignore

            weighted_pct += metric_pct * weight

        return weighted_pct
