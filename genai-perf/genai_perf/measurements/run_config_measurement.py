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
from typing import Any, Dict, Optional, TypeAlias, Union

import genai_perf.logging as logging
from genai_perf.measurements.model_config_measurement import (
    ModelConfigMeasurement,
    ModelConfigMeasurements,
)
from genai_perf.measurements.run_constraints import RunConstraints
from genai_perf.record.record import Record
from genai_perf.types import (
    CheckpointObject,
    ConstraintName,
    ConstraintValue,
    GpuId,
    GpuMetricObjectives,
    GpuRecords,
    ModelName,
    ModelWeights,
    PerfMetricName,
    PerfMetricObjectives,
    PerfRecords,
)

logger = logging.getLogger(__name__)


WeightedMcmScores: TypeAlias = Dict[ModelName, float]
WeightedRcmScore: TypeAlias = float


@dataclass(frozen=True)
class RunConfigMeasurementDefaults:
    MODEL_WEIGHTING = 1

    METRIC_OBJECTIVE = None

    SELF_IS_BETTER = 1
    OTHER_IS_BETTER = -1
    EQUALIVILENT = 0

    COMPARISON_SCORE_THRESHOLD = 0


@total_ordering
class RunConfigMeasurement:
    """
    Encapsulates the set of metrics obtained from all models
    in a single RunConfig, as well as the GPU(s) metrics
    """

    def __init__(
        self,
        gpu_metrics: Optional[GpuRecords] = None,
        run_constraints: Optional[RunConstraints] = None,
    ):
        """
        gpu_metrics:
            Metrics that are associated with GPU UUID(s), these are
            typically telemetry metrics

        run_constraints:
            A set of constraints (set by the user) used to determine if
            this is a valid measurement
        """
        self._gpu_metrics = gpu_metrics if gpu_metrics else {}
        self._gpu_metric_objectives: Optional[GpuMetricObjectives] = (
            RunConfigMeasurementDefaults.METRIC_OBJECTIVE
        )

        # Since this is not stored in the checkpoint it is optional, and
        # can be later set by an accessor method
        self._constraints = run_constraints

        # These are per model values
        self._model_config_measurements: ModelConfigMeasurements = {}
        self._model_weights: ModelWeights = {}

    ###########################################################################
    # Get Accessor Methods
    ###########################################################################
    def get_all_gpu_metrics(self) -> GpuRecords:
        """
        Returns GPU metrics for all GPUs
        """
        return self._gpu_metrics

    def get_gpu_metric(self, name: str) -> Optional[GpuRecords]:
        """
        Returns a specific GPU metric for all GPUs
        """
        for gpu_id, gpu_records in self._gpu_metrics.items():
            metric_found = False
            for gpu_record in gpu_records:
                if name == gpu_record:
                    metric_found = True

            if not metric_found:
                return None

        gpu_metrics = {
            gpu_id: {name: gpu_metrics[name]}
            for gpu_id, gpu_metrics in self._gpu_metrics.items()
        }

        return gpu_metrics

    def get_gpu_metric_value(
        self,
        gpu_id: str,
        name: str,
        return_value: int = 0,
    ) -> Any:
        gpu_metrics = self.get_gpu_metric(name)

        gpu_metric = None
        if gpu_metrics:
            if gpu_id in gpu_metrics:
                if name in gpu_metrics[gpu_id]:
                    gpu_metric = gpu_metrics[gpu_id][name]

        return (
            gpu_metric.value() / (10**gpu_metric.reduction_factor)
            if gpu_metric
            else return_value
        )

    def get_model_config_measurements(self) -> ModelConfigMeasurements:
        return self._model_config_measurements

    def get_model_config_measurement(
        self, model_name: ModelName
    ) -> Optional[ModelConfigMeasurement]:
        return (
            self._model_config_measurements[model_name]
            if model_name in self._model_config_measurements
            else None
        )

    def get_all_perf_metrics(self) -> Dict[ModelName, PerfRecords]:
        perf_metrics = {
            model_name: mcm.get_perf_metrics()
            for model_name, mcm in self._model_config_measurements.items()
        }

        return perf_metrics

    def get_model_perf_metrics(self, model_name: ModelName) -> Optional[PerfRecords]:
        if model_name in self._model_config_measurements:
            return self._model_config_measurements[model_name].get_perf_metrics()
        else:
            return None

    def get_model_perf_metric(
        self, model_name: ModelName, perf_metric_name: PerfMetricName
    ) -> Optional[Record]:
        if model_name in self._model_config_measurements:
            return self._model_config_measurements[model_name].get_perf_metric(
                perf_metric_name
            )
        else:
            return None

    def get_model_perf_metric_value(
        self,
        model_name: ModelName,
        perf_metric_name: PerfMetricName,
        return_value: int = 0,
    ) -> Any:
        if model_name in self._model_config_measurements:
            return self._model_config_measurements[model_name].get_perf_metric_value(
                perf_metric_name, return_value
            )
        else:
            return return_value

    def get_weighted_perf_metric_values(
        self,
        perf_metric_name: PerfMetricName,
        return_value: int = 0,
    ) -> Dict[ModelName, Any]:
        """
        Returns per model perf metric value based on model weighting
        """
        assert self._model_weights.keys() == self._model_config_measurements.keys()

        per_model_weighted_perf_metric_values = {
            model_name: mcm.get_perf_metric_value(perf_metric_name)
            * self._model_weights[model_name]
            for model_name, mcm in self._model_config_measurements.items()
        }

        return per_model_weighted_perf_metric_values

    ###########################################################################
    # Set Accessor Methods
    ###########################################################################
    def set_model_weighting(self, model_weights: ModelWeights) -> None:
        """
        Sets the model weightings used when calculating
        weighted metrics

        Weights are the relative importance of the model
        with respect to one another
        """
        assert model_weights.keys() == self._model_config_measurements.keys()

        self._model_weights = {
            model_name: model_weight / sum(model_weights.values())
            for model_name, model_weight in model_weights.items()
        }

    def set_gpu_metric_objectives(
        self, gpu_metric_objectives: GpuMetricObjectives
    ) -> None:
        """
        Sets the metric weighting for all GPU metric based measurements
        """
        for model_name in gpu_metric_objectives.keys():
            assert model_name in self._model_weights.keys()

        self._gpu_metric_objectives = {}
        for model_name, model_gpu_metric_objectives in gpu_metric_objectives.items():
            self._gpu_metric_objectives[model_name] = {
                objective: (value / sum(model_gpu_metric_objectives.values()))
                for objective, value in model_gpu_metric_objectives.items()
            }

    def set_perf_metric_objectives(
        self, perf_metric_objectives: PerfMetricObjectives
    ) -> None:
        """
        Sets the metric weighting for all perf metric based measurements
        """
        for key in perf_metric_objectives.keys():
            assert key in self._model_config_measurements.keys()

        for model_name, metric_objectives in perf_metric_objectives.items():
            self._model_config_measurements[model_name].set_metric_objectives(
                metric_objectives
            )

    def set_constraints(self, constraints: RunConstraints) -> None:
        self._constraints = constraints

    def add_perf_metrics(
        self,
        model_name: ModelName,
        perf_metrics: PerfRecords,
    ) -> None:
        """
        Adds the perf metrics obtained from profiling a single model
        """
        self._model_config_measurements[model_name] = ModelConfigMeasurement(
            perf_metrics
        )
        self._model_weights[model_name] = RunConfigMeasurementDefaults.MODEL_WEIGHTING

    ###########################################################################
    # Checkpoint Methods
    ###########################################################################
    def create_checkpoint_object(self) -> CheckpointObject:
        """
        Converts the class data into a dictionary that can be written to
        the checkpoint file
        """
        rcm_dict = deepcopy(self.__dict__)

        # Values based solely on user/config settings (that can vary from run to run)
        # are not stored in the checkpoint
        del rcm_dict["_gpu_metric_objectives"]
        del rcm_dict["_model_weights"]
        del rcm_dict["_constraints"]

        return rcm_dict

    @classmethod
    def create_class_from_checkpoint(
        cls, rcm_dict: CheckpointObject
    ) -> "RunConfigMeasurement":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of a RCM
        """
        gpu_metrics = cls._read_gpu_metrics_from_checkpoint(rcm_dict["_gpu_metrics"])
        rcm = RunConfigMeasurement(gpu_metrics=gpu_metrics)

        rcm._model_config_measurements = (
            cls._read_model_config_measurements_from_checkpoint(
                rcm_dict["_model_config_measurements"],
            )
        )

        # Need to (re)set a default weighting for each model
        for model_name in rcm._model_config_measurements.keys():
            rcm._model_weights[model_name] = (
                RunConfigMeasurementDefaults.MODEL_WEIGHTING
            )

        return rcm

    @classmethod
    def _read_gpu_metrics_from_checkpoint(
        cls, gpu_metrics_dict: Dict[GpuId, Any]
    ) -> GpuRecords:
        gpu_metrics = {}
        for gpu_uuid, gpu_metrics_dict in gpu_metrics_dict.items():
            gpu_metric_dict: Any = {}
            for tag, record_dict in gpu_metrics_dict.values():
                record = Record.get(tag)
                record = record.create_class_from_checkpoint(record_dict)  # type: ignore
                gpu_metric_dict[tag] = record

            gpu_metrics[gpu_uuid] = gpu_metric_dict

        return gpu_metrics

    @classmethod
    def _read_model_config_measurements_from_checkpoint(
        cls, mcm_dicts: Dict[str, Any]
    ) -> ModelConfigMeasurements:
        model_config_measurements: ModelConfigMeasurements = {}
        for model_name, mcm_dict in mcm_dicts.items():
            model_config_measurements[model_name] = (
                ModelConfigMeasurement.create_class_from_checkpoint(mcm_dict)
            )

        return model_config_measurements

    ###########################################################################
    # Comparison Methods
    ###########################################################################
    def get_score(self, other: "RunConfigMeasurement") -> float:
        """
        Compares the measurements and returns a score.
        The larger the positive value the better self is,
        the larger the negative value the better other is
        """
        score = self._compare_measurements(other, return_score=True)

        return score

    def is_better_than(self, other: "RunConfigMeasurement") -> bool:
        return (
            self._compare_measurements(other)
            == RunConfigMeasurementDefaults.SELF_IS_BETTER
        )

    def __lt__(self, other: "RunConfigMeasurement") -> bool:
        return (
            self._compare_measurements(other)
            == RunConfigMeasurementDefaults.OTHER_IS_BETTER
        )

    def __gt__(self, other: "RunConfigMeasurement") -> bool:
        return (
            self._compare_measurements(other)
            == RunConfigMeasurementDefaults.SELF_IS_BETTER
        )

    def __eq__(self, other: "RunConfigMeasurement") -> bool:  # type: ignore
        return (
            self._compare_measurements(other)
            == RunConfigMeasurementDefaults.EQUALIVILENT
        )

    def _compare_measurements(
        self, other: "RunConfigMeasurement", return_score: bool = False
    ) -> Union[int, float]:
        """
        Compares two RunConfigMeasurements based on each
        ModelConfigs weighted metric objectives and the
        ModelConfigs weighted value within the RunConfigMeasurement
        """
        # Step 1: for each model determine the weighted score
        weighted_mcm_scores = self._calculate_weighted_mcm_scores(other)
        weighted_rcm_score = self._calculate_weighted_rcm_score(other)

        # Step 2: combine these using the model weighting
        weighted_combined_score = self._calculate_weighted_rcm_and_mcm_score(
            weighted_rcm_score, weighted_mcm_scores
        )

        # Step 2.5: if only the score is wanted stop here
        if return_score:
            return weighted_combined_score

        # Step 3: Determine which RCM is better
        if (
            weighted_combined_score
            > RunConfigMeasurementDefaults.COMPARISON_SCORE_THRESHOLD
        ):
            return RunConfigMeasurementDefaults.SELF_IS_BETTER
        elif (
            weighted_combined_score
            < RunConfigMeasurementDefaults.COMPARISON_SCORE_THRESHOLD
        ):
            return RunConfigMeasurementDefaults.OTHER_IS_BETTER
        else:
            return RunConfigMeasurementDefaults.EQUALIVILENT

    ###########################################################################
    # Calculation Methods
    ###########################################################################
    def _calculate_weighted_mcm_scores(
        self, other: "RunConfigMeasurement"
    ) -> WeightedMcmScores:
        """
        Returns a weighted score for each ModelConfigMeasurement
        """
        assert (
            self._model_config_measurements.keys()
            == other._model_config_measurements.keys()
        )

        weighted_mcm_scores = {
            model_name: self._model_config_measurements[model_name].get_weighted_score(
                other._model_config_measurements[model_name]
            )
            for model_name in self._model_config_measurements.keys()
        }

        return weighted_mcm_scores

    def _calculate_weighted_rcm_score(
        self, other: "RunConfigMeasurement"
    ) -> WeightedRcmScore:
        """
        Return a weighted score for GPU metrics, averaged across all GPUs
        """
        weighted_rcm_scores = []
        for gpu_id in self._gpu_metrics.keys():
            weighted_rcm_scores.append(
                self._calculate_weighted_model_rcm_score(other, gpu_id)
            )

        if not weighted_rcm_scores:
            return RunConfigMeasurementDefaults.EQUALIVILENT
        else:
            return mean(weighted_rcm_scores)

    def _calculate_weighted_model_rcm_score(
        self, other: "RunConfigMeasurement", gpu_id: GpuId
    ) -> float:
        """
        Return a weighted score for this models GPU metrics
        """
        weighted_score = 0.0
        if not self._gpu_metric_objectives:
            return weighted_score

        for per_model_gpu_metric_objectives in self._gpu_metric_objectives.values():
            for objective, weight in per_model_gpu_metric_objectives.items():
                self_metric = self.get_gpu_metric(objective)[gpu_id][objective]  # type: ignore
                other_metric = other.get_gpu_metric(objective)[gpu_id][objective]  # type: ignore

                # This handles the case where metric(s) do not exist
                if self_metric and other_metric is None:
                    return RunConfigMeasurementDefaults.SELF_IS_BETTER
                elif other_metric and self_metric is None:
                    return RunConfigMeasurementDefaults.OTHER_IS_BETTER
                elif self_metric is None and other_metric is None:
                    return RunConfigMeasurementDefaults.EQUALIVILENT

                metric_diff = self_metric - other_metric  # type: ignore
                average = mean([self_metric.value(), other_metric.value()])  # type: ignore
                weighted_score += weight * (metric_diff.value() / average)

        return weighted_score

    def _calculate_weighted_rcm_and_mcm_score(
        self,
        weighted_rcm_score: WeightedRcmScore,
        weighted_mcm_scores: WeightedMcmScores,
    ) -> float:
        """
        A positive value indicates this RCM is better
        than the other RCM
        """
        assert self._model_weights.keys() == weighted_mcm_scores.keys()

        weighted_combined_mcm_scores = [
            (weighted_mcm_scores[model_name]) * self._model_weights[model_name]
            for model_name in self._model_weights.keys()
        ]

        weighted_combined_score = sum(weighted_combined_mcm_scores) + weighted_rcm_score

        return weighted_combined_score

    ###########################################################################
    # Percentage Calculation Methods
    ###########################################################################
    def calculate_weighted_percentage_gain(
        self, other: "RunConfigMeasurement"
    ) -> float:
        """
        Calculates the weighted percentage gain between
        two RunConfigMeasurements based on each model's
        weighted metric objectives and the model's
        weighted value within the RunConfigMeasurement

        Returns
        -------
           The weighted percentage gain. A positive value indicates
           this RCM is better than the other RCM
        """

        # For each model determine the weighted percentage gain
        weighted_rcm_pct_gain = self._calculate_weighted_rcm_percentage_gain(other)
        weighted_mcm_pct_gains = self._calculate_weighted_mcm_percentage_gains(other)

        # And, then combine these using the model's weighting
        weighted_mean_gain = self._calculate_weighted_rcm_and_mcm_percentage_gain(
            weighted_rcm_pct_gain, weighted_mcm_pct_gains
        )

        return weighted_mean_gain

    def _calculate_weighted_rcm_percentage_gain(
        self, other: "RunConfigMeasurement"
    ) -> WeightedRcmScore:
        weighted_rcm_gains = []
        for gpu_id in self._gpu_metrics.keys():
            weighted_rcm_gains.append(
                self._calculate_weighted_model_rcm_percentage_gain(other, gpu_id)
            )

        return mean(weighted_rcm_gains)

    def _calculate_weighted_model_rcm_percentage_gain(
        self, other: "RunConfigMeasurement", gpu_id: GpuId
    ) -> float:
        """
        Return a weighted score for this models GPU metrics
        """
        weighted_pct = 0.0
        if not self._gpu_metric_objectives:
            return weighted_pct

        for per_model_gpu_metric_objectives in self._gpu_metric_objectives.values():
            for objective, weight in per_model_gpu_metric_objectives.items():
                self_metric = self.get_gpu_metric(objective)[gpu_id][objective]  # type: ignore
                other_metric = other.get_gpu_metric(objective)[gpu_id][objective]  # type: ignore

                # This handles the case where metric(s) do not exist
                if self_metric and other_metric is None:
                    return 100 * RunConfigMeasurementDefaults.SELF_IS_BETTER
                elif other_metric and self_metric is None:
                    return 100 * RunConfigMeasurementDefaults.OTHER_IS_BETTER
                elif self_metric is None and other_metric is None:
                    return 100 * RunConfigMeasurementDefaults.EQUALIVILENT

                metric_pct = self_metric.calculate_percentage_gain(other_metric)  # type: ignore

                weighted_pct += metric_pct * weight

        return weighted_pct

    def _calculate_weighted_mcm_percentage_gains(
        self, other: "RunConfigMeasurement"
    ) -> WeightedMcmScores:
        """
        Returns a weighted percentage (as a score) for each ModelConfigMeasurement
        """
        assert (
            self._model_config_measurements.keys()
            == other._model_config_measurements.keys()
        )

        weighted_mcm_percentage_gains = {
            model_name: self._model_config_measurements[
                model_name
            ].calculate_weighted_percentage_gain(
                other._model_config_measurements[model_name]
            )
            for model_name in self._model_config_measurements.keys()
        }

        return weighted_mcm_percentage_gains

    def _calculate_weighted_rcm_and_mcm_percentage_gain(
        self,
        weighted_rcm_gain: WeightedRcmScore,
        weighted_mcm_gains: WeightedMcmScores,
    ) -> float:
        assert self._model_weights.keys() == weighted_mcm_gains.keys()

        weighted_combined_mcm_gains = [
            (weighted_mcm_gains[model_name]) * self._model_weights[model_name]
            for model_name in self._model_weights.keys()
        ]

        perf_metric_objectives_set = bool(
            list(self._model_config_measurements.values())[0].get_perf_metrics()
        )

        weighted_mean_gain = 0.0
        if self._gpu_metric_objectives and perf_metric_objectives_set:
            weighted_mean_gain = mean(
                [mean(weighted_combined_mcm_gains), weighted_rcm_gain]
            )
        elif self._gpu_metric_objectives:
            weighted_mean_gain = weighted_rcm_gain
        elif perf_metric_objectives_set:
            weighted_mean_gain = mean(weighted_combined_mcm_gains)

        return weighted_mean_gain

    def is_passing_constraints(self) -> bool:
        """
        Returns true if all model measurements pass
        their respective constraints
        """

        if not self._constraints:
            return True

        passing_constraints = True
        for model_name in self._constraints.constraints.keys():  # type: ignore
            if self._constraints.constraints[model_name]:  # type: ignore
                for model_constraints in self._constraints.constraints.values():  # type: ignore
                    if model_constraints:
                        for (
                            constraint_name,
                            value,
                        ) in model_constraints.constraints.items():  # type: ignore
                            if not self._passing_model_constraint(
                                model_name, constraint_name, value
                            ):
                                passing_constraints = False

        return passing_constraints

    def _passing_model_constraint(
        self,
        model_name: ModelName,
        constraint_name: ConstraintName,
        constraint_value: ConstraintValue,
    ) -> bool:
        passing_constraint = self._passing_gpu_metric_constraint(
            model_name, constraint_name, constraint_value
        )
        passing_constraint &= self._passing_perf_metric_constraint(
            model_name, constraint_name, constraint_value
        )

        return passing_constraint

    def _passing_gpu_metric_constraint(
        self,
        model_name: ModelName,
        constraint_name: ConstraintName,
        constraint_value: ConstraintValue,
    ) -> bool:
        passing_constraint = True
        gpu_metric = self.get_gpu_metric(constraint_name)

        if gpu_metric:
            avg_gpu_metric_value = mean(
                [
                    gpu_record[constraint_name].value()
                    for gpu_record in gpu_metric.values()
                ]
            )

            avg_gpu_record_dict = list(gpu_metric.values())[0]
            avg_gpu_record = deepcopy(list(avg_gpu_record_dict.values())[0])
            avg_gpu_record._value = avg_gpu_metric_value

            passing_constraint = avg_gpu_record.is_passing_constraint(constraint_value)

        return passing_constraint

    def _passing_perf_metric_constraint(
        self,
        model_name: ModelName,
        constraint_name: ConstraintName,
        constraint_value: ConstraintValue,
    ) -> bool:
        passing_constraint = True
        perf_metric = self.get_model_perf_metric(model_name, constraint_name)

        if perf_metric:
            passing_constraint = perf_metric.is_passing_constraint(constraint_value)

        return passing_constraint
