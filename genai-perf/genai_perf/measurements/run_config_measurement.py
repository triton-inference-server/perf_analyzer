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

import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import total_ordering
from typing import Any, Dict, List, Optional, TypeAlias, Union

from genai_perf.measurements.model_config_measurement import (
    MetricObjectives,
    ModelConfigMeasurement,
    PerfRecords,
)
from genai_perf.measurements.run_constraints import RunConstraints
from genai_perf.record.gpu_record import GPURecord
from genai_perf.record.record import Record

logger = logging.getLogger(__name__)

ModelName: TypeAlias = str
ModelWeights: TypeAlias = Dict[ModelName, Union[int, float]]
ModelConfigMeasurements: TypeAlias = Dict[ModelName, ModelConfigMeasurement]

GpuId: TypeAlias = str
TelemetryRecords: TypeAlias = Dict[str, GPURecord]
GpuRecords: TypeAlias = Dict[GpuId, TelemetryRecords]

WeightedMcmScores: TypeAlias = Dict[ModelName, float]


@dataclass(frozen=True)
class RunConfigMeasurementDefaults:
    MODEL_WEIGHTING = 1

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
        gpu_metrics: GpuRecords,
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
        self._gpu_metrics = gpu_metrics

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
        self, model_name: ModelName, perf_metric_name: str
    ) -> Optional[Record]:
        if model_name in self._model_config_measurements:
            return self._model_config_measurements[model_name].get_perf_metric(
                perf_metric_name
            )
        else:
            return None

    def get_model_perf_metric_value(
        self, model_name: ModelName, perf_metric_name: str, return_value: int = 0
    ) -> Any:
        if model_name in self._model_config_measurements:
            return self._model_config_measurements[model_name].get_perf_metric_value(
                perf_metric_name, return_value
            )
        else:
            return return_value

    def get_weighted_perf_metric_values(
        self,
        perf_metric_name: str,
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

    def set_perf_metric_objectives(
        self, perf_metric_objectives: Dict[ModelName, MetricObjectives]
    ) -> None:
        """
        Sets the metric weighting for all perf metric based measurements
        """
        assert perf_metric_objectives.keys() == self._model_config_measurements.keys()

        for model_name, metric_objectives in perf_metric_objectives.items():
            self._model_config_measurements[model_name].set_metric_objectives(
                metric_objectives
            )

    def set_constraints(self, constraints: RunConstraints) -> None:
        self._constraints = constraints

    def add_model_config_measurement(
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
    def write_to_checkpoint(self) -> Dict[str, Any]:
        """
        Converts the class data into a dictionary that can be written to
        the checkpoint file
        """
        rcm_dict = deepcopy(self.__dict__)

        # Values based solely on user/config settings (that can vary from run to run)
        # are not stored in the checkpoint
        del rcm_dict["_model_weights"]
        del rcm_dict["_constraints"]

        return rcm_dict

    @classmethod
    def read_from_checkpoint(cls, rcm_dict: Dict[str, Any]) -> "RunConfigMeasurement":
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
                record = record.read_from_checkpoint(record_dict)  # type: ignore
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
                ModelConfigMeasurement.read_from_checkpoint(mcm_dict)
            )

        return model_config_measurements

    ###########################################################################
    # Comparison Methods
    ###########################################################################
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

    def _compare_measurements(self, other: "RunConfigMeasurement") -> int:
        """
        Compares two RunConfigMeasurements based on each
        ModelConfigs weighted metric objectives and the
        ModelConfigs weighted value within the RunConfigMeasurement

        Returns
        -------
        float
           Positive value if other is better
           Negative value is self is better
           Zero if they are equal
        """
        # Step 1: for each model determine the weighted score
        weighted_mcm_scores = self._calculate_weighted_mcm_scores(other)

        # Step 2: combine these using the model weighting
        weighted_rcm_score = self._calculate_weighted_rcm_score(weighted_mcm_scores)

        # Step 3: Determine which RCM is better
        if weighted_rcm_score > RunConfigMeasurementDefaults.COMPARISON_SCORE_THRESHOLD:
            return RunConfigMeasurementDefaults.SELF_IS_BETTER
        elif (
            weighted_rcm_score < RunConfigMeasurementDefaults.COMPARISON_SCORE_THRESHOLD
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
        self, weighted_mcm_scores: WeightedMcmScores
    ) -> float:
        """
        A positive value indicates this RCM is better
        than the other RCM
        """
        assert self._model_weights.keys() == weighted_mcm_scores.keys()

        weighted_rcm_scores = [
            weighted_mcm_scores[model_name] * self._model_weights[model_name]
            for model_name in self._model_weights.keys()
        ]

        weighted_rcm_score = sum(weighted_rcm_scores)

        return weighted_rcm_score

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
        weighted_mcm_pct_gains = self._calculate_weighted_mcm_percentage_gains(other)

        # And, then combine these using the model's weighting
        weighted_rcm_pct_gain = self._calculate_weighted_rcm_score(
            weighted_mcm_pct_gains
        )

        return weighted_rcm_pct_gain

    def _calculate_weighted_mcm_percentage_gains(
        self, other: "RunConfigMeasurement"
    ) -> WeightedMcmScores:
        """
        Returns a weighted percentager (as a score) for each ModelConfigMeasurement
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

    # TODO: OPTIMIZE
    # def is_passing_constraints(self) -> bool:
    #     """
    #     Returns true if all model measurements pass
    #     their respective constraints
    #     """

    #     assert self._constraint_manager is not None
    #     return self._constraint_manager.satisfies_constraints(self)

    # def compare_constraints(self, other: "RunConfigMeasurement") -> Optional[float]:
    #     """
    #     Compares two RunConfigMeasurements based on how close
    #     each RCM is to passing their constraints

    #     Parameters
    #     ----------
    #     other: RunConfigMeasurement

    #     Returns
    #     -------
    #     float
    #        Positive value if other is closer to passing constraints
    #        Negative value if self is closer to passing constraints
    #        Zero if they are equally close to passing constraints
    #        None if either RCM is passing constraints
    #     """

    #     assert (
    #         self._constraint_manager is not None
    #         and other._constraint_manager is not None
    #     )

    #     if self.is_passing_constraints() or other.is_passing_constraints():
    #         return None

    #     self_failing_pct = self._constraint_manager.constraint_failure_percentage(self)
    #     other_failing_pct = other._constraint_manager.constraint_failure_percentage(
    #         other
    #     )

    #     return (self_failing_pct - other_failing_pct) / 100
