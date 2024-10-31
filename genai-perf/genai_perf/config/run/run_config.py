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
from functools import total_ordering
from typing import Any, Dict, Optional

from genai_perf.config.generate.genai_perf_config import GenAIPerfConfig
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.measurements.run_config_measurement import RunConfigMeasurement
from genai_perf.measurements.run_constraints import RunConstraints
from genai_perf.record.record import Record
from genai_perf.types import (
    GpuRecords,
    MetricObjectives,
    ModelName,
    ModelWeights,
    PerfMetricName,
    PerfRecords,
    RunConfigName,
)


@dataclass
@total_ordering
class RunConfig:
    """
    Encapsulates all the information needed to profile a model
    and capture the results
    """

    # TODO: OPTIMIZE
    # triton_env: Dict[str, Any]
    # model_run_configs: List[ModelRunConfig]

    name: RunConfigName
    genai_perf_config: GenAIPerfConfig
    perf_analyzer_config: PerfAnalyzerConfig
    measurement: RunConfigMeasurement

    ###########################################################################
    # Checkpoint Methods
    ###########################################################################
    def write_to_checkpoint(self) -> Dict[str, Any]:
        """
        Converts the class data into a dictionary that can be written to
        the checkpoint file
        """
        run_config_dict = {
            "name": self.name,
            "genai_perf_config": self.genai_perf_config.write_to_checkpoint(),
            "perf_analyzer_config": self.perf_analyzer_config.write_to_checkpoint(),
            "measurement": self.measurement.write_to_checkpoint(),
        }

        return run_config_dict

    @classmethod
    def read_from_checkpoint(cls, run_config_dict: Dict[str, Any]) -> "RunConfig":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of a RCM
        """
        name = run_config_dict["name"]
        genai_perf_config = GenAIPerfConfig.read_from_checkpoint(
            run_config_dict["genai_perf_config"]
        )
        perf_analyzer_config = PerfAnalyzerConfig.read_from_checkpoint(
            run_config_dict["perf_analyzer_config"]
        )
        measurement = RunConfigMeasurement.read_from_checkpoint(
            run_config_dict["measurement"]
        )

        run_config = RunConfig(
            name, genai_perf_config, perf_analyzer_config, measurement
        )

        return run_config

    ###########################################################################
    # Get Accessor Methods
    ###########################################################################
    def get_all_gpu_metrics(self) -> GpuRecords:
        return self.measurement.get_all_gpu_metrics()

    def get_gpu_metric(self, name: str) -> Optional[GpuRecords]:
        return self.measurement.get_gpu_metric(name)

    def get_all_perf_metrics(self) -> Dict[ModelName, PerfRecords]:
        return self.measurement.get_all_perf_metrics()

    def get_model_perf_metrics(self, model_name: ModelName) -> Optional[PerfRecords]:
        return self.measurement.get_model_perf_metrics(model_name)

    def get_model_perf_metric(
        self, model_name: ModelName, perf_metric_name: PerfMetricName
    ) -> Optional[Record]:
        return self.measurement.get_model_perf_metric(model_name, perf_metric_name)

    def get_model_perf_metric_value(
        self,
        model_name: ModelName,
        perf_metric_name: PerfMetricName,
        return_value: int = 0,
    ) -> Any:
        return self.measurement.get_model_perf_metric_value(
            model_name, perf_metric_name, return_value
        )

    def get_weighted_perf_metric_values(
        self,
        perf_metric_name: PerfMetricName,
        return_value: int = 0,
    ) -> Dict[ModelName, Any]:
        return self.measurement.get_weighted_perf_metric_values(
            perf_metric_name, return_value
        )

    ###########################################################################
    # Set Accessor Methods
    ###########################################################################
    def set_model_weighting(self, model_weights: ModelWeights) -> None:
        self.measurement.set_model_weighting(model_weights)

    def set_gpu_metric_objectives(
        self, gpu_metric_objectives: Dict[ModelName, MetricObjectives]
    ) -> None:
        self.measurement.set_gpu_metric_objectives(gpu_metric_objectives)

    def set_perf_metric_objectives(
        self, perf_metric_objectives: Dict[ModelName, MetricObjectives]
    ) -> None:
        self.measurement.set_perf_metric_objectives(perf_metric_objectives)

    def set_constraints(self, constraints: RunConstraints) -> None:
        self.measurement.set_constraints(constraints)

    def add_perf_metrics(
        self,
        model_name: ModelName,
        perf_metrics: PerfRecords,
    ) -> None:
        self.measurement.add_perf_metrics(model_name, perf_metrics)

    ###########################################################################
    # Comparison Methods
    ###########################################################################
    def __lt__(self, other: "RunConfig") -> bool:
        return self.measurement < other.measurement

    def __gt__(self, other: "RunConfig") -> bool:
        return self.measurement > other.measurement

    def __eq__(self, other: "RunConfig") -> bool:  # type: ignore
        return self.measurement == other.measurement
