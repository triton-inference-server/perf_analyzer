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

from dataclasses import dataclass, field
from typing import Any, Dict, List

from genai_perf.config.run.run_config import RunConfig
from genai_perf.measurements.run_constraints import RunConstraints
from genai_perf.types import GpuMetricObjectives, ModelWeights, PerfMetricObjectives


@dataclass
class Results:
    """
    Holds a sorted list of RunConfigs (best is first) and methods
    for adding configs, and setting objectives, weights or constraints
    """

    run_configs: List[RunConfig] = field(default_factory=list)

    ###########################################################################
    # Checkpoint Methods
    ###########################################################################
    def write_to_checkpoint(self) -> Dict[str, Any]:
        """
        Converts the class data into a dictionary that can be written to
        the checkpoint file
        """
        run_config_dicts = [
            run_config.write_to_checkpoint() for run_config in self.run_configs
        ]

        results_dict = {"run_configs": run_config_dicts}

        return results_dict

    @classmethod
    def read_from_checkpoint(cls, results_dict: Dict[str, Any]) -> "Results":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of Results
        """
        results = Results()
        for run_config_dict in results_dict["run_configs"]:
            run_config = RunConfig.read_from_checkpoint(run_config_dict)
            results.add_run_config(run_config)

        return results

    ###########################################################################
    # Get Accessor Methods
    ###########################################################################
    def get_results_passing_constraints(self) -> "Results":
        passing_results = Results()
        for run_config in self.run_configs:
            if run_config.is_passing_constraints():
                passing_results.add_run_config(run_config)

        return passing_results

    def get_results_failing_constraints(self) -> "Results":
        failing_results = Results()
        for run_config in self.run_configs:
            if not run_config.is_passing_constraints():
                failing_results.add_run_config(run_config)

        return failing_results

    ###########################################################################
    # Set Accessor Methods
    ###########################################################################
    def add_run_config(self, run_config: RunConfig) -> None:
        self.run_configs.append(run_config)
        self.run_configs.sort(reverse=True)

    def set_gpu_metric_objectives(
        self, gpu_metric_objectives: GpuMetricObjectives
    ) -> None:
        for run_config in self.run_configs:
            run_config.set_gpu_metric_objectives(gpu_metric_objectives)

        self.run_configs.sort(reverse=True)

    def set_perf_metric_objectives(
        self, perf_metric_objectives: PerfMetricObjectives
    ) -> None:
        for run_config in self.run_configs:
            run_config.set_perf_metric_objectives(perf_metric_objectives)

        self.run_configs.sort(reverse=True)

    def set_model_weighting(self, model_weights: ModelWeights) -> None:
        for run_config in self.run_configs:
            run_config.set_model_weighting(model_weights)

    def set_constraints(self, constraints: RunConstraints) -> None:
        for run_config in self.run_configs:
            run_config.set_constraints(constraints)
