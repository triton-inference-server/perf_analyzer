# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
from typing import List

from genai_perf.config.input.config_defaults import default_field
from genai_perf.config.run.run_config import RunConfig
from genai_perf.measurements.run_constraints import RunConstraints
from genai_perf.types import (
    CheckpointObject,
    GpuMetricObjectives,
    ModelName,
    ModelWeights,
    PerfMetricObjectives,
    RunConfigName,
)


@dataclass(frozen=True)
class ResultsDefaults:
    STARTING_ID = -1


@dataclass
class Results:
    """
    Holds a sorted list of RunConfigs (best is first) and methods
    for adding configs, and setting objectives, weights or constraints
    """

    run_configs: List[RunConfig] = default_field([])

    ###########################################################################
    # Checkpoint Methods
    ###########################################################################
    def create_checkpoint_object(self) -> CheckpointObject:
        """
        Converts the class data into a dictionary that can be written to
        the checkpoint file
        """
        run_config_dicts = [
            run_config.create_checkpoint_object() for run_config in self.run_configs
        ]

        results_dict = {"run_configs": run_config_dicts}

        return results_dict

    @classmethod
    def create_class_from_checkpoint(cls, results_dict: CheckpointObject) -> "Results":
        """
        Takes the checkpoint's representation of the class and creates (and populates)
        a new instance of Results
        """
        results = Results()
        for run_config_dict in results_dict["run_configs"]:
            run_config = RunConfig.create_class_from_checkpoint(run_config_dict)
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

    def get_run_config_name_based_on_representation(
        self, model_name: ModelName, representation: str
    ) -> RunConfigName:
        """
        Returns the name of the RunConfig if the representation is found,
        else creates a new name by incrementing the config ID
        """
        max_run_config_id = ResultsDefaults.STARTING_ID
        for run_config in self.run_configs:
            if representation == run_config.representation():
                return run_config.name
            else:
                max_run_config_id = max(
                    max_run_config_id, int(run_config.get_name_id())
                )

        return f"{model_name}_run_config_{max_run_config_id+1}"

    ###########################################################################
    # Set Accessor Methods
    ###########################################################################
    def add_run_config(self, run_config: RunConfig) -> None:
        self.run_configs.append(deepcopy(run_config))
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

    ###########################################################################
    # Misc Methods
    ###########################################################################
    def found_representation(self, representation: str) -> bool:
        for run_config in self.run_configs:
            if representation == run_config.representation():
                return True

        return False
