# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict

from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_defaults import (
    PerfAnalyzerDefaults,
    PerfAnalyzerMeasurementDefaults,
)
from genai_perf.config.input.config_field import ConfigField
from genai_perf.inputs.input_constants import PerfAnalyzerMeasurementMode


class ConfigPerfAnalyzer(BaseConfig):
    """
    Describes the configuration for PerfAnalyzer options
    """

    def __init__(self) -> None:
        super().__init__()
        self.path: Any = ConfigField(
            default=PerfAnalyzerDefaults.PATH,
            verbose_template_comment="Path to Perf Analyzer binary",
        )
        self.verbose: Any = ConfigField(
            default=PerfAnalyzerDefaults.VERBOSE,
            verbose_template_comment="Enables verbose output from Perf Analyzer",
        )
        self.stimulus: Any = ConfigField(
            default=PerfAnalyzerDefaults.STIMULUS,
            choices=[
                "concurrency",
                "request_rate",
                "session_concurrency",
                "session_request_rate",
                "fixed_schedule",
            ],
            verbose_template_comment="The type and value of stimulus to benchmark",
        )
        self.stability_percentage: Any = ConfigField(
            default=PerfAnalyzerDefaults.STABILITY_PERCENTAGE,
            bounds={"min": 1, "max": 999},
            verbose_template_comment="The allowed variation in latency measurements when determining if a result is stable.\
            \nThe measurement is considered as stable if the ratio of max / min\
            \nfrom the recent 3 measurements is within (stability percentage)\
            \nin terms of both infer per second and latency.",
        )
        self.measurement = ConfigPerfAnalyzerMeasurement()

        self.warmup_request_count: Any = ConfigField(
            default=PerfAnalyzerDefaults.WARMUP_REQUEST_COUNT,
            bounds={"min": 0},
            verbose_template_comment="The number of warmup requests to send before benchmarking.",
        )

    def parse(self, perf_analyzer: Dict[str, Any]) -> None:
        for key, value in perf_analyzer.items():
            if key == "path":
                self.path = value
            elif key == "verbose":
                self.verbose = value
            elif key == "stimulus":
                self._parse_stimulus(value)
            elif key == "stability_percentage":
                self.stability_percentage = value
            elif key == "measurement_interval":
                self.measurement_interval = value
            elif key == "warmup_request_count":
                self.warmup_request_count = value
            elif key == "measurement":
                self.measurement.parse(value)
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid perf_analyzer parameter"
                )

    def _parse_stimulus(self, stimulus: Any) -> None:
        if type(stimulus) is str:
            self.stimulus = {stimulus: None}
        elif type(stimulus) is dict:
            self.stimulus = stimulus
        else:
            raise ValueError("User Config: stimulus must be a string or dict")

    def infer_settings(self) -> None:
        self.measurement._infer_measurement_num_based_on_mode()


###########################################################################
# Sub-Config Classes
###########################################################################
class ConfigPerfAnalyzerMeasurement(BaseConfig):
    def __init__(self) -> None:
        super().__init__()

        self.mode: Any = ConfigField(
            default=PerfAnalyzerMeasurementDefaults.MODE,
            choices=PerfAnalyzerMeasurementMode,
            verbose_template_comment="The mode of measurement to use in PA: request_count or interval",
        )
        self.num: Any = ConfigField(
            default=PerfAnalyzerMeasurementDefaults.NUM,
            bounds={"min": 0},
            template_comment="If left unset, the default value will be inferred based on the mode.",
            verbose_template_comment="The number to use for measurement. By default, this will be inferred based on the mode.",
        )

    def parse(self, perf_analyzer_measurement: Dict[str, Any]) -> None:
        for key, value in perf_analyzer_measurement.items():
            if key == "mode":
                if value:
                    self.mode = PerfAnalyzerMeasurementMode(value.upper())
            elif key == "num":
                self.num = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid perf_analyzer_measurement parameter"
                )

    ###########################################################################
    # Infer Methods
    ###########################################################################
    def _infer_measurement_num_based_on_mode(self) -> None:
        if not self.get_field("num").is_set_by_user:
            if self.mode == PerfAnalyzerMeasurementMode.REQUEST_COUNT:
                # This will be set at PA Config generation time
                pass
            elif self.mode == PerfAnalyzerMeasurementMode.INTERVAL:
                self.num = PerfAnalyzerMeasurementDefaults.INTERVAL
            else:
                raise ValueError(
                    f"User Config: {self.mode} is not a valid perf_analyzer_measurement mode"
                )
