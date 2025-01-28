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
from genai_perf.config.input.config_defaults import PerfAnalyzerDefaults
from genai_perf.config.input.config_field import ConfigField


class ConfigPerfAnalyzer(BaseConfig):
    """
    Describes the configuration for PerfAnalyzer options
    """

    def __init__(self):
        super().__init__()
        self.path: Any = ConfigField(default=PerfAnalyzerDefaults.PATH)
        self.stimulus: Any = ConfigField(
            default=PerfAnalyzerDefaults.STIMULUS,
            choices=["concurrency", "request_rate"],
        )
        self.stability_percentage: Any = ConfigField(
            default=PerfAnalyzerDefaults.STABILITY_PERCENTAGE,
            bounds={"min": 1, "max": 999},
        )
        self.measurement_interval: Any = ConfigField(
            default=PerfAnalyzerDefaults.MEASUREMENT_INTERVAL, bounds={"min": 1}
        )
        self.skip_args: Any = ConfigField(default=PerfAnalyzerDefaults.SKIP_ARGS)

    def parse(self, perf_analyzer: Dict[str, Any]) -> None:
        for key, value in perf_analyzer.items():
            if key == "path":
                self.path = value
            elif key == "stimulus":
                self.stimulus = value
            elif key == "stability_percentage":
                self.stability_percentage = value
            elif key == "measurement_interval":
                self.measurement_interval = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid perf_analyzer parameter"
                )
