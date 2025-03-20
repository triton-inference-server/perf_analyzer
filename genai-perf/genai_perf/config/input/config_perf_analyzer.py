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
    RequestCountDefaults,
)
from genai_perf.config.input.config_field import ConfigField


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
        self.measurement_interval: Any = ConfigField(
            default=PerfAnalyzerDefaults.MEASUREMENT_INTERVAL,
            bounds={"min": 1},
            verbose_template_comment="The time interval used for each measurement in milliseconds.\
                \nPerf Analyzer will sample a time interval specified and take measurement\
                \nover the requests completed within that time interval.",
        )

        self.request_count = ConfigRequestCount()

    def parse(self, perf_analyzer: Dict[str, Any]) -> None:
        for key, value in perf_analyzer.items():
            if key == "path":
                self.path = value
            elif key == "verbose":
                self.verbose = value
            elif key == "stimulus":
                self.stimulus = value
            elif key == "stability_percentage":
                self.stability_percentage = value
            elif key == "measurement_interval":
                self.measurement_interval = value
            elif key == "request_count":
                self.request_count.parse(value)
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid perf_analyzer parameter"
                )

        self._validate_measurement_options()

    def _validate_measurement_options(self) -> None:
        """
        Validates the measurement options:
        1. Measurement_interval and request_count are mutually exclusive.
        2. If neither is set, default to request_count=1.
        """
        measurement_interval_set = self.get_field("measurement_interval").is_set_by_user
        request_count_set = self.request_count.get_field("num").is_set_by_user

        if measurement_interval_set:
            if request_count_set:
                raise ValueError(
                    "The measurement_interval and request_count options are mutually exclusive."
                )
            else:
                self.request_count.num = 0
                self.request_count.get_field("num").is_set_by_user = True


class ConfigRequestCount(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.num: Any = ConfigField(
            default=RequestCountDefaults.NUM,
            bounds={"min": 0},
            verbose_template_comment="The number of requests to use for measurement.\
                \nBy default, the benchmark does not terminate based on request count.\
                \nInstead, it continues until stabilization is detected.",
        )
        self.warmup: Any = ConfigField(
            default=RequestCountDefaults.WARMUP,
            bounds={"min": 0},
            verbose_template_comment="The number of warmup requests to send before benchmarking.",
        )

    def parse(self, request_count: Dict[str, Any]) -> None:
        for key, value in request_count.items():
            if key == "num":
                self.num = value
            elif key == "warmup":
                self.warmup = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid request_count parameter"
                )
