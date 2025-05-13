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

from typing import Any, Dict, List, Tuple

from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_defaults import AnalyzeDefaults, Range
from genai_perf.config.input.config_field import ConfigField
from genai_perf.constants import (
    all_parameters,
    runtime_gap_parameters,
    runtime_pa_parameters,
)


class ConfigAnalyze(BaseConfig):
    """
    Describes the configuration for the analyze subcommand
    """

    def __init__(self) -> None:
        super().__init__()
        # yapf: disable
        sweep_parameter_template_comment = \
        (f"Uncomment the lines below to enable the analyze subcommand\n"
         f"# For further details see analyze.md\n"
         f"  concurrency:\n"
         f"    start: {AnalyzeDefaults.MIN_CONCURRENCY}\n"
         f"    stop: {AnalyzeDefaults.MAX_CONCURRENCY}")
        # yapf: enable

        self.sweep_parameters: Any = ConfigField(
            default=AnalyzeDefaults.SWEEP_PARAMETER,
            choices=all_parameters,
            add_to_template=False,
            template_comment=sweep_parameter_template_comment,
        )

    ###########################################################################
    # Parsing Methods
    ###########################################################################
    def parse(self, analyze: Dict[str, Any]) -> None:
        if not analyze:
            return

        sweep_parameters: Dict[str, Any] = {}
        for sweep_type, range_dict in analyze.items():
            if (
                sweep_type in runtime_pa_parameters
                or sweep_type in runtime_gap_parameters
            ):
                if "step" in range_dict:
                    range_list = self._create_range_list(sweep_type, range_dict)
                    sweep_parameters[sweep_type] = range_list
                else:
                    start, stop = self._determine_start_and_stop(sweep_type, range_dict)
                    sweep_parameters[sweep_type] = Range(min=start, max=stop)
            else:
                raise ValueError(
                    f"User Config: {sweep_type} is not a valid analyze parameter"
                )
        self.sweep_parameters = sweep_parameters

    def _create_range_list(
        self, sweep_type: str, range_dict: Dict[str, int]
    ) -> List[int]:
        start = self._get_start(sweep_type, range_dict)
        stop = self._get_stop(sweep_type, range_dict)
        step = self._get_step(sweep_type, range_dict)

        return [value for value in range(start, stop + 1, step)]

    def _determine_start_and_stop(
        self, sweep_type: str, range_dict: Dict[str, int]
    ) -> Tuple[int, int]:
        start = self._get_start(sweep_type, range_dict)
        stop = self._get_stop(sweep_type, range_dict)

        return start, stop

    def _get_start(self, sweep_type: str, range_dict: Dict[str, int]) -> int:
        if "start" in range_dict:
            return range_dict["start"]
        else:
            return self._get_default_start(sweep_type)

    def _get_stop(self, sweep_type: str, range_dict: Dict[str, int]) -> int:
        if "stop" in range_dict:
            return range_dict["stop"]
        else:
            return self._get_default_stop(sweep_type)

    def _get_step(self, sweep_type: str, range_dict: Dict[str, int]) -> int:
        if "step" in range_dict:
            return range_dict["step"]
        else:
            return self._get_default_step(sweep_type)

    def _get_default_start(self, sweep_type: str) -> int:
        if sweep_type == "concurrency":
            return AnalyzeDefaults.MIN_CONCURRENCY
        elif sweep_type == "runtime_batch_size":
            return AnalyzeDefaults.MIN_MODEL_BATCH_SIZE
        elif sweep_type == "request_rate":
            return AnalyzeDefaults.MIN_REQUEST_RATE
        elif sweep_type == "num_dataset_entries":
            return AnalyzeDefaults.MIN_NUM_DATASET_ENTRIES
        elif sweep_type == "input_sequence_length":
            return AnalyzeDefaults.MIN_INPUT_SEQUENCE_LENGTH
        else:
            raise ValueError(f"User Config: {sweep_type} is not a valid sweep type")

    def _get_default_stop(self, sweep_type: str) -> int:
        if sweep_type == "concurrency":
            return AnalyzeDefaults.MAX_CONCURRENCY
        elif sweep_type == "runtime_batch_size":
            return AnalyzeDefaults.MAX_MODEL_BATCH_SIZE
        elif sweep_type == "request_rate":
            return AnalyzeDefaults.MAX_REQUEST_RATE
        elif sweep_type == "num_dataset_entries":
            return AnalyzeDefaults.MAX_NUM_DATASET_ENTRIES
        elif sweep_type == "input_sequence_length":
            return AnalyzeDefaults.MAX_INPUT_SEQUENCE_LENGTH
        else:
            raise ValueError(f"User Config: {sweep_type} is not a valid sweep type")

    def _get_default_step(self, sweep_type: str) -> int:
        return AnalyzeDefaults.STEP
