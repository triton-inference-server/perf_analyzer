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
from genai_perf.config.input.config_defaults import OutputDefaults
from genai_perf.config.input.config_field import ConfigField


class ConfigOutput(BaseConfig):
    """
    Describes the configuration output options
    """

    def __init__(self):
        super().__init__()
        self.artifact_directory: Any = ConfigField(
            default=OutputDefaults.ARTIFACT_DIRECTORY
        )
        self.checkpoint_directory: Any = ConfigField(
            default=OutputDefaults.CHECKPOINT_DIRECTORY
        )
        self.profile_export_file: Any = ConfigField(
            default=OutputDefaults.PROFILE_EXPORT_FILE
        )
        self.generate_plots: Any = ConfigField(default=OutputDefaults.GENERATE_PLOTS)

    def parse(self, output: Dict[str, Any]) -> None:
        for key, value in output.items():
            if key == "artifact_directory":
                self.artifact_directory = value
            elif key == "checkpoint_directory":
                self.checkpoint_directory = value
            elif key == "profile_export_file":
                self.profile_export_file = value
            elif key == "generate_plots":
                self.generate_plots = value
            else:
                raise ValueError(f"User Config: {key} is not a valid output parameter")
