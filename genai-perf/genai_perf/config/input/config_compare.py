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

from pathlib import Path
from typing import Any, Dict

from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_defaults import CompareDefaults
from genai_perf.config.input.config_field import ConfigField


class ConfigCompare(BaseConfig):
    """
    Describes the configuration compare options
    """

    def __init__(self) -> None:
        super().__init__()
        # yapf: disable
        compare_template_comment = \
            (f"Uncomment the lines below to enable the compare subcommand\n"
             f"plot_config:\n"
             f"files:")

        verbose_compare_template_comment = \
            (f"Uncomment the lines below to enable the compare subcommand\n"
             f"#  The path to the YAML file that specifies plot configurations\n"
             f"#  for comparing multiple runs.\n"
             f"plot_config:\n"
             f"#\n"
             f"#  List of paths to the profile export JSON files.\n"
             f"#  Users can specify this option instead of `plot_config` if they would like\n"
             f"#  GenAI-Perf to generate default plots as well as initial YAML config file.\n"
             f"files:")
        # yapf: enable

        self.plot_config: Any = ConfigField(
            default=CompareDefaults.PLOT_CONFIG,
            add_to_template=False,
            template_comment=compare_template_comment,
            verbose_template_comment=verbose_compare_template_comment,
        )
        self.files: Any = ConfigField(
            default=CompareDefaults.FILES,
            add_to_template=False,
        )

    def parse(self, output: Dict[str, Any]) -> None:
        for key, value in output.items():
            if key == "plot_config":
                self.plot_config = Path(value)
            elif key == "files":
                self.files = value
            else:
                raise ValueError(f"User Config: {key} is not a valid compare parameter")

    ###########################################################################
    # Illegal Combination Methods
    ###########################################################################
    def check_for_illegal_combinations(self) -> None:
        if self.plot_config and self.files:
            raise ValueError(
                "User Config: `plot_config` and `files` cannot be set at the same time"
            )
