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
from genai_perf.config.input.config_defaults import ProcessDefaults
from genai_perf.config.input.config_field import ConfigField


class ConfigProcess(BaseConfig):
    """
    Describes the configuration for the process-export-files subcommand
    """

    def __init__(self) -> None:
        super().__init__()
        # yapf: disable
        process_export_files_comment = \
        ("Uncomment the lines below to enable the process-export-files subcommand\n"
        "  input_path: "
        )
        # yapf: enable

        self.input_path: Any = ConfigField(
            default=ProcessDefaults.INPUT_PATH,
            add_to_template=False,
            template_comment=process_export_files_comment,
        )

    ###########################################################################
    # Parsing Methods
    ###########################################################################
    def parse(self, process: Dict[str, Any]) -> None:
        if not process:
            return
        self.input_path = Path(process["input_path"])
