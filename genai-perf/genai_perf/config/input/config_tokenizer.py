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
from genai_perf.config.input.config_defaults import TokenizerDefaults
from genai_perf.config.input.config_field import ConfigField


class ConfigTokenizer(BaseConfig):
    """
    Describes the configuration tokenizer options
    """

    def __init__(self) -> None:
        super().__init__()
        self.name: Any = ConfigField(default=TokenizerDefaults.NAME)
        self.revision: Any = ConfigField(default=TokenizerDefaults.REVISION)
        self.trust_remote_code: Any = ConfigField(
            default=TokenizerDefaults.TRUST_REMOTE_CODE
        )

    def parse(self, tokenizer: Dict[str, Any]) -> None:
        for key, value in tokenizer.items():
            if key == "name":
                self.name = value
            elif key == "revision":
                self.revision = value
            elif key == "trust_remote_code":
                self.trust_remote_code = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid tokenizer parameter"
                )
