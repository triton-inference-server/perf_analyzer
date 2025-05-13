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

from typing import Any, Dict, Optional

import genai_perf.logging as logging
from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_defaults import TokenizerDefaults
from genai_perf.config.input.config_field import ConfigField

logger = logging.getLogger(__name__)


class ConfigTokenizer(BaseConfig):
    """
    Describes the configuration tokenizer options
    """

    def __init__(self, enable_debug_logging: bool = True) -> None:
        super().__init__()
        self.name: Any = ConfigField(
            default=TokenizerDefaults.NAME,
            template_comment="By default this is the model's name",
            verbose_template_comment="The HuggingFace tokenizer to use to interpret token metrics\
                \nfrom prompts and responses. The value can be the\
                \nname of a tokenizer or the filepath of the tokenizer.\
                \nThe default value is the model name.",
        )
        self.revision: Any = ConfigField(
            default=TokenizerDefaults.REVISION,
            verbose_template_comment="The specific model version to use.\
                                    \nIt can be a branch name, tag name, or commit ID.",
        )
        self.trust_remote_code: Any = ConfigField(
            default=TokenizerDefaults.TRUST_REMOTE_CODE,
            verbose_template_comment="Allows custom tokenizer to be downloaded and executed.\
                \nThis carries security risks and should only be used for repositories you trust.\
                \nThis is only necessary for custom tokenizers stored in HuggingFace Hub.",
        )

        self._enable_debug_logging: Any = ConfigField(
            default=None, add_to_template=False, value=enable_debug_logging
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

    def infer_settings(self, model_name: Optional[str] = None) -> None:
        """
        Infer settings that are not explicitly set by the user.

        Args:
            model_name: The model name to use for inferring tokenizer name if not set
        """
        if not self.get_field("name").is_set_by_user and model_name:
            self.name = model_name
            if self._enable_debug_logging:
                logger.debug(f"Inferred tokenizer from model name: {self.name}")
