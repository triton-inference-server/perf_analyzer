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

import genai_perf.logging as logging
from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_defaults import (
    ImageDefaults,
    InputDefaults,
    OutputTokenDefaults,
    PrefixPromptDefaults,
    RequestCountDefaults,
    SyntheticTokenDefaults,
)
from genai_perf.config.input.config_field import ConfigField
from genai_perf.inputs.input_constants import PromptSource
from genai_perf.inputs.retrievers.synthetic_image_generator import ImageFormat

logger = logging.getLogger(__name__)


class ConfigInput(BaseConfig):
    """
    Describes the configuration input options
    """

    def __init__(self) -> None:
        super().__init__()
        self.batch_size: Any = ConfigField(default=InputDefaults.BATCH_SIZE)
        self.extra: Any = ConfigField(default=InputDefaults.EXTRA)
        self.goodput: Any = ConfigField(default=InputDefaults.GOODPUT)
        self.header: Any = ConfigField(default=InputDefaults.HEADER)
        self.file: Any = ConfigField(default=InputDefaults.FILE)
        self.num_dataset_entries: Any = ConfigField(
            default=InputDefaults.NUM_DATASET_ENTRIES
        )
        self.random_seed: Any = ConfigField(default=InputDefaults.RANDOM_SEED)

        self.image = ConfigImage()
        self.output_tokens = ConfigOutputTokens()
        self.synthetic_tokens = ConfigSyntheticTokens()
        self.prefix_prompt = ConfigPrefixPrompt()
        self.request_count = ConfigRequestCount()

    def parse(self, input: Dict[str, Any]) -> None:
        for key, value in input.items():
            if key == "batch_size":
                self.batch_size = value
            elif key == "extra":
                self.extra = value
            elif key == "goodput":
                self.goodput = value
            elif key == "header":
                self.header = value
            elif key == "file":
                self.file = value
            elif key == "num_dataset_entries":
                self.num_dataset_entries = value
            elif key == "random_seed":
                self.random_seed = value
            elif key == "image":
                self.image.parse(value)
            elif key == "output_tokens":
                self.output_tokens.parse(value)
            elif key == "synthetic_tokens":
                self.synthetic_tokens.parse(value)
            elif key == "prefix_prompt":
                self.prefix_prompt.parse(value)
            elif key == "request_count":
                self.request_count.parse(value)
            else:
                raise ValueError(f"User Config: {key} is not a valid input parameter")

    ###########################################################################
    # Infer Methods
    ###########################################################################
    def infer_settings(self) -> None:
        self._infer_prompt_source()
        self._infer_synthetic_input_files()

    def _infer_prompt_source(self) -> None:
        self.prompt_source: Any = ConfigField(
            default=PromptSource.SYNTHETIC, choices=PromptSource
        )

        if self.file:
            if str(self.file).startswith("synthetic:"):
                self.prompt_source = PromptSource.SYNTHETIC
            else:
                self.prompt_source = PromptSource.FILE
                logger.debug(f"Input source is the following path: {self.file}")

    def _infer_synthetic_input_files(self) -> None:
        self.synthetic_input_files: Any = ConfigField(default=[])

        if self.file:
            if str(self.file).startswith("synthetic:"):
                synthetic_input_files_str = str(self.file).split(":", 1)[1]
                self.synthetic_input_files = synthetic_input_files_str.split(",")
                logger.debug(
                    f"Input source is synthetic data: {self.synthetic_input_files}"
                )


class ConfigImage(BaseConfig):
    """
    Describes the configuration image options
    """

    def __init__(self) -> None:
        super().__init__()
        self.batch_size: Any = ConfigField(
            default=ImageDefaults.BATCH_SIZE, bounds={"min": 0}
        )
        self.width_mean: Any = ConfigField(
            default=ImageDefaults.WIDTH_MEAN, bounds={"min": 0}
        )
        self.width_stddev: Any = ConfigField(
            default=ImageDefaults.WIDTH_STDDEV, bounds={"min": 0}
        )
        self.height_mean: Any = ConfigField(
            default=ImageDefaults.HEIGHT_MEAN, bounds={"min": 0}
        )
        self.height_stddev: Any = ConfigField(
            default=ImageDefaults.HEIGHT_STDDEV, bounds={"min": 0}
        )
        self.format: Any = ConfigField(
            default=ImageDefaults.FORMAT, choices=ImageFormat
        )

    def parse(self, image: Dict[str, Any]) -> None:
        for key, value in image.items():
            if key == "batch_size":
                self.batch_size = value
            elif key == "width_mean":
                self.width_mean = value
            elif key == "width_stddev":
                self.width_stddev = value
            elif key == "height_mean":
                self.height_mean = value
            elif key == "height_stddev":
                self.height_stddev = value
            elif key == "format":
                self.format = ImageFormat(value.upper())
            else:
                raise ValueError(f"User Config: {key} is not a valid image parameter")


class ConfigOutputTokens(BaseConfig):
    """
    Describes the configuration output tokens options
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean: Any = ConfigField(
            default=OutputTokenDefaults.MEAN, bounds={"min": 0}
        )
        self.deterministic: Any = ConfigField(default=OutputTokenDefaults.DETERMINISTIC)
        self.stddev: Any = ConfigField(
            default=OutputTokenDefaults.STDDEV, bounds={"min": 0}
        )

    def parse(self, output_tokens: Dict[str, Any]) -> None:
        for key, value in output_tokens.items():
            if key == "mean":
                self.mean = value
            elif key == "deterministic":
                self.deterministic = value
            elif key == "stddev":
                self.stddev = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid output_tokens parameter"
                )

        self._check_output_tokens()

    def _check_output_tokens(self) -> None:
        if (
            self.get_field("stddev").is_set_by_user
            and not self.get_field("mean").is_set_by_user
        ):
            raise ValueError("User Config: If stddev is set, mean must also be set")

        if (
            self.get_field("deterministic").is_set_by_user
            and not self.get_field("mean").is_set_by_user
        ):
            raise ValueError(
                "User Config: If deterministic is set, mean must also be set"
            )


class ConfigSyntheticTokens(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.mean: Any = ConfigField(
            default=SyntheticTokenDefaults.MEAN, bounds={"min": 0}
        )
        self.stddev: Any = ConfigField(
            default=SyntheticTokenDefaults.STDDEV, bounds={"min": 0}
        )

    def parse(self, synthetic_tokens: Dict[str, Any]) -> None:
        for key, value in synthetic_tokens.items():
            if key == "mean":
                if type(value) is int:
                    self.mean = value
                else:
                    raise ValueError(
                        "User Config: synthetic_tokens mean must be an integer"
                    )
            elif key == "stddev":
                self.stddev = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid synthetic_tokens parameter"
                )


class ConfigPrefixPrompt(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.num: Any = ConfigField(default=PrefixPromptDefaults.NUM, bounds={"min": 0})
        self.length: Any = ConfigField(
            default=PrefixPromptDefaults.LENGTH, bounds={"min": 0}
        )

    def parse(self, prefix_prompt: Dict[str, Any]) -> None:
        for key, value in prefix_prompt.items():
            if key == "num":
                self.num = value
            elif key == "length":
                self.length = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid prefix_prompt parameter"
                )


class ConfigRequestCount(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.warmup: Any = ConfigField(
            default=RequestCountDefaults.WARMUP, bounds={"min": 0}
        )

    def parse(self, request_count: Dict[str, Any]) -> None:
        for key, value in request_count.items():
            if key == "warmup":
                self.warmup = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid request_count parameter"
                )
