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

import genai_perf.logging as logging
from genai_perf.config.input.base_config import BaseConfig
from genai_perf.config.input.config_defaults import (
    AudioDefaults,
    ImageDefaults,
    InputDefaults,
    OutputTokenDefaults,
    PrefixPromptDefaults,
    SessionDefaults,
    SessionTurnDelayDefaults,
    SessionTurnsDefaults,
    SyntheticTokenDefaults,
)
from genai_perf.config.input.config_field import ConfigField
from genai_perf.inputs.input_constants import AudioFormat, ImageFormat, PromptSource
from genai_perf.utils import split_and_strip_whitespace

logger = logging.getLogger(__name__)


class ConfigInput(BaseConfig):
    """
    Describes the configuration input options
    """

    def __init__(self) -> None:
        super().__init__()
        self.batch_size: Any = ConfigField(
            default=InputDefaults.BATCH_SIZE,
            verbose_template_comment="The batch size of text requests GenAI-Perf should send.\
            \nThis is currently supported with the embeddings and rankings endpoint types",
        )
        self.extra: Any = ConfigField(
            default=InputDefaults.EXTRA,
            verbose_template_comment="Provide additional inputs to include with every request.\
                \nInputs should be in an 'input_name:value' format.",
        )
        self.goodput: Any = ConfigField(
            default=InputDefaults.GOODPUT,
            verbose_template_comment="An option to provide constraints in order to compute goodput.\
                \nSpecify goodput constraints as 'key:value' pairs,\
                \nwhere the key is a valid metric name, and the value is a number representing\
                \neither milliseconds or a throughput value per second.\
                \nFor example:\
                \n  request_latency:300\
                \n  output_token_throughput_per_user:600",
        )
        self.header: Any = ConfigField(
            default=InputDefaults.HEADER,
            verbose_template_comment="Adds a custom header to the requests.\
                \nHeaders must be specified as 'Header:Value' pairs.",
        )
        self.file: Any = ConfigField(
            default=InputDefaults.FILE,
            verbose_template_comment="The file or directory containing the content to use for profiling.\
                \nExample:\
                \n  text: \"Your prompt here\"\
                \n\nTo use synthetic files for a converter that needs multiple files,\
                \nprefix the path with 'synthetic:' followed by a comma-separated list of file names.\
                \nThe synthetic filenames should not have extensions.\
                \nExample:\
                \n  synthetic: queries,passages",
        )
        self.num_dataset_entries: Any = ConfigField(
            default=InputDefaults.NUM_DATASET_ENTRIES,
            bounds={"min": 1},
            verbose_template_comment="The number of unique payloads to sample from.\
                \nThese will be reused until benchmarking is complete.",
        )
        self.random_seed: Any = ConfigField(
            default=InputDefaults.RANDOM_SEED,
            verbose_template_comment="The seed used to generate random values.",
        )

        self.audio = ConfigAudio()
        self.image = ConfigImage()
        self.output_tokens = ConfigOutputTokens()
        self.synthetic_tokens = ConfigSyntheticTokens()
        self.prefix_prompt = ConfigPrefixPrompt()

        self.sessions = ConfigSessions()

    ###########################################################################
    # Parse Methods
    ###########################################################################
    def parse(self, input: Dict[str, Any]) -> None:
        for key, value in input.items():
            if key == "batch_size":
                self.batch_size = value
            elif key == "extra":
                self.extra = value
            elif key == "goodput":
                if value:
                    self._parse_goodput(value)
            elif key == "header":
                self.header = value
            elif key == "file":
                self._parse_file(value)
            elif key == "num_dataset_entries":
                self.num_dataset_entries = value
            elif key == "random_seed":
                self.random_seed = value
            elif key == "audio":
                self.audio.parse(value)
            elif key == "image":
                self.image.parse(value)
            elif key == "output_tokens":
                self.output_tokens.parse(value)
            elif key == "synthetic_tokens":
                self.synthetic_tokens.parse(value)
            elif key == "prefix_prompt":
                self.prefix_prompt.parse(value)
            elif key == "sessions":
                self.sessions.parse(value)
            else:
                raise ValueError(f"User Config: {key} is not a valid input parameter")

    def _parse_goodput(self, goodputs: Dict[str, Any]) -> None:
        constraints = {}
        for target_metric, target_value in goodputs.items():
            if isinstance(target_value, int) or isinstance(target_value, float):
                if target_value < 0:
                    raise ValueError(
                        "User Config: Goodput values must be non-negative ({target_metric}: {target_value})"
                    )

                constraints[target_metric] = float(target_value)
            else:
                raise ValueError(
                    "User Config: Goodput values must be integers or floats"
                )

        self.goodput = constraints

    def _parse_file(self, value: str) -> None:
        if not value:
            return
        elif value.startswith("synthetic:") or value.startswith("payload"):
            self.file = Path(value)
        else:
            path = Path(value)
            if path.is_file() or path.is_dir():
                self.file = path
            else:
                raise ValueError(f"'{value}' is not a valid file or directory")

    ###########################################################################
    # Illegal Combination Methods
    ###########################################################################
    def check_for_illegal_combinations(self) -> None:
        self.output_tokens._check_output_tokens()

    ###########################################################################
    # Infer Methods
    ###########################################################################
    def infer_settings(self) -> None:
        self._infer_prompt_source()
        self._infer_synthetic_files()
        self._infer_payload_file()

    def _infer_prompt_source(self) -> None:
        self.prompt_source: Any = ConfigField(
            default=PromptSource.SYNTHETIC, choices=PromptSource, add_to_template=False
        )

        if self.file:
            if str(self.file).startswith("synthetic:"):
                self.prompt_source = PromptSource.SYNTHETIC
            elif str(self.file).startswith("payload:"):
                self.prompt_source = PromptSource.PAYLOAD
            else:
                self.prompt_source = PromptSource.FILE
                logger.debug(f"Input source is the following path: {self.file}")

    def _infer_synthetic_files(self) -> None:
        self.synthetic_files: Any = ConfigField(default=[], add_to_template=False)

        if self.file:
            if str(self.file).startswith("synthetic:"):
                synthetic_files_str = str(self.file).split(":", 1)[1]
                self.synthetic_files = synthetic_files_str.split(",")
                logger.debug(f"Input source is synthetic data: {self.synthetic_files}")

    def _infer_payload_file(self) -> None:
        self.payload_file: Any = ConfigField(default=None, add_to_template=False)

        if self.file:
            if str(self.file).startswith("payload:"):
                self.payload_file = Path(str(self.file).split(":", 1)[1])
                if not self.payload_file:
                    raise ValueError("Invalid file path: Path is None or empty.")
                if not self.payload_file.is_file():
                    raise ValueError(f"File not found: {self.payload_file}")

                logger.debug(
                    f"Input source is a payload file with timing information in the following path: {self.payload_file}"
                )


###########################################################################
# Sub-Config Classes
###########################################################################
class ConfigAudio(BaseConfig):
    """
    Describes the configuration audio options
    """

    def __init__(self) -> None:
        super().__init__()
        self.batch_size: Any = ConfigField(
            default=AudioDefaults.BATCH_SIZE,
            bounds={"min": 0},
            verbose_template_comment="The audio batch size of the requests GenAI-Perf should send.\
                \nThis is currently supported with the OpenAI `multimodal` endpoint type.",
        )
        self.length = ConfigAudioLength()
        self.format: Any = ConfigField(
            default=AudioDefaults.FORMAT,
            choices=AudioFormat,
            verbose_template_comment="The audio format of the audio files (wav or mp3).",
        )
        self.depths: Any = ConfigField(
            default=AudioDefaults.DEPTHS,
            bounds={"min": 1},
            verbose_template_comment="A list of audio bit depths to randomly select from in bits.",
        )
        self.sample_rates: Any = ConfigField(
            default=AudioDefaults.SAMPLE_RATES,
            bounds={"min": 1},
            verbose_template_comment="A list of audio sample rates to randomly select from in kHz.",
        )
        self.num_channels: Any = ConfigField(
            default=AudioDefaults.NUM_CHANNELS,
            choices=[1, 2],
            verbose_template_comment="The number of audio channels to use for the audio data generation.",
        )

    def parse(self, audio: Dict[str, Any]) -> None:
        for key, value in audio.items():
            if key == "batch_size":
                self.batch_size = value
            elif key == "length":
                self.length.parse(value)
            elif key == "format":
                if value:
                    self.format = AudioFormat(value.upper())
            elif key == "depths":
                self.depths = self._parse_int_list(value)
            elif key == "sample_rates":
                self.sample_rates = self._parse_int_list(value)
            elif key == "num_channels":
                self.num_channels = value
            else:
                raise ValueError(f"User Config: {key} is not a valid audio parameter")

    def _parse_int_list(self, value: Any) -> list:
        if isinstance(value, list):
            return [int(i) for i in value]
        elif type(value) is str:
            return list(map(int, split_and_strip_whitespace(value)))
        elif type(value) is int:
            return [int(value)]
        else:
            raise ValueError(
                "User Config: Audio depths and sample rates must be lists of integers"
            )


class ConfigAudioLength(BaseConfig):
    """
    Describes the configuration audio length options
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean: Any = ConfigField(
            default=AudioDefaults.LENGTH_MEAN,
            bounds={"min": 0},
            verbose_template_comment="The mean length of the audio in seconds.",
        )
        self.stddev: Any = ConfigField(
            default=AudioDefaults.LENGTH_STDDEV,
            bounds={"min": 0},
            verbose_template_comment="The standard deviation of the length of the audio in seconds.",
        )

    def parse(self, audio_length: Dict[str, Any]) -> None:
        for key, value in audio_length.items():
            if key == "mean":
                self.mean = value
            elif key == "stddev":
                self.stddev = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid audio_length parameter"
                )


class ConfigImage(BaseConfig):
    """
    Describes the configuration image options
    """

    def __init__(self) -> None:
        super().__init__()
        self.width = ConfigImageWidth()
        self.height = ConfigImageHeight()

        self.batch_size: Any = ConfigField(
            default=ImageDefaults.BATCH_SIZE,
            bounds={"min": 0},
            verbose_template_comment="The image batch size of the requests GenAI-Perf should send.\
                \nThis is currently supported with the image retrieval endpoint type.",
        )
        self.format: Any = ConfigField(
            default=ImageDefaults.FORMAT,
            choices=ImageFormat,
            verbose_template_comment="The compression format of the images.",
        )

    def parse(self, image: Dict[str, Any]) -> None:
        for key, value in image.items():
            if key == "batch_size":
                self.batch_size = value
            elif key == "width":
                self.width.parse(value)
            elif key == "height":
                self.height.parse(value)
            elif key == "format":
                if value:
                    self.format = ImageFormat(value.upper())
            else:
                raise ValueError(f"User Config: {key} is not a valid image parameter")


class ConfigImageWidth(BaseConfig):
    """
    Describes the configuration image width options
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean: Any = ConfigField(
            default=ImageDefaults.WIDTH_MEAN,
            bounds={"min": 0},
            verbose_template_comment="The mean width of the images when generating synthetic image data.",
        )
        self.stddev: Any = ConfigField(
            default=ImageDefaults.WIDTH_STDDEV,
            bounds={"min": 0},
            verbose_template_comment="The standard deviation of width of images when generating synthetic image data.",
        )

    def parse(self, image_width: Dict[str, Any]) -> None:
        for key, value in image_width.items():
            if key == "mean":
                self.mean = value
            elif key == "stddev":
                self.stddev = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid image_width parameter"
                )


class ConfigImageHeight(BaseConfig):
    """
    Describes the configuration image height options
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean: Any = ConfigField(
            default=ImageDefaults.HEIGHT_MEAN,
            bounds={"min": 0},
            verbose_template_comment="The mean height of images when generating synthetic image data.",
        )
        self.stddev: Any = ConfigField(
            default=ImageDefaults.HEIGHT_STDDEV,
            bounds={"min": 0},
            verbose_template_comment="The standard deviation of height of images when generating synthetic image data.",
        )

    def parse(self, image_height: Dict[str, Any]) -> None:
        for key, value in image_height.items():
            if key == "mean":
                self.mean = value
            elif key == "stddev":
                self.stddev = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid image_height parameter"
                )


class ConfigOutputTokens(BaseConfig):
    """
    Describes the configuration output tokens options
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean: Any = ConfigField(
            default=OutputTokenDefaults.MEAN,
            bounds={"min": 0},
            verbose_template_comment="The mean number of tokens in each output.",
        )
        self.deterministic: Any = ConfigField(
            default=OutputTokenDefaults.DETERMINISTIC,
            verbose_template_comment="This can be set to improve the precision of the mean by setting the\
            \nminimum number of tokens equal to the requested number of tokens.\
            \nThis is currently supported with Triton.",
        )
        self.stddev: Any = ConfigField(
            default=OutputTokenDefaults.STDDEV,
            bounds={"min": 0},
            verbose_template_comment="The standard deviation of the number of tokens in each output.",
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

    def _check_output_tokens(self) -> None:
        if (
            self.get_field("stddev").is_set_by_user
            and not self.get_field("mean").is_set_by_user
        ):
            raise ValueError(
                "User Config: If output tokens stddev is set, mean must also be set"
            )

        if (
            self.get_field("deterministic").is_set_by_user
            and not self.get_field("mean").is_set_by_user
        ):
            raise ValueError(
                "User Config: If output tokens deterministic is set, mean must also be set"
            )


class ConfigSyntheticTokens(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.mean: Any = ConfigField(
            default=SyntheticTokenDefaults.MEAN,
            bounds={"min": 0},
            verbose_template_comment="The mean of number of tokens in the generated prompts when using synthetic data.",
        )
        self.stddev: Any = ConfigField(
            default=SyntheticTokenDefaults.STDDEV,
            bounds={"min": 0},
            verbose_template_comment="The standard deviation of number of tokens in the generated prompts when using synthetic data.",
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
        self.num: Any = ConfigField(
            default=PrefixPromptDefaults.NUM,
            bounds={"min": 0},
            verbose_template_comment="The number of prefix prompts to select from.\
            \nIf this value is not zero, these are prompts that are prepended to input prompts.\
            \nThis is useful for benchmarking models that use a K-V cache.",
        )
        self.length: Any = ConfigField(
            default=PrefixPromptDefaults.LENGTH,
            bounds={"min": 0},
            verbose_template_comment='The number of tokens in each prefix prompt.\
            \nThis is only used if "num" is greater than zero.\
            \nNote that due to the prefix and user prompts being concatenated,\
            \nthe number of tokens in the final prompt may be off by one.',
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


class ConfigSessions(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.num: Any = ConfigField(
            default=SessionDefaults.NUM,
            bounds={"min": 0},
            verbose_template_comment="The number of sessions to simulate",
        )
        self.turns = ConfigSessionTurns()
        self.turn_delay = ConfigSessionTurnDelay()

    def parse(self, sessions: Dict[str, Any]) -> None:
        for key, value in sessions.items():
            if key == "num":
                self.num = value
            elif key == "turns":
                self.turns.parse(value)
            elif key == "turn_delay":
                self.turn_delay.parse(value)
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid sessions parameter"
                )


class ConfigSessionTurns(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.mean: Any = ConfigField(
            default=SessionTurnsDefaults.MEAN,
            bounds={"min": 0},
            verbose_template_comment="The mean number of turns in a session",
        )
        self.stddev: Any = ConfigField(
            default=SessionTurnsDefaults.STDDEV,
            bounds={"min": 0},
            verbose_template_comment="The standard deviation of the number of turns in a session",
        )

    def parse(self, turns: Dict[str, Any]) -> None:
        for key, value in turns.items():
            if key == "mean":
                self.mean = value
            elif key == "stddev":
                self.stddev = value
            else:
                raise ValueError(f"User Config: {key} is not a valid turns parameter")


class ConfigSessionTurnDelay(BaseConfig):
    def __init__(self) -> None:
        super().__init__()
        self.mean: Any = ConfigField(
            default=SessionTurnDelayDefaults.MEAN,
            bounds={"min": 0},
            verbose_template_comment="The mean delay (in ms) between turns in a session",
        )
        self.ratio: Any = ConfigField(
            default=SessionDefaults.DELAY_RATIO,
            bounds={"min": 0},
            verbose_template_comment="A ratio to scale multi-turn delays when using a payload file",
        )
        self.stddev: Any = ConfigField(
            default=SessionTurnDelayDefaults.STDDEV,
            bounds={"min": 0},
            verbose_template_comment="The standard deviation (in ms) of the delay between turns in a session",
        )

    def parse(self, turn_delay: Dict[str, Any]) -> None:
        for key, value in turn_delay.items():
            if key == "mean":
                self.mean = value
            elif key == "stddev":
                self.stddev = value
            elif key == "ratio":
                self.ratio = value
            else:
                raise ValueError(
                    f"User Config: {key} is not a valid turn_delay parameter"
                )
