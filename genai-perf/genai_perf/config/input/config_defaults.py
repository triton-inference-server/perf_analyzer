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

from copy import deepcopy
from dataclasses import dataclass, field

from genai_perf.inputs.input_constants import ModelSelectionStrategy, OutputFormat
from genai_perf.inputs.retrievers.synthetic_image_generator import ImageFormat


def default_field(obj):
    return field(default_factory=lambda: deepcopy(obj))


@dataclass
class Range:
    min: int
    max: int

    def __dict__(self):
        return {"min": self.min, "max": self.max}


@dataclass(frozen=True)
class TopLevelDefaults:
    MODEL_NAME = ""


@dataclass(frozen=True)
class AnalyzeDefaults:
    MIN_CONCURRENCY = 1
    MAX_CONCURRENCY = 1024
    MIN_REQUEST_RATE = 16
    MAX_REQUEST_RATE = 8192
    MIN_MODEL_BATCH_SIZE = 1
    MAX_MODEL_BATCH_SIZE = 128
    MIN_NUM_DATASET_ENTRIES = 100
    MAX_NUM_DATASET_ENTRIES = 1000
    MIN_INPUT_SEQUENCE_LENGTH = 100
    MAX_INPUT_SEQUENCE_LENGTH = 1000
    STEP = 1

    STIMULUS_TYPE = "concurrency"
    SWEEP_PARAMETER = {"concurrency": Range(min=MIN_CONCURRENCY, max=MAX_CONCURRENCY)}


@dataclass(frozen=True)
class EndPointDefaults:
    MODEL_SELECTION_STRATEGY = ModelSelectionStrategy.ROUND_ROBIN
    BACKEND = OutputFormat.TENSORRTLLM
    CUSTOM = ""
    TYPE = ""
    SERVICE_KIND = "triton"
    STREAMING = False
    SERVER_METRICS_URLS = ["http://localhost:8002/metrics"]
    URL = "localhost:8001"
    GRPC_METHOD = ""


@dataclass(frozen=True)
class PerfAnalyzerDefaults:
    PATH = "perf_analyzer"
    VERBOSE = False
    STIMULUS = {"concurrency": 1}
    STABILITY_PERCENTAGE = 999
    MEASUREMENT_INTERVAL = 10000


@dataclass(frozen=True)
class ImageDefaults:
    BATCH_SIZE = 1
    WIDTH_MEAN = 100
    WIDTH_STDDEV = 0
    HEIGHT_MEAN = 100
    HEIGHT_STDDEV = 0
    FORMAT = ImageFormat.PNG


@dataclass(frozen=True)
class OutputTokenDefaults:
    MEAN = None
    DETERMINISTIC = False
    STDDEV = 0


@dataclass(frozen=True)
class SyntheticTokenDefaults:
    MEAN = 550
    STDDEV = 0


@dataclass(frozen=True)
class PrefixPromptDefaults:
    NUM = 0
    LENGTH = 100


@dataclass(frozen=True)
class RequestCountDefaults:
    NUM = 0
    WARMUP = 0


@dataclass(frozen=True)
class InputDefaults:
    BATCH_SIZE = 1
    EXTRA = ""
    GOODPUT = ""
    HEADER = ""
    FILE = ""
    NUM_DATASET_ENTRIES = 100
    RANDOM_SEED = 0


@dataclass(frozen=True)
class SessionDefaults:
    NUM = 0


@dataclass(frozen=True)
class SessionTurnsDefaults:
    MEAN = 1
    STDDEV = 0


@dataclass(frozen=True)
class SessionTurnDelayDefaults:
    MEAN = 0
    STDDEV = 0


@dataclass(frozen=True)
class OutputDefaults:
    ARTIFACT_DIRECTORY = "./artifacts"
    CHECKPOINT_DIRECTORY = "./checkpoint"
    PROFILE_EXPORT_FILE = "profile_export.json"
    GENERATE_PLOTS = False


@dataclass(frozen=True)
class TokenizerDefaults:
    NAME = "hf-internal-testing/llama-tokenizer"
    REVISION = "main"
    TRUST_REMOTE_CODE = False


@dataclass(frozen=True)
class TemplateDefaults:
    FILENAME = "genai_perf_config.yaml"
