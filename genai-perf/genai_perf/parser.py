# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import sys
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import genai_perf.logging as logging
import genai_perf.utils as utils
from genai_perf.config.endpoint_config import endpoint_type_map
from genai_perf.config.input.config_command import ConfigCommand, Subcommand
from genai_perf.config.input.config_defaults import (
    AnalyzeDefaults,
    EndPointDefaults,
    TemplateDefaults,
)
from genai_perf.config.input.config_field import ConfigField
from genai_perf.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_PROFILE_EXPORT_FILE
from genai_perf.inputs import input_constants as ic
from genai_perf.subcommand.analyze import analyze_handler
from genai_perf.subcommand.common import get_extra_inputs_as_dict
from genai_perf.subcommand.compare import compare_handler
from genai_perf.subcommand.process_export_files import process_export_files_handler
from genai_perf.subcommand.profile import profile_handler
from genai_perf.subcommand.template import template_handler
from genai_perf.tokenizer import DEFAULT_TOKENIZER_REVISION

from . import __version__


class PathType(Enum):
    FILE = auto()
    DIRECTORY = auto()


logger = logging.getLogger(__name__)


def _parse_goodput(values):
    constraints = {}
    try:
        for item in values:
            target_metric, target_val = item.split(":")
            constraints[target_metric] = float(target_val)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid format found for goodput constraints. "
            f"The expected format is 'key:value' pairs. The key should be a "
            f"service level objective name (e.g. request_latency). The value "
            f"should be a number representing either milliseconds "
            f"or a throughput value per second."
        )
    return constraints


def _check_goodput_args(args):
    """
    Parse and check goodput args
    """
    if args.goodput:
        args.goodput = _parse_goodput(args.goodput)
        for target_metric, target_val in args.goodput.items():
            if target_val < 0:
                raise ValueError(
                    f"Invalid value found, {target_metric}: {target_val}. "
                    f"The goodput constraint value should be non-negative. "
                )
    return args


def _check_compare_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """
    Check compare subcommand args
    """
    if not args.config and not args.files:
        parser.error("Either the --config or --files option must be specified.")
    return args


def _process_sweep_args(args):
    """
    Process the sweep args which can either be a list or
    a min:max:[step]
    """
    if args.sweep_list:
        _parse_sweep_list(args)
    elif args.sweep_range:
        args.sweep_min, args.sweep_max, args.sweep_step = _parse_sweep_range(
            args.sweep_range
        )
        _check_sweep_range(args)

        if args.sweep_step:
            _create_sweep_list(args)

    return args


def _parse_sweep_range(sweep_range: str) -> Tuple[int, int, Optional[int]]:
    sweep_range_list = sweep_range.split(":")

    if len(sweep_range_list) == 2:
        return (int(sweep_range_list[0]), int(sweep_range_list[1]), None)
    elif len(sweep_range_list) == 3:
        return tuple(int(x) for x in sweep_range_list)  # type: ignore
    else:
        raise argparse.ArgumentTypeError(
            "The format of '--sweep-range' must be min:max or min:max:step."
        )


def _check_sweep_range(args):
    if not isinstance(args.sweep_min, int):
        raise argparse.ArgumentTypeError("--sweep-range min value must be an int.")
    if not isinstance(args.sweep_max, int):
        raise argparse.ArgumentTypeError("--sweep-range max value must be an int.")
    if args.sweep_step and not isinstance(args.sweep_step, int):
        raise argparse.ArgumentTypeError("--sweep-range step value must be an int.")

    if args.sweep_min < 1 or args.sweep_max < 1:
        raise argparse.ArgumentTypeError("--sweep-range min/max value must be positive")
    if args.sweep_step and args.sweep_step < 1:
        raise argparse.ArgumentTypeError("--sweep-range step value must be positive")
    if args.sweep_min >= args.sweep_max:
        raise argparse.ArgumentTypeError("--sweep-range max must be greater than min")

    if args.sweep_type == "concurrency":
        if not utils.is_power_of_two(args.sweep_min) or not utils.is_power_of_two(
            args.sweep_max
        ):
            raise argparse.ArgumentTypeError(
                "--sweep-range min/max values must be powers of 2 when sweeping concurrency"
            )
        if args.sweep_step:
            raise argparse.ArgumentTypeError(
                "--sweep-range step is not supported when sweeping concurrency"
            )


def _parse_sweep_list(args):
    args.sweep_list = [int(entry) for entry in args.sweep_list.split(",")]
    _check_sweep_list(args)


def _check_sweep_list(args):
    for entry in args.sweep_list:
        if entry < 0:
            raise argparse.ArgumentTypeError("Values in -sweep-list must be positive")


def _create_sweep_list(args):
    args.sweep_list = [
        value for value in range(args.sweep_min, args.sweep_max + 1, args.sweep_step)
    ]


def _print_warnings(config: ConfigCommand) -> None:
    if config.tokenizer.trust_remote_code:
        logger.warning(
            "--tokenizer-trust-remote-code is enabled. "
            "Custom tokenizer code can be executed. "
            "This should only be used with repositories you trust."
        )
    if (
        config.input.prompt_source == ic.PromptSource.PAYLOAD
        and config.input.output_tokens.mean != ic.DEFAULT_OUTPUT_TOKENS_MEAN
    ):
        logger.warning(
            "--output-tokens-mean is incompatible with output_length"
            " in the payload input file. output-tokens-mean"
            " will be ignored in favour of per payload settings."
        )


### Types ###


def directory(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        return path
    raise ValueError(f"'{value}' is not a valid directory")


def file_or_directory(value: str) -> Path:
    if value.startswith("synthetic:") or value.startswith("payload"):
        return Path(value)
    else:
        path = Path(value)
        if path.is_file() or path.is_dir():
            return path

    raise ValueError(f"'{value}' is not a valid file or directory")


def positive_integer(value: str) -> int:
    try:
        int_value = int(value)
        if int_value <= 0:
            raise argparse.ArgumentTypeError("The value must be greater than zero.")
    except ValueError:
        raise argparse.ArgumentTypeError("The value must be an integer.")
    return int_value


### Parsers ###


def _add_analyze_args(parser):
    analyze_group = parser.add_argument_group("Analyze")

    analyze_group.add_argument(
        "--sweep-type",
        type=str,
        default=AnalyzeDefaults.STIMULUS_TYPE,
        choices=[
            "batch_size",
            "concurrency",
            "num_dataset_entries",
            "input_sequence_length",
            "request_rate",
        ],
        required=False,
        help=f"The stimulus type that GAP will sweep.",
    )
    analyze_group.add_argument(
        "--sweep-range",
        type=str,
        default=f"{AnalyzeDefaults.MIN_CONCURRENCY}:{AnalyzeDefaults.MAX_CONCURRENCY}",
        required=False,
        help=f"The range the stimulus will be swept. Represented as 'min:max' or 'min:max:step'.",
    )
    analyze_group.add_argument(
        "--sweep-list",
        type=str,
        default=None,
        required=False,
        help=f"A comma-separated list of values that stimulus will be swept over.",
    )


def _add_audio_input_args(parser):
    input_group = parser.add_argument_group("Audio Input")

    input_group.add_argument(
        "--audio-length-mean",
        type=float,
        default=ic.DEFAULT_AUDIO_LENGTH_MEAN,
        required=False,
        help=f"The mean length of audio data in seconds. Default is 10 seconds.",
    )

    input_group.add_argument(
        "--audio-length-stddev",
        type=float,
        default=ic.DEFAULT_AUDIO_LENGTH_STDDEV,
        required=False,
        help=f"The standard deviation of the length of audio data in seconds. "
        "Default is 0.",
    )

    input_group.add_argument(
        "--audio-format",
        type=str,
        choices=utils.get_enum_names(ic.AudioFormat),
        default=ic.DEFAULT_AUDIO_FORMAT,
        required=False,
        help=f"The format of the audio data. Currently we support wav and "
        "mp3 format. Default is 'wav'.",
    )

    input_group.add_argument(
        "--audio-depths",
        type=int,
        default=ic.DEFAULT_AUDIO_DEPTHS,
        nargs="*",
        required=False,
        help=f"A list of audio bit depths to randomly select from in bits. "
        "Default is [16].",
    )

    input_group.add_argument(
        "--audio-sample-rates",
        type=float,
        default=ic.DEFAULT_AUDIO_SAMPLE_RATES,
        nargs="*",
        required=False,
        help=f"A list of audio sample rates to randomly select from in kHz. "
        "Default is [16].",
    )

    input_group.add_argument(
        "--audio-num-channels",
        type=int,
        default=ic.DEFAULT_AUDIO_NUM_CHANNELS,
        choices=[1, 2],
        required=False,
        help=f"The number of audio channels to use for the audio data generation. "
        "Currently only 1 (mono) and 2 (stereo) are supported. "
        "Default is 1 (mono channel).",
    )


def _add_compare_args(parser):
    compare_group = parser.add_argument_group("Input")
    mx_group = compare_group.add_mutually_exclusive_group(required=False)
    mx_group.add_argument(
        "--config",
        type=Path,
        default=None,
        help="The path to the YAML file that specifies plot configurations for "
        "comparing multiple runs.",
    )
    mx_group.add_argument(
        "-f",
        "--files",
        nargs="+",
        default=[],
        help="List of paths to the profile export JSON files. Users can specify "
        "this option instead of the `--config` option if they would like "
        "GenAI-Perf to generate default plots as well as initial YAML config file.",
    )


def _add_endpoint_args(parser):
    endpoint_group = parser.add_argument_group("Endpoint")

    endpoint_group.add_argument(
        "-m",
        "--model",
        nargs="+",
        required=True,
        help=f"The name of the model(s) to benchmark.",
    )
    endpoint_group.add_argument(
        "--model-selection-strategy",
        type=str,
        choices=utils.get_enum_names(ic.ModelSelectionStrategy),
        default="round_robin",
        required=False,
        help=f"When multiple model are specified, this is how a specific model "
        "should be assigned to a prompt.  round_robin means that ith prompt in the "
        "list gets assigned to i mod len(models).  random means that assignment is "
        "uniformly random",
    )

    endpoint_group.add_argument(
        "--backend",
        type=str,
        choices=utils.get_enum_names(ic.OutputFormat)[0:2],
        default=ic.DEFAULT_BACKEND,
        required=False,
        help=f'When using the "triton" service-kind, '
        "this is the backend of the model. "
        "For the TENSORRT-LLM backend, you currently must set "
        "'exclude_input_in_output' to true in the model config to "
        "not echo the input tokens in the output.",
    )

    endpoint_group.add_argument(
        "--endpoint",
        type=str,
        required=False,
        help=f"Set a custom endpoint that differs from the OpenAI defaults.",
    )

    endpoint_group.add_argument(
        "--endpoint-type",
        type=str,
        choices=list(endpoint_type_map.keys()),
        required=False,
        help=f"The endpoint-type to send requests to on the " "server.",
    )

    endpoint_group.add_argument(
        "--service-kind",
        type=str,
        choices=["dynamic_grpc", "openai", "tensorrtllm_engine", "triton"],
        default="triton",
        required=False,
        help="The kind of service perf_analyzer will "
        'generate load for. In order to use "openai", '
        "you must specify an api via --endpoint-type.",
    )

    endpoint_group.add_argument(
        "--server-metrics-url",
        "--server-metrics-urls",
        type=str,
        nargs="+",
        default=[],
        required=False,
        help="The list of Triton server metrics URLs. These are used for "
        "Telemetry metric reporting with the Triton service-kind. Example "
        "usage: --server-metrics-url http://server1:8002/metrics "
        "http://server2:8002/metrics",
    )

    endpoint_group.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        help=f"An option to enable the use of the streaming API.",
    )

    endpoint_group.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        dest="u",
        metavar="URL",
        help="URL of the endpoint to target for benchmarking.",
    )


def _add_image_input_args(parser):
    input_group = parser.add_argument_group("Image Input")

    input_group.add_argument(
        "--image-width-mean",
        type=int,
        default=ic.DEFAULT_IMAGE_WIDTH_MEAN,
        required=False,
        help=f"The mean width of images when generating synthetic image data.",
    )

    input_group.add_argument(
        "--image-width-stddev",
        type=int,
        default=ic.DEFAULT_IMAGE_WIDTH_STDDEV,
        required=False,
        help=f"The standard deviation of width of images when generating synthetic image data.",
    )

    input_group.add_argument(
        "--image-height-mean",
        type=int,
        default=ic.DEFAULT_IMAGE_HEIGHT_MEAN,
        required=False,
        help=f"The mean height of images when generating synthetic image data.",
    )

    input_group.add_argument(
        "--image-height-stddev",
        type=int,
        default=ic.DEFAULT_IMAGE_HEIGHT_STDDEV,
        required=False,
        help=f"The standard deviation of height of images when generating synthetic image data.",
    )

    input_group.add_argument(
        "--image-format",
        type=str,
        choices=utils.get_enum_names(ic.ImageFormat),
        required=False,
        help=f"The compression format of the images. "
        "If format is not selected, format of generated image is selected at random",
    )


def _add_input_args(parser):
    input_group = parser.add_argument_group("Input")

    input_group.add_argument(
        "--batch-size-audio",
        type=int,
        default=ic.DEFAULT_BATCH_SIZE,
        required=False,
        help=f"The audio batch size of the requests GenAI-Perf should send. "
        "This is currently supported with the OpenAI `multimodal` endpoint type.",
    )

    input_group.add_argument(
        "--batch-size-image",
        type=int,
        default=ic.DEFAULT_BATCH_SIZE,
        required=False,
        help=f"The image batch size of the requests GenAI-Perf should send. "
        "This is currently supported with the image retrieval endpoint type.",
    )

    input_group.add_argument(
        "--batch-size-text",
        "--batch-size",
        "-b",
        type=int,
        default=ic.DEFAULT_BATCH_SIZE,
        required=False,
        help=f"The text batch size of the requests GenAI-Perf should send. "
        "This is currently supported with the embeddings and rankings "
        "endpoint types.",
    )

    input_group.add_argument(
        "--extra-inputs",
        action="append",
        help="Provide additional inputs to include with every request. "
        "You can repeat this flag for multiple inputs. Inputs should be in an 'input_name:value' format. "
        "Alternatively, a string representing a json formatted dict can be provided.",
    )

    input_group.add_argument(
        "--goodput",
        "-g",
        nargs="+",
        required=False,
        help="An option to provide constraints in order to compute goodput. "
        "Specify goodput constraints as 'key:value' pairs, where the key is a "
        "valid metric name, and the value is a number representing "
        "either milliseconds or a throughput value per second. For example, "
        "'request_latency:300' or 'output_token_throughput_per_request:600'. "
        "Multiple key:value pairs can be provided, separated by spaces. ",
    )

    input_group.add_argument(
        "--header",
        "-H",
        action="append",
        help="Add a custom header to the requests. "
        "Headers must be specified as 'Header:Value'. "
        "You can repeat this flag for multiple headers.",
    )

    input_group.add_argument(
        "--input-file",
        type=file_or_directory,
        default=None,
        required=False,
        help="The input file or directory containing the content to use for "
        "profiling. Each line should be a JSON object with a 'text' or "
        "'image' field in JSONL format. Example: {\"text\": "
        '"Your prompt here"}. To use synthetic files for a converter that '
        "needs multiple files, prefix the path with 'synthetic:', followed "
        "by a comma-separated list of filenames. The synthetic filenames "
        "should not have extensions. For example, "
        "'synthetic:queries,passages'. For payload data, prefix the path with 'payload:', "
        "followed by a JSON string representing a payload object. The payload should "
        "contain a 'timestamp' field "
        "and you can optionally add 'input_length', 'output_length','text_input', 'session_id', 'hash_ids', and 'priority'. "
        'Example: \'payload:{"timestamp": 123.45, "input_length": 10, "output_length": 12, '
        '"session_id": 1, "priority": 5, "text_input": "Your prompt here"}\'.',
    )

    input_group.add_argument(
        "--num-dataset-entries",
        "--num-prompts",
        type=positive_integer,
        default=ic.DEFAULT_NUM_DATASET_ENTRIES,
        required=False,
        help=f"The number of unique payloads to sample from. "
        "These will be reused until benchmarking is complete.",
    )

    input_group.add_argument(
        "--num-prefix-prompts",
        type=int,
        default=ic.DEFAULT_NUM_PREFIX_PROMPTS,
        required=False,
        help=f"The number of prefix prompts to select from. "
        "If this value is not zero, these are prompts that are "
        "prepended to input prompts. This is useful for "
        "benchmarking models that use a K-V cache.",
    )

    input_group.add_argument(
        "--output-tokens-mean",
        "--osl",
        type=int,
        required=False,
        help=f"The mean number of tokens in each output. "
        "Ensure the --tokenizer value is set correctly. ",
    )

    input_group.add_argument(
        "--output-tokens-mean-deterministic",
        action="store_true",
        required=False,
        help=f"When using --output-tokens-mean, this flag can be set to "
        "improve precision by setting the minimum number of tokens "
        "equal to the requested number of tokens. This is currently "
        "supported with the Triton service-kind. "
        "Note that there is still some variability in the requested number "
        "of output tokens, but GenAi-Perf attempts its best effort with your "
        "model to get the right number of output tokens. ",
    )

    input_group.add_argument(
        "--output-tokens-stddev",
        type=int,
        default=ic.DEFAULT_OUTPUT_TOKENS_STDDEV,
        required=False,
        help=f"The standard deviation of the number of tokens in each output. "
        "This is only used when --output-tokens-mean is provided.",
    )

    input_group.add_argument(
        "--random-seed",
        type=int,
        required=False,
        help="The seed used to generate random values. If not provided, a "
        "random seed will be used.",
    )

    input_group.add_argument(
        "--grpc-method",
        type=str,
        required=False,
        help="A fully-qualified gRPC method name in "
        "'<package>.<service>/<method>' format. The option is only "
        "supported by dynamic gRPC service kind and is required to identify "
        "the RPC to use when sending requests to the server.",
    )

    input_group.add_argument(
        "--synthetic-input-tokens-mean",
        "--isl",
        type=int,
        default=ic.DEFAULT_PROMPT_TOKENS_MEAN,
        required=False,
        help=f"The mean of number of tokens in the generated prompts when using synthetic data.",
    )

    input_group.add_argument(
        "--synthetic-input-tokens-stddev",
        type=int,
        default=ic.DEFAULT_PROMPT_TOKENS_STDDEV,
        required=False,
        help=f"The standard deviation of number of tokens in the generated prompts when using synthetic data.",
    )

    input_group.add_argument(
        "--prefix-prompt-length",
        type=int,
        default=ic.DEFAULT_PREFIX_PROMPT_LENGTH,
        required=False,
        help=f"The number of tokens in each prefix prompt. This value is only "
        "used if --num-prefix-prompts is positive. Note that due to "
        "the prefix and user prompts being concatenated, the number of tokens "
        "in the final prompt may be off by one.",
    )

    input_group.add_argument(
        "--warmup-request-count",
        "--num-warmup-requests",
        type=int,
        default=ic.DEFAULT_WARMUP_REQUEST_COUNT,
        required=False,
        help=f"The number of warmup requests to send before benchmarking.",
    )


def _add_other_args(parser):
    other_group = parser.add_argument_group("Other")

    other_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="An option to enable verbose mode.",
    )


def _add_output_args(parser):
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(DEFAULT_ARTIFACT_DIR),
        help="The directory to store all the (output) artifacts generated by "
        "GenAI-Perf and Perf Analyzer.",
    )
    output_group.add_argument(
        "--generate-plots",
        action="store_true",
        required=False,
        help="An option to enable the generation of plots.",
    )
    output_group.add_argument(
        "--profile-export-file",
        type=Path,
        default=Path(DEFAULT_PROFILE_EXPORT_FILE),
        help="The path where the perf_analyzer profile export will be "
        "generated. By default, the profile export will be to "
        "profile_export.json. The genai-perf file will be exported to "
        "<profile_export_file>_genai_perf.csv. For example, if the profile "
        "export file is profile_export.json, the genai-perf file will be "
        "exported to profile_export_genai_perf.csv.",
    )


def _add_process_export_files_args(parser):
    process_export_files_group = parser.add_argument_group("Process Export Files")
    process_export_files_group.add_argument(
        "input_path",
        nargs=1,
        type=directory,
        help="The path to the directory containing the profile export files.",
    )


def _add_profile_args(parser):
    profile_group = parser.add_argument_group("Profiling")
    load_management_group = profile_group.add_mutually_exclusive_group(required=False)
    measurement_group = profile_group.add_mutually_exclusive_group(required=False)

    load_management_group.add_argument(
        "--concurrency",
        type=int,
        required=False,
        help="The concurrency value to benchmark.",
    )

    measurement_group.add_argument(
        "--measurement-interval",
        "-p",
        type=int,
        default=ic.DEFAULT_MEASUREMENT_INTERVAL,
        required=False,
        help="The time interval used for each measurement in milliseconds. "
        "Perf Analyzer will sample a time interval specified and take "
        "measurement over the requests completed within that time interval. "
        "When using the default stability percentage, GenAI-Perf will benchmark  "
        "for 3*(measurement_interval) milliseconds.",
    )

    measurement_group.add_argument(
        "--request-count",
        "--num-requests",
        type=int,
        required=False,
        help="The number of requests to use for measurement.",
    )

    load_management_group.add_argument(
        "--request-rate",
        type=float,
        required=False,
        help="Sets the request rate for the load generated by PA.",
    )

    profile_group.add_argument(
        "-s",
        "--stability-percentage",
        type=float,
        default=999,
        required=False,
        help="The allowed variation in "
        "latency measurements when determining if a result is stable. The "
        "measurement is considered as stable if the ratio of max / min "
        "from the recent 3 measurements is within (stability percentage) "
        "in terms of both infer per second and latency.",
    )


def _add_session_args(parser):
    session_group = parser.add_argument_group("Session")

    session_load_management_group = session_group.add_mutually_exclusive_group(
        required=False
    )

    session_group.add_argument(
        "--num-sessions",
        type=int,
        default=ic.DEFAULT_NUM_SESSIONS,
        help="The number of sessions to simulate.",
    )

    session_load_management_group.add_argument(
        "--session-concurrency",
        type=int,
        required=False,
        help="The number of concurrent sessions to benchmark.",
    )

    session_group.add_argument(
        "--session-delay-ratio",
        type=float,
        default=ic.DEFAULT_SESSION_DELAY_RATIO,
        help="A ratio to scale multi-turn delays when using a payload file. "
        "For example, a value of 0.5 will halve the specified delays.",
    )

    session_group.add_argument(
        "--session-turn-delay-mean",
        type=int,
        default=ic.DEFAULT_SESSION_TURN_DELAY_MEAN_MS,
        help="The mean delay (in ms) between turns in a session.",
    )

    session_group.add_argument(
        "--session-turn-delay-stddev",
        type=int,
        default=ic.DEFAULT_SESSION_TURN_DELAY_STDDEV_MS,
        help="The standard deviation (in ms) of the delay between turns in "
        "a session.",
    )

    session_group.add_argument(
        "--session-turns-mean",
        type=int,
        default=ic.DEFAULT_SESSION_TURNS_MEAN,
        help="The mean number of turns per session.",
    )

    session_group.add_argument(
        "--session-turns-stddev",
        type=int,
        default=ic.DEFAULT_SESSION_TURNS_STDDEV,
        help="The standard deviation of the number of turns per session.",
    )


def _add_tokenizer_args(parser):
    tokenizer_group = parser.add_argument_group("Tokenizer")

    tokenizer_group.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        help="The HuggingFace tokenizer to use to interpret token metrics "
        "from prompts and responses. The value can be the name of a tokenizer "
        "or the filepath of the tokenizer. The default value is the model "
        "name.",
    )
    tokenizer_group.add_argument(
        "--tokenizer-revision",
        type=str,
        default=DEFAULT_TOKENIZER_REVISION,
        required=False,
        help="The specific model version to use. It can be a branch name, "
        "tag name, or commit ID.",
    )
    tokenizer_group.add_argument(
        "--tokenizer-trust-remote-code",
        action="store_true",
        required=False,
        help="Allow custom tokenizer to be downloaded and executed. "
        "This carries security risks and should only be used "
        "for repositories you trust. This is only necessary for custom "
        "tokenizers stored in HuggingFace Hub. ",
    )


def _parse_template_args(subparsers) -> argparse.ArgumentParser:
    template = subparsers.add_parser(
        Subcommand.TEMPLATE.value,
        description="Subcommand to generate a template YAML file for profiling.",
    )
    _add_other_args(template)
    template.set_defaults(func=template_handler)
    return template


def _parse_compare_args(subparsers) -> argparse.ArgumentParser:
    compare = subparsers.add_parser(
        Subcommand.COMPARE.value,
        description="Subcommand to generate plots that compare multiple profile runs.",
    )
    _add_compare_args(compare)
    _add_tokenizer_args(compare)
    compare.set_defaults(func=compare_handler)
    return compare


def _parse_profile_args(subparsers) -> argparse.ArgumentParser:
    profile = subparsers.add_parser(
        Subcommand.PROFILE.value,
        description="Subcommand to profile LLMs and Generative AI models.",
    )
    _add_audio_input_args(profile)
    _add_endpoint_args(profile)
    _add_image_input_args(profile)
    _add_input_args(profile)
    _add_other_args(profile)
    _add_output_args(profile)
    _add_profile_args(profile)
    _add_session_args(profile)
    _add_tokenizer_args(profile)
    profile.set_defaults(func=profile_handler)
    return profile


def _parse_analyze_args(subparsers) -> argparse.ArgumentParser:
    analyze = subparsers.add_parser(
        Subcommand.ANALYZE.value,
        description="Subcommand to analyze LLMs and Generative AI models.",
    )
    _add_analyze_args(analyze)
    _add_audio_input_args(analyze)
    _add_endpoint_args(analyze)
    _add_image_input_args(analyze)
    _add_input_args(analyze)
    _add_other_args(analyze)
    _add_output_args(analyze)
    _add_profile_args(analyze)
    _add_session_args(analyze)
    _add_tokenizer_args(analyze)

    analyze.set_defaults(func=analyze_handler)
    return analyze


def _parse_process_export_files_args(subparsers) -> argparse.ArgumentParser:
    process_export_files = subparsers.add_parser(
        Subcommand.PROCESS.value,
        description="Subcommand to process export files and aggregate the results.",
    )
    _add_process_export_files_args(process_export_files)
    _add_output_args(process_export_files)
    _add_other_args(process_export_files)
    process_export_files.set_defaults(func=process_export_files_handler)
    return process_export_files


### Parser Initialization ###


def init_parsers():
    parser = argparse.ArgumentParser(
        prog="genai-perf",
        description="CLI to profile LLMs and Generative AI models with Perf Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
        help=f"An option to print the version and exit.",
    )

    # Add subcommands
    subparsers = parser.add_subparsers(
        help="List of subparser commands.", dest="subcommand"
    )
    _ = _parse_compare_args(subparsers)
    _ = _parse_profile_args(subparsers)
    _ = _parse_analyze_args(subparsers)
    _ = _parse_template_args(subparsers)
    _ = _parse_process_export_files_args(subparsers)
    subparsers.required = False

    return parser


def get_passthrough_args_index(argv: list) -> int:
    if "--" in argv:
        passthrough_index = argv.index("--")
        logger.info(f"Detected passthrough args: {argv[passthrough_index + 1:]}")
    else:
        passthrough_index = len(argv)

    return passthrough_index


def refine_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    if args.subcommand == Subcommand.PROFILE.value:
        args = _check_goodput_args(args)
    elif args.subcommand == Subcommand.ANALYZE.value:
        args = _process_sweep_args(args)
        args = _check_goodput_args(args)
    elif args.subcommand == Subcommand.COMPARE.value:
        args = _check_compare_args(parser, args)
    elif args.subcommand == Subcommand.TEMPLATE.value:
        pass
    elif args.subcommand == Subcommand.PROCESS.value:
        pass
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")

    return args


def add_cli_options_to_config(
    config: ConfigCommand, args: argparse.Namespace
) -> ConfigCommand:
    # These can only be set via the CLI and are added here
    config.subcommand = ConfigField(
        default="profile", value=args.subcommand, required=True
    )
    config.verbose = ConfigField(default=False, value=args.verbose)

    # Analyze
    if args.subcommand == "analyze":
        config.analyze.sweep_parameters = {}
        if args.sweep_list:
            sweep_parameters = {args.sweep_type: args.sweep_list}
        else:
            sweep_parameters = {
                args.sweep_type: {"start": args.sweep_min, "stop": args.sweep_max}
            }
            if args.sweep_step:
                sweep_parameters[args.sweep_type]["step"] = args.sweep_step

        config.analyze.parse(sweep_parameters)

    # Process Export Files
    elif args.subcommand == "process-export-files":
        config.input.path = ConfigField(
            default=None, value=args.input_path[0], required=True
        )
        config.output.artifact_directory = args.artifact_dir
        config.output.profile_export_file = args.profile_export_file
        config.output.generate_plots = args.generate_plots
        return config

    # Endpoint
    config.endpoint.model_selection_strategy = ic.ModelSelectionStrategy(
        args.model_selection_strategy.upper()
    )

    if args.backend != ic.DEFAULT_BACKEND:
        config.endpoint.backend = ic.OutputFormat(args.backend.upper())

    config.endpoint.custom = args.endpoint
    config.endpoint.type = args.endpoint_type
    config.endpoint.service_kind = args.service_kind
    config.endpoint.streaming = args.streaming

    if args.server_metrics_url:
        config.endpoint.server_metrics_urls = args.server_metrics_url

    if args.u:
        config.endpoint.url = args.u

    if args.grpc_method:
        config.endpoint.grpc_method = args.grpc_method

    # Perf Analyzer
    # config.perf_analyzer.path - There is no equivalent setting in the CLI
    stimulus = _convert_args_to_stimulus(args)
    if stimulus:
        config.perf_analyzer.stimulus = stimulus

    config.perf_analyzer.stability_percentage = args.stability_percentage
    config.perf_analyzer.warmup_request_count = args.warmup_request_count

    if args.measurement_interval:
        config.perf_analyzer.measurement.mode = ic.PerfAnalyzerMeasurementMode.INTERVAL
        config.perf_analyzer.measurement.num = args.measurement_interval
    elif args.request_count:
        config.perf_analyzer.measurement.mode = (
            ic.PerfAnalyzerMeasurementMode.REQUEST_COUNT
        )
        config.perf_analyzer.measurement.num = args.request_count

    # Input
    config.input.batch_size = args.batch_size_text
    config.input.extra = get_extra_inputs_as_dict(args)
    config.input.goodput = args.goodput
    config.input.header = args.header
    config.input.file = args.input_file
    config.input.num_dataset_entries = args.num_dataset_entries
    config.input.random_seed = args.random_seed

    # Input - Audio
    config.input.audio.batch_size = args.batch_size_audio
    config.input.audio.length.mean = args.audio_length_mean
    config.input.audio.length.stddev = args.audio_length_stddev
    config.input.audio.format = ic.AudioFormat(args.audio_format.upper())
    config.input.audio.depths = args.audio_depths
    config.input.audio.sample_rates = args.audio_sample_rates
    config.input.audio.num_channels = args.audio_num_channels

    # Input - Image
    config.input.image.batch_size = args.batch_size_image
    config.input.image.width.mean = args.image_width_mean
    config.input.image.width.stddev = args.image_width_stddev
    config.input.image.height.mean = args.image_height_mean
    config.input.image.height.stddev = args.image_height_stddev

    if args.image_format:
        config.input.image.format = ic.ImageFormat(args.image_format.upper())

    # Input - Output Tokens
    if args.output_tokens_mean:
        config.input.output_tokens.mean = args.output_tokens_mean

    if args.output_tokens_mean_deterministic:
        config.input.output_tokens.deterministic = args.output_tokens_mean_deterministic

    if args.output_tokens_stddev:
        config.input.output_tokens.stddev = args.output_tokens_stddev

    # Input - Synthetic Tokens
    config.input.synthetic_tokens.mean = args.synthetic_input_tokens_mean
    config.input.synthetic_tokens.stddev = args.synthetic_input_tokens_stddev

    # Input - Prefix Prompt
    config.input.prefix_prompt.num = args.num_prefix_prompts
    config.input.prefix_prompt.length = args.prefix_prompt_length

    # Input - Sessions
    config.input.sessions.num = args.num_sessions
    config.input.sessions.turn_delay.mean = args.session_turn_delay_mean
    config.input.sessions.turn_delay.ratio = args.session_delay_ratio
    config.input.sessions.turn_delay.stddev = args.session_turn_delay_stddev
    config.input.sessions.turns.mean = args.session_turns_mean
    config.input.sessions.turns.stddev = args.session_turns_stddev

    # Output
    config.output.artifact_directory = args.artifact_dir
    # config.output.checkpoint_directory - There is no equivalent setting in the CLI
    config.output.profile_export_file = args.profile_export_file
    config.output.generate_plots = args.generate_plots

    # Tokenizer
    config.tokenizer.name = args.tokenizer
    config.tokenizer.revision = args.tokenizer_revision
    config.tokenizer.trust_remote_code = args.tokenizer_trust_remote_code

    return config


def _convert_args_to_stimulus(args: argparse.Namespace) -> Optional[Dict[str, int]]:
    if args.session_concurrency:
        return {"session_concurrency": args.session_concurrency}
    elif args.concurrency:
        return {"concurrency": args.concurrency}
    elif args.request_rate:
        return {"request_rate": args.request_rate}
    else:
        return None


### Entrypoint ###


def parse_args():
    argv = sys.argv
    parser = init_parsers()
    passthrough_index = get_passthrough_args_index(argv)

    # Assumption is that the subcommand will be implied by
    # the fields set in the config file
    if (
        subcommand_found(argv)
        or "-h" in argv
        or "--help" in argv
        or "--version" in argv
    ):
        args = parser.parse_args(argv[1:passthrough_index])
        args = refine_args(parser, args)

        if args.subcommand == Subcommand.TEMPLATE.value:
            config = _create_template_config(args, argv)
            return args, config, None
        elif args.subcommand == Subcommand.PROCESS.value:
            config = ConfigCommand({"model_name": ""})
            config = add_cli_options_to_config(config, args)
            return args, config, None
        else:
            # For all other subcommands, parse the CLI fully (no config file)
            config = ConfigCommand(
                {"model_names": args.model}, skip_inferencing_and_checking=True
            )
            config = add_cli_options_to_config(config, args)
            config.infer_and_check_options()
            _print_warnings(config)

            logger.info(f"Profiling these models: {', '.join(config.model_names)}")
            return args, config, argv[passthrough_index + 1 :]
    else:  # No subcommmand on CLI
        # Assumption is the last argument before the
        # passthrough is the user config file
        user_config = utils.load_yaml(argv[passthrough_index - 1])
        config = ConfigCommand(user_config)
        _print_warnings(config)

        # Set subcommand
        if config.analyze.get_field("sweep_parameters").is_set_by_user:
            config_args = ["analyze"]
        else:
            config_args = ["profile"]

        # Setup args in the way that argparse expects
        config_args += ["-m", config.model_names[0]]
        config_args += argv[1 : passthrough_index - 2]

        args = parser.parse_args(config_args)

        config.subcommand = ConfigField(
            default="profile", value=args.subcommand, required=True
        )

        # Set verbose
        config.verbose = ConfigField(default=False, value=args.verbose)

        logger.info(f"Profiling these models: {', '.join(config.model_names)}")
        return args, config, argv[passthrough_index + 1 :]


def _create_template_config(args: argparse.Namespace, argv: List[str]) -> ConfigCommand:
    config = ConfigCommand({"model_name": ""}, skip_inferencing_and_checking=True)

    config.verbose = ConfigField(
        default=False, value=args.verbose, add_to_template=False
    )
    config.subcommand = ConfigField(
        default="profile",
        value=args.subcommand,
        required=True,
        add_to_template=False,
    )

    # The template command is: genai-perf template [filename] [-v/--verbose]
    if "-v" in argv:
        del argv[argv.index("-v")]
    if "--verbose" in argv:
        del argv[argv.index("--verbose")]

    # Assumption is the final argument is the filename (if it exists)
    filename = argv[2] if len(argv) > 2 else TemplateDefaults.FILENAME
    config.template_filename = ConfigField(
        default=TemplateDefaults.FILENAME, value=filename, add_to_template=False
    )

    return config


def subcommand_found(argv) -> bool:
    for sc in Subcommand:
        if sc.value in argv:
            return True

    return False
