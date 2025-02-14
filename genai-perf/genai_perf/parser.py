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
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import genai_perf.logging as logging
import genai_perf.utils as utils
from genai_perf.config.input.config_command import RunConfigDefaults
from genai_perf.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_PROFILE_EXPORT_FILE
from genai_perf.inputs import input_constants as ic
from genai_perf.inputs.retrievers.synthetic_image_generator import ImageFormat
from genai_perf.subcommand.analyze import analyze_handler
from genai_perf.subcommand.compare import compare_handler
from genai_perf.subcommand.profile import profile_handler
from genai_perf.tokenizer import DEFAULT_TOKENIZER, DEFAULT_TOKENIZER_REVISION

from . import __version__


class PathType(Enum):
    FILE = auto()
    DIRECTORY = auto()

    def to_lowercase(self):
        return self.name.lower()


class Subcommand(Enum):
    PROFILE = auto()
    COMPARE = auto()
    ANALYZE = auto()

    def to_lowercase(self):
        return self.name.lower()


logger = logging.getLogger(__name__)


@dataclass
class EndpointConfig:
    endpoint: Optional[str]
    service_kind: str
    output_format: ic.OutputFormat


_endpoint_type_map = {
    "chat": EndpointConfig(
        "v1/chat/completions", "openai", ic.OutputFormat.OPENAI_CHAT_COMPLETIONS
    ),
    "completions": EndpointConfig(
        "v1/completions", "openai", ic.OutputFormat.OPENAI_COMPLETIONS
    ),
    "embeddings": EndpointConfig(
        "v1/embeddings", "openai", ic.OutputFormat.OPENAI_EMBEDDINGS
    ),
    "image_retrieval": EndpointConfig(
        "v1/infer", "openai", ic.OutputFormat.IMAGE_RETRIEVAL
    ),
    "nvclip": EndpointConfig("v1/embeddings", "openai", ic.OutputFormat.NVCLIP),
    "rankings": EndpointConfig("v1/ranking", "openai", ic.OutputFormat.RANKINGS),
    "vision": EndpointConfig(
        "v1/chat/completions", "openai", ic.OutputFormat.OPENAI_VISION
    ),
    "generate": EndpointConfig(
        "v2/models/{MODEL_NAME}/generate", "triton", ic.OutputFormat.TRITON_GENERATE
    ),
    "kserve": EndpointConfig(None, "triton", ic.OutputFormat.TENSORRTLLM),
    "template": EndpointConfig(None, "triton", ic.OutputFormat.TEMPLATE),
    "tensorrtllm_engine": EndpointConfig(
        None, "tensorrtllm_engine", ic.OutputFormat.TENSORRTLLM_ENGINE
    ),
}


def _check_model_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """
    Check arguments associated with the model and apply any necessary formatting.
    """
    logger.info(f"Profiling these models: {', '.join(args.model)}")
    args = _convert_str_to_enum_entry(
        args, "model_selection_strategy", ic.ModelSelectionStrategy
    )
    _generate_formatted_model_name(args)
    return args


def _generate_formatted_model_name(args: argparse.Namespace) -> None:
    if len(args.model) == 1:
        args.formatted_model_name = args.model[0]
    elif len(args.model) == 0:
        args.model = None
        args.formatted_model_name = None
    else:
        args.formatted_model_name = args.model[0] + "_multi"


def _check_compare_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """
    Check compare subcommand args
    """
    if not args.config and not args.files:
        parser.error("Either the --config or --files option must be specified.")
    return args


def _check_image_input_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """
    Sanity check the image input args
    """
    if args.image_width_mean <= 0 or args.image_height_mean <= 0:
        parser.error(
            "Both --image-width-mean and --image-height-mean values must be positive."
        )
    if args.image_width_stddev < 0 or args.image_height_stddev < 0:
        parser.error(
            "Both --image-width-stddev and --image-height-stddev values must be non-negative."
        )

    args = _convert_str_to_enum_entry(args, "image_format", ImageFormat)
    return args


def _check_conditional_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """
    Check for conditional args and raise an error if they are not set.
    """

    # Endpoint and output format checks
    if args.service_kind == "openai":
        if args.endpoint_type is None:
            parser.error(
                "The --endpoint-type option is required when using the 'openai' service-kind."
            )

    if args.service_kind == "triton" and args.endpoint_type is None:
        args.endpoint_type = "kserve"

    if args.service_kind == "tensorrtllm_engine" and args.endpoint_type is None:
        args.endpoint_type = "tensorrtllm_engine"

    if args.endpoint_type and args.endpoint_type not in _endpoint_type_map:
        parser.error(f"Invalid endpoint type {args.endpoint_type}")

    endpoint_config = _endpoint_type_map[args.endpoint_type]
    args.output_format = endpoint_config.output_format

    if endpoint_config.service_kind != args.service_kind:
        parser.error(
            f"Invalid endpoint-type '{args.endpoint_type}' for service-kind '{args.service_kind}'."
        )

    if args.endpoint is not None:
        args.endpoint = args.endpoint.lstrip(" /")
    else:
        if args.model:
            model_name = args.model[0]
        else:
            model_name = ""
        if endpoint_config.endpoint:
            args.endpoint = endpoint_config.endpoint.format(MODEL_NAME=model_name)

    if args.service_kind == "triton" and args.endpoint_type in ["kserve", "template"]:
        args = _convert_str_to_enum_entry(args, "backend", ic.OutputFormat)
        if args.endpoint_type == "kserve":
            args.output_format = args.backend
    else:
        if args.backend is not ic.DEFAULT_BACKEND:
            parser.error(
                "The --backend option should only be used when using the 'triton' service-kind and 'kserve' endpoint-type."
            )

    if args.service_kind == "triton" and args.endpoint_type == "generate":
        # TODO: infer service_kind from endpoint_type and deprecate service_kind argument
        args.service_kind = "openai"

    if args.service_kind == "tensorrtllm_engine":
        args.output_format = ic.OutputFormat.TENSORRTLLM_ENGINE

    # Output token distribution checks
    if args.output_tokens_mean == ic.DEFAULT_OUTPUT_TOKENS_MEAN:
        if args.output_tokens_stddev != ic.DEFAULT_OUTPUT_TOKENS_STDDEV:
            parser.error(
                "The --output-tokens-mean option is required when using --output-tokens-stddev."
            )
        if args.output_tokens_mean_deterministic:
            parser.error(
                "The --output-tokens-mean option is required when using --output-tokens-mean-deterministic."
            )

    if args.service_kind not in ["triton", "tensorrtllm_engine"]:
        if args.output_tokens_mean_deterministic:
            parser.error(
                "The --output-tokens-mean-deterministic option is only supported "
                "with the Triton and TensorRT-LLM Engine service-kind."
            )

    if args.output_format in [
        ic.OutputFormat.IMAGE_RETRIEVAL,
        ic.OutputFormat.NVCLIP,
        ic.OutputFormat.OPENAI_EMBEDDINGS,
        ic.OutputFormat.RANKINGS,
        ic.OutputFormat.TEMPLATE,
    ]:
        if args.generate_plots:
            parser.error(
                f"The --generate-plots option is not currently supported with the {args.endpoint_type} endpoint type."
            )

    return args


def _check_load_manager_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Check inference load args
    """
    # If no concurrency or request rate is set, default to 1
    if not args.concurrency and not args.request_rate:
        args.concurrency = 1
    return args


def _check_goodput_args(args):
    """
    Parse and check goodput args
    """
    if args.goodput:
        args.goodput = parse_goodput(args.goodput)
        for target_metric, target_val in args.goodput.items():
            if target_val < 0:
                raise ValueError(
                    f"Invalid value found, {target_metric}: {target_val}. "
                    f"The goodput constraint value should be non-negative. "
                )
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


def _is_valid_url(parser: argparse.ArgumentParser, url: str) -> None:
    """
    Validates a URL to ensure it meets the following criteria:
    - The scheme must be 'http' or 'https'.
    - The netloc (domain) must be present OR the URL must be a valid localhost
    address.
    - The path must contain '/metrics'.
    - The port must be specified.

    Raises:
        `parser.error()` if the URL is invalid.

    The URL structure is expected to follow:
    <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
    """
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ["http", "https"]:
        parser.error(
            f"Invalid scheme '{parsed_url.scheme}' in URL: {url}. Use 'http' "
            "or 'https'."
        )

    valid_localhost = parsed_url.hostname in ["localhost", "127.0.0.1"]

    if not parsed_url.netloc and not valid_localhost:
        parser.error(
            f"Invalid domain in URL: {url}. Use a valid hostname or " "'localhost'."
        )

    if "/metrics" not in parsed_url.path:
        parser.error(
            f"Invalid URL path '{parsed_url.path}' in {url}. The path must "
            "include '/metrics'."
        )

    if parsed_url.port is None:
        parser.error(
            f"Port missing in URL: {url}. A port number is required " "(e.g., ':8002')."
        )


def _print_warnings(args: argparse.Namespace) -> None:
    if args.tokenizer_trust_remote_code:
        logger.warning(
            "--tokenizer-trust-remote-code is enabled. "
            "Custom tokenizer code can be executed. "
            "This should only be used with repositories you trust."
        )


def _check_server_metrics_url(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """
    Checks if each server metrics URL passed is valid
    """

    if args.service_kind == "triton" and args.server_metrics_url:
        for url in args.server_metrics_url:
            _is_valid_url(parser, url)

    return args


def _set_artifact_paths(args: argparse.Namespace) -> argparse.Namespace:
    """
    Set paths for all the artifacts.
    """
    if args.artifact_dir == Path(DEFAULT_ARTIFACT_DIR):
        # Preprocess Huggingface model names that include '/' in their model name.
        if (args.formatted_model_name is not None) and (
            "/" in args.formatted_model_name
        ):
            filtered_name = "_".join(args.formatted_model_name.split("/"))
            logger.info(
                f"Model name '{args.formatted_model_name}' cannot be used to create artifact "
                f"directory. Instead, '{filtered_name}' will be used."
            )
            name = [f"{filtered_name}"]
        else:
            name = [f"{args.formatted_model_name}"]

        if args.service_kind == "openai":
            name += [f"{args.service_kind}-{args.endpoint_type}"]
        elif args.service_kind == "triton":
            name += [f"{args.service_kind}-{args.backend.to_lowercase()}"]
        elif args.service_kind == "tensorrtllm_engine":
            name += [f"{args.service_kind}"]
        else:
            raise ValueError(f"Unknown service kind '{args.service_kind}'.")

        if args.concurrency:
            name += [f"concurrency{args.concurrency}"]
        elif args.request_rate:
            name += [f"request_rate{args.request_rate}"]
        args.artifact_dir = args.artifact_dir / Path("-".join(name))

    if args.profile_export_file.parent != Path(""):
        raise ValueError(
            "Please use --artifact-dir option to define intermediary paths to "
            "the profile export file."
        )

    args.profile_export_file = args.artifact_dir / args.profile_export_file
    return args


def parse_goodput(values):
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


def _infer_prompt_source(args: argparse.Namespace) -> argparse.Namespace:
    args.synthetic_input_files = None

    if args.input_file:
        if str(args.input_file).startswith("synthetic:"):
            args.prompt_source = ic.PromptSource.SYNTHETIC
            synthetic_input_files_str = str(args.input_file).split(":", 1)[1]
            args.synthetic_input_files = synthetic_input_files_str.split(",")
            logger.debug(
                f"Input source is synthetic data: {args.synthetic_input_files}"
            )
        else:
            args.prompt_source = ic.PromptSource.FILE
            logger.debug(f"Input source is the following path: {args.input_file}")
    else:
        args.prompt_source = ic.PromptSource.SYNTHETIC
    return args


def _convert_str_to_enum_entry(args, option, enum):
    """
    Convert string option to corresponding enum entry
    """
    attr_val = getattr(args, option)
    if attr_val is not None:
        setattr(args, f"{option}", utils.get_enum_entry(attr_val, enum))
    return args


### Types ###


def file_or_directory(value: str) -> Path:
    if value.startswith("synthetic:"):
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
        default=RunConfigDefaults.STIMULUS_TYPE,
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
        default=f"{RunConfigDefaults.MIN_CONCURRENCY}:{RunConfigDefaults.MAX_CONCURRENCY}",
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
        choices=list(_endpoint_type_map.keys()),
        required=False,
        help=f"The endpoint-type to send requests to on the " "server.",
    )

    endpoint_group.add_argument(
        "--service-kind",
        type=str,
        choices=["triton", "openai", "tensorrtllm_engine"],
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
        choices=utils.get_enum_names(ImageFormat),
        required=False,
        help=f"The compression format of the images. "
        "If format is not selected, format of generated image is selected at random",
    )


def _add_input_args(parser):
    input_group = parser.add_argument_group("Input")

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
        "'image' field 'in JSONL format. Example: {\"text\":"
        ' "Your prompt here"}\'. To use synthetic files for a converter that '
        "needs multiple files, prefix the path with 'synthetic:', followed "
        "by a comma-separated list of filenames. The synthetic filenames "
        "should not have extensions. For example, "
        "'synthetic:queries,passages'. ",
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
        default=ic.DEFAULT_OUTPUT_TOKENS_MEAN,
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
        default=ic.DEFAULT_RANDOM_SEED,
        required=False,
        help="The seed used to generate random values.",
    )

    input_group.add_argument(
        "--request-count",
        "--num-requests",
        type=int,
        default=ic.DEFAULT_REQUEST_COUNT,
        required=False,
        help="The number of requests to use for measurement."
        "By default, the benchmark does not terminate based on request count. "
        "Instead, it continues until stabilization is detected.",
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


def _add_profile_args(parser):
    profile_group = parser.add_argument_group("Profiling")
    load_management_group = profile_group.add_mutually_exclusive_group(required=False)

    load_management_group.add_argument(
        "--concurrency",
        type=int,
        required=False,
        help="The concurrency value to benchmark.",
    )

    profile_group.add_argument(
        "--measurement-interval",
        "-p",
        type=int,
        default="10000",
        required=False,
        help="The time interval used for each measurement in milliseconds. "
        "Perf Analyzer will sample a time interval specified and take "
        "measurement over the requests completed within that time interval.",
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


def _add_tokenizer_args(parser):
    tokenizer_group = parser.add_argument_group("Tokenizer")

    tokenizer_group.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        required=False,
        help="The HuggingFace tokenizer to use to interpret token metrics "
        "from prompts and responses. The value can be the name of a tokenizer "
        "or the filepath of the tokenizer.",
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


def _parse_compare_args(subparsers) -> argparse.ArgumentParser:
    compare = subparsers.add_parser(
        Subcommand.COMPARE.to_lowercase(),
        description="Subcommand to generate plots that compare multiple profile runs.",
    )
    _add_compare_args(compare)
    _add_tokenizer_args(compare)
    compare.set_defaults(func=compare_handler)
    return compare


def _parse_profile_args(subparsers) -> argparse.ArgumentParser:
    profile = subparsers.add_parser(
        Subcommand.PROFILE.to_lowercase(),
        description="Subcommand to profile LLMs and Generative AI models.",
    )
    _add_endpoint_args(profile)
    _add_image_input_args(profile)
    _add_input_args(profile)
    _add_other_args(profile)
    _add_output_args(profile)
    _add_profile_args(profile)
    _add_tokenizer_args(profile)
    profile.set_defaults(func=profile_handler)
    return profile


def _parse_analyze_args(subparsers) -> argparse.ArgumentParser:
    analyze = subparsers.add_parser(
        Subcommand.ANALYZE.to_lowercase(),
        description="Subcommand to analyze LLMs and Generative AI models.",
    )
    _add_analyze_args(analyze)
    _add_endpoint_args(analyze)
    _add_image_input_args(analyze)
    _add_input_args(analyze)
    _add_other_args(analyze)
    _add_output_args(analyze)
    _add_profile_args(analyze)
    _add_tokenizer_args(analyze)
    analyze.set_defaults(func=analyze_handler)
    return analyze


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
    subparsers.required = True

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
    if args.subcommand == Subcommand.PROFILE.to_lowercase():
        args = _infer_prompt_source(args)
        args = _check_model_args(parser, args)
        args = _check_conditional_args(parser, args)
        args = _check_image_input_args(parser, args)
        args = _check_load_manager_args(args)
        args = _check_server_metrics_url(parser, args)
        args = _set_artifact_paths(args)
        args = _check_goodput_args(args)
        _print_warnings(args)
    elif args.subcommand == Subcommand.ANALYZE.to_lowercase():
        args = _infer_prompt_source(args)
        args = _check_model_args(parser, args)
        args = _check_conditional_args(parser, args)
        args = _check_image_input_args(parser, args)
        args = _check_server_metrics_url(parser, args)
        args = _check_goodput_args(args)
        args = _process_sweep_args(args)
        _print_warnings(args)
    elif args.subcommand == Subcommand.COMPARE.to_lowercase():
        args = _check_compare_args(parser, args)
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")

    return args


### Entrypoint ###


def parse_args():
    argv = sys.argv

    parser = init_parsers()
    passthrough_index = get_passthrough_args_index(argv)
    args = parser.parse_args(argv[1:passthrough_index])
    args = refine_args(parser, args)

    return args, argv[passthrough_index + 1 :]
