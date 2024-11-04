# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import sys
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import genai_perf.logging as logging
import genai_perf.utils as utils
from genai_perf.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_COMPARE_DIR
from genai_perf.inputs import input_constants as ic
from genai_perf.inputs.retrievers.synthetic_image_generator import ImageFormat
from genai_perf.plots.plot_config_parser import PlotConfigParser
from genai_perf.plots.plot_manager import PlotManager
from genai_perf.telemetry_data import TelemetryDataCollector
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

    def to_lowercase(self):
        return self.name.lower()


logger = logging.getLogger(__name__)

_endpoint_type_map = {
    "chat": "v1/chat/completions",
    "completions": "v1/completions",
    "embeddings": "v1/embeddings",
    "image_retrieval": "v1/infer",
    "nvclip": "v1/embeddings",
    "rankings": "v1/ranking",
    "vision": "v1/chat/completions",
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
        else:
            if args.endpoint_type == "chat":
                args.output_format = ic.OutputFormat.OPENAI_CHAT_COMPLETIONS
            elif args.endpoint_type == "completions":
                args.output_format = ic.OutputFormat.OPENAI_COMPLETIONS
            elif args.endpoint_type == "embeddings":
                args.output_format = ic.OutputFormat.OPENAI_EMBEDDINGS
            elif args.endpoint_type == "rankings":
                args.output_format = ic.OutputFormat.RANKINGS
            elif args.endpoint_type == "image_retrieval":
                args.output_format = ic.OutputFormat.IMAGE_RETRIEVAL

            # (TMA-1986) deduce vision format from chat completions + image CLI
            # because there's no openai vision endpoint.
            elif args.endpoint_type == "vision":
                args.output_format = ic.OutputFormat.OPENAI_VISION
            elif args.endpoint_type == "nvclip":
                args.output_format = ic.OutputFormat.NVCLIP

            if args.endpoint is not None:
                args.endpoint = args.endpoint.lstrip(" /")
            else:
                args.endpoint = _endpoint_type_map[args.endpoint_type]
    elif args.endpoint_type is not None:
        parser.error(
            "The --endpoint-type option should only be used when using the 'openai' service-kind."
        )

    if args.service_kind == "triton":
        args = _convert_str_to_enum_entry(args, "backend", ic.OutputFormat)
        args.output_format = args.backend

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


def _is_valid_url(parser: argparse.ArgumentParser, url: str) -> None:
    """
    Validates a URL to ensure it meets the following criteria:
    - The scheme must be 'http' or 'https'.
    - The netloc (domain) must be present.
    - The path must contain '/metrics'.

    Raises:
        `parser.error()` if the URL is invalid.

    The URL structure is expected to follow:
    <scheme>://<netloc>/<path>;<params>?<query>#<fragment>
    """
    parsed_url = urlparse(url)

    if (
        parsed_url.scheme not in ["http", "https"]
        or not parsed_url.netloc
        or "/metrics" not in parsed_url.path
        or parsed_url.port is None
    ):
        parser.error(
            "The URL passed for --server-metrics-url is invalid. "
            "It must use 'http' or 'https', have a valid domain and port, "
            "and contain '/metrics' in the path. The expected structure is: "
            "<scheme>://<netloc>/<path>;<params>?<query>#<fragment>"
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
    Checks if the server metrics URL passed is valid
    """

    # Check if the URL is valid and contains the expected path
    if args.service_kind == "triton" and args.server_metrics_url:
        _is_valid_url(parser, args.server_metrics_url)

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
        "You can repeat this flag for multiple inputs. Inputs should be in an input_name:value format. "
        "Alternatively, a string representing a json formatted dict can be provided.",
    )

    input_group.add_argument(
        "--input-file",
        type=file_or_directory,
        default=None,
        required=False,
        help="The input file or directory containing the content to use for "
        "profiling. To use synthetic files for a converter that needs "
        "multiple files, prefix the path with 'synthetic:', followed by a "
        "comma-separated list of filenames. The synthetic filenames should "
        "not have extensions. For example, 'synthetic:queries,passages'. "
        "Each line should be a JSON object with a 'text' or 'image' field "
        'in JSONL format. Example: {"text": "Your prompt here"}',
    )

    input_group.add_argument(
        "--num-prompts",
        type=positive_integer,
        default=ic.DEFAULT_NUM_PROMPTS,
        required=False,
        help=f"The number of unique prompts to generate as stimulus.",
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
        default="tensorrtllm",
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
        choices=[
            "chat",
            "completions",
            "embeddings",
            "nvclip",
            "image_retrieval",
            "rankings",
            "vision",
        ],
        required=False,
        help=f"The endpoint-type to send requests to on the "
        'server. This is only used with the "openai" service-kind.',
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
        type=str,
        default=None,
        required=False,
        help="The full URL to access the server metrics endpoint. "
        "This argument is required if the metrics are available on "
        "a different machine than localhost (where GenAI-Perf is running).",
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
        default=Path("profile_export.json"),
        help="The path where the perf_analyzer profile export will be "
        "generated. By default, the profile export will be to "
        "profile_export.json. The genai-perf file will be exported to "
        "<profile_export_file>_genai_perf.csv. For example, if the profile "
        "export file is profile_export.json, the genai-perf file will be "
        "exported to profile_export_genai_perf.csv.",
    )


def _add_other_args(parser):
    other_group = parser.add_argument_group("Other")

    other_group.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        required=False,
        help="The HuggingFace tokenizer to use to interpret token metrics "
        "from prompts and responses. The value can be the name of a tokenizer "
        "or the filepath of the tokenizer.",
    )
    other_group.add_argument(
        "--tokenizer-revision",
        type=str,
        default=DEFAULT_TOKENIZER_REVISION,
        required=False,
        help="The specific model version to use. It can be a branch name, "
        "tag name, or commit ID.",
    )
    other_group.add_argument(
        "--tokenizer-trust-remote-code",
        action="store_true",
        required=False,
        help="Allow custom tokenizer to be downloaded and executed. "
        "This carries security risks and should only be used "
        "for repositories you trust. This is only necessary for custom "
        "tokenizers stored in HuggingFace Hub. ",
    )

    other_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="An option to enable verbose mode.",
    )


def get_extra_inputs_as_dict(args: argparse.Namespace) -> dict:
    request_inputs = {}
    if args.extra_inputs:
        for input_str in args.extra_inputs:
            if input_str.startswith("{") and input_str.endswith("}"):
                request_inputs.update(utils.load_json_str(input_str))
            else:
                semicolon_count = input_str.count(":")
                if semicolon_count != 1:
                    raise ValueError(
                        f"Invalid input format for --extra-inputs: {input_str}\n"
                        "Expected input format: 'input_name:value'"
                    )
                input_name, value = input_str.split(":", 1)

                if not input_name or not value:
                    raise ValueError(
                        f"Input name or value is empty in --extra-inputs: {input_str}\n"
                        "Expected input format: 'input_name:value'"
                    )

                is_bool = value.lower() in ["true", "false"]
                is_int = value.isdigit()
                is_float = value.count(".") == 1 and (
                    value[0] == "." or value.replace(".", "").isdigit()
                )

                if is_bool:
                    value = value.lower() == "true"
                elif is_int:
                    value = int(value)
                elif is_float:
                    value = float(value)

                if input_name in request_inputs:
                    raise ValueError(
                        f"Input name already exists in request_inputs dictionary: {input_name}"
                    )
                request_inputs[input_name] = value

    return request_inputs


def _parse_compare_args(subparsers) -> argparse.ArgumentParser:
    compare = subparsers.add_parser(
        Subcommand.COMPARE.to_lowercase(),
        description="Subcommand to generate plots that compare multiple profile runs.",
    )
    compare_group = compare.add_argument_group("Input")
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
    compare.set_defaults(func=compare_handler)
    return compare


def _parse_profile_args(subparsers) -> argparse.ArgumentParser:
    profile = subparsers.add_parser(
        Subcommand.PROFILE.to_lowercase(),
        description="Subcommand to profile LLMs and Generative AI models.",
    )
    _add_endpoint_args(profile)
    _add_input_args(profile)
    _add_image_input_args(profile)
    _add_profile_args(profile)
    _add_output_args(profile)
    _add_other_args(profile)
    profile.set_defaults(func=profile_handler)
    return profile


### Handlers ###


def create_compare_dir() -> None:
    if not os.path.exists(DEFAULT_COMPARE_DIR):
        os.mkdir(DEFAULT_COMPARE_DIR)


def compare_handler(args: argparse.Namespace):
    """Handles `compare` subcommand workflow."""
    if args.files:
        create_compare_dir()
        output_dir = Path(f"{DEFAULT_COMPARE_DIR}")
        PlotConfigParser.create_init_yaml_config(args.files, output_dir)
        args.config = output_dir / "config.yaml"

    config_parser = PlotConfigParser(args.config)
    plot_configs = config_parser.generate_configs()
    plot_manager = PlotManager(plot_configs)
    plot_manager.generate_plots()


def profile_handler(
    args, extra_args, telemetry_data_collector: Optional[TelemetryDataCollector]
) -> None:
    from genai_perf.wrapper import Profiler

    Profiler.run(
        args=args,
        extra_args=extra_args,
        telemetry_data_collector=telemetry_data_collector,
    )


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
