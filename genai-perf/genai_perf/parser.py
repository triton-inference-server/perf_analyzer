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
from typing import List, Optional, Tuple

import genai_perf.logging as logging
import genai_perf.utils as utils
from genai_perf.config.endpoint_config import endpoint_type_map
from genai_perf.inputs import input_constants as ic
from genai_perf.subcommand.analyze import analyze_handler
from genai_perf.subcommand.config import config_handler
from genai_perf.subcommand.process_export_files import process_export_files_handler
from genai_perf.subcommand.profile import profile_handler
from genai_perf.subcommand.template import template_handler

from . import __version__

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


### Types ###


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
        choices=[
            "batch_size",
            "concurrency",
            "num_dataset_entries",
            "input_sequence_length",
            "request_rate",
        ],
        help=f"The stimulus type that GAP will sweep.",
    )
    analyze_group.add_argument(
        "--sweep-range",
        type=str,
        help=f"The range the stimulus will be swept. Represented as 'min:max' or 'min:max:step'.",
    )
    analyze_group.add_argument(
        "--sweep-list",
        type=str,
        help=f"A comma-separated list of values that stimulus will be swept over.",
    )


def _add_audio_input_args(parser):
    input_group = parser.add_argument_group("Audio Input")

    input_group.add_argument(
        "--audio-length-mean",
        type=float,
        help=f"The mean length of audio data in seconds. Default is 10 seconds.",
    )

    input_group.add_argument(
        "--audio-length-stddev",
        type=float,
        help=f"The standard deviation of the length of audio data in seconds. "
        "Default is 0.",
    )

    input_group.add_argument(
        "--audio-format",
        type=str,
        choices=utils.get_enum_names(ic.AudioFormat),
        help=f"The format of the audio data. Currently we support wav and "
        "mp3 format. Default is 'wav'.",
    )

    input_group.add_argument(
        "--audio-depths",
        type=int,
        nargs="*",
        help=f"A list of audio bit depths to randomly select from in bits. "
        "Default is [16].",
    )

    input_group.add_argument(
        "--audio-sample-rates",
        type=float,
        nargs="*",
        help=f"A list of audio sample rates to randomly select from in kHz. "
        "Default is [16].",
    )

    input_group.add_argument(
        "--audio-num-channels",
        type=int,
        choices=[1, 2],
        help=f"The number of audio channels to use for the audio data generation. "
        "Currently only 1 (mono) and 2 (stereo) are supported. "
        "Default is 1 (mono channel).",
    )


def _add_template_args(parser):
    template_group = parser.add_argument_group("Template")

    template_group.add_argument(
        "-f",
        "--file",
        type=Path,
        help="The name to the template file that will be created.",
    )


def _add_config_args(parser):
    config_group = parser.add_argument_group("Config")

    config_group.add_argument(
        "-f",
        "--file",
        type=Path,
        required=True,
        help="The path to the config file.",
    )

    config_group.add_argument(
        "--override-config",
        action="store_true",
        help="Setting this flag enables the user to override config values via the CLI.",
    )


def _add_endpoint_args(parser):
    endpoint_group = parser.add_argument_group("Endpoint")

    endpoint_group.add_argument(
        "-m",
        "--model",
        nargs="+",
        help=f"The name of the model(s) to benchmark.",
    )
    endpoint_group.add_argument(
        "--model-selection-strategy",
        type=str,
        choices=utils.get_enum_names(ic.ModelSelectionStrategy),
        help=f"When multiple model are specified, this is how a specific model "
        "should be assigned to a prompt.  round_robin means that ith prompt in the "
        "list gets assigned to i mod len(models).  random means that assignment is "
        "uniformly random",
    )

    endpoint_group.add_argument(
        "--backend",
        type=str,
        choices=utils.get_enum_names(ic.OutputFormat)[0:2],
        help=f"When benchmarking Triton, this is the backend of the model. ",
    )

    endpoint_group.add_argument(
        "--endpoint",
        type=str,
        help=f"Set a custom endpoint that differs from the OpenAI defaults.",
    )

    endpoint_group.add_argument(
        "--endpoint-type",
        type=str,
        choices=list(endpoint_type_map.keys()),
        help=f"The endpoint-type to send requests to on the server.",
    )

    endpoint_group.add_argument(
        "--server-metrics-url",
        "--server-metrics-urls",
        type=str,
        nargs="+",
        help="The list of Triton server metrics URLs. These are used for "
        "Telemetry metric reporting with Triton. Example "
        "usage: --server-metrics-url http://server1:8002/metrics "
        "http://server2:8002/metrics",
    )

    endpoint_group.add_argument(
        "--streaming",
        action="store_true",
        help=f"An option to enable the use of the streaming API.",
    )

    endpoint_group.add_argument(
        "-u",
        "--url",
        type=str,
        dest="u",
        metavar="URL",
        help="URL of the endpoint to target for benchmarking.",
    )


def _add_image_input_args(parser):
    input_group = parser.add_argument_group("Image Input")

    input_group.add_argument(
        "--image-width-mean",
        type=int,
        help=f"The mean width of images when generating synthetic image data.",
    )

    input_group.add_argument(
        "--image-width-stddev",
        type=int,
        help=f"The standard deviation of width of images when generating synthetic image data.",
    )

    input_group.add_argument(
        "--image-height-mean",
        type=int,
        help=f"The mean height of images when generating synthetic image data.",
    )

    input_group.add_argument(
        "--image-height-stddev",
        type=int,
        help=f"The standard deviation of height of images when generating synthetic image data.",
    )

    input_group.add_argument(
        "--image-format",
        type=str,
        choices=utils.get_enum_names(ic.ImageFormat),
        help=f"The compression format of the images. "
        "If format is not selected, format of generated image is selected at random",
    )


def _add_input_args(parser):
    input_group = parser.add_argument_group("Input")

    input_group.add_argument(
        "--batch-size-audio",
        type=int,
        help=f"The audio batch size of the requests GenAI-Perf should send. "
        "This is currently supported with the OpenAI `multimodal` endpoint type.",
    )

    input_group.add_argument(
        "--batch-size-image",
        type=int,
        help=f"The image batch size of the requests GenAI-Perf should send. "
        "This is currently supported with the image retrieval endpoint type.",
    )

    input_group.add_argument(
        "--batch-size-text",
        "--batch-size",
        "-b",
        type=int,
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
        help="An option to provide constraints in order to compute goodput. "
        "Specify goodput constraints as 'key:value' pairs, where the key is a "
        "valid metric name, and the value is a number representing "
        "either milliseconds or a throughput value per second. For example, "
        "'request_latency:300' or 'output_token_throughput_per_user:600'. "
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
        help=f"The number of unique payloads to sample from. "
        "These will be reused until benchmarking is complete.",
    )

    input_group.add_argument(
        "--num-prefix-prompts",
        type=int,
        help=f"The number of prefix prompts to select from. "
        "If this value is not zero, these are prompts that are "
        "prepended to input prompts. This is useful for "
        "benchmarking models that use a K-V cache.",
    )

    input_group.add_argument(
        "--output-tokens-mean",
        "--osl",
        type=int,
        help=f"The mean number of tokens in each output. "
        "Ensure the --tokenizer value is set correctly. ",
    )

    input_group.add_argument(
        "--output-tokens-mean-deterministic",
        action="store_true",
        help=f"When using --output-tokens-mean, this flag can be set to "
        "improve precision by setting the minimum number of tokens "
        "equal to the requested number of tokens. This is currently "
        "supported with Triton. "
        "Note that there is still some variability in the requested number "
        "of output tokens, but GenAi-Perf attempts its best effort with your "
        "model to get the right number of output tokens. ",
    )

    input_group.add_argument(
        "--output-tokens-stddev",
        type=int,
        help=f"The standard deviation of the number of tokens in each output. "
        "This is only used when --output-tokens-mean is provided.",
    )

    input_group.add_argument(
        "--random-seed",
        type=int,
        help="The seed used to generate random values. If not provided, a "
        "random seed will be used.",
    )

    input_group.add_argument(
        "--grpc-method",
        type=str,
        help="A fully-qualified gRPC method name in "
        "'<package>.<service>/<method>' format. The option is only "
        "supported by dynamic gRPC service kind and is required to identify "
        "the RPC to use when sending requests to the server.",
    )

    input_group.add_argument(
        "--synthetic-input-tokens-mean",
        "--isl",
        type=int,
        help=f"The mean of number of tokens in the generated prompts when using synthetic data.",
    )

    input_group.add_argument(
        "--synthetic-input-tokens-stddev",
        type=int,
        help=f"The standard deviation of number of tokens in the generated prompts when using synthetic data.",
    )

    input_group.add_argument(
        "--prefix-prompt-length",
        type=int,
        help=f"The number of tokens in each prefix prompt. This value is only "
        "used if --num-prefix-prompts is positive. Note that due to "
        "the prefix and user prompts being concatenated, the number of tokens "
        "in the final prompt may be off by one.",
    )

    input_group.add_argument(
        "--warmup-request-count",
        "--num-warmup-requests",
        type=int,
        help=f"The number of warmup requests to send before benchmarking.",
    )


def _add_other_args(parser):
    other_group = parser.add_argument_group("Other")

    other_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="An option to enable verbose mode.",
    )


def _add_output_args(parser):
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--artifact-dir",
        type=Path,
        help="The directory to store all the (output) artifacts generated by "
        "GenAI-Perf and Perf Analyzer.",
    )
    output_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="The directory to store/restore the checkpoint generated by GenAI-Perf.",
    )
    output_group.add_argument(
        "--generate-plots",
        action="store_true",
        help="An option to enable the generation of plots.",
    )
    output_group.add_argument(
        "--enable-checkpointing",
        action="store_true",
        help="Enables checkpointing of the GenAI-Perf state. "
        "This is useful for running GenAI-Perf in a stateful manner.",
    )
    output_group.add_argument(
        "--profile-export-file",
        type=Path,
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
        "--input-directory",
        "-d",
        dest="input_path",
        nargs=1,
        type=str,
        required=False,
        help="The path to the directory containing the profile export files.",
    )


def _add_profile_args(parser):
    profile_group = parser.add_argument_group("Profiling")
    load_management_group = profile_group.add_mutually_exclusive_group(required=False)
    measurement_group = profile_group.add_mutually_exclusive_group(required=False)

    load_management_group.add_argument(
        "--concurrency",
        type=int,
        help="The concurrency value to benchmark.",
    )

    measurement_group.add_argument(
        "--measurement-interval",
        "-p",
        type=int,
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
        help="The number of requests to use for measurement.",
    )

    load_management_group.add_argument(
        "--request-rate",
        type=float,
        help="Sets the request rate for the load generated by PA.",
    )

    load_management_group.add_argument(
        "--fixed-schedule",
        type=bool,
        help="An option to enable fixed schedule (trace) inference load mode.",
    )

    profile_group.add_argument(
        "-s",
        "--stability-percentage",
        type=float,
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
        help="The number of sessions to simulate.",
    )

    session_load_management_group.add_argument(
        "--session-concurrency",
        type=int,
        help="The number of concurrent sessions to benchmark.",
    )

    session_group.add_argument(
        "--session-delay-ratio",
        type=float,
        help="A ratio to scale multi-turn delays when using a payload file. "
        "For example, a value of 0.5 will halve the specified delays.",
    )

    session_group.add_argument(
        "--session-turn-delay-mean",
        type=int,
        help="The mean delay (in ms) between turns in a session.",
    )

    session_group.add_argument(
        "--session-turn-delay-stddev",
        type=int,
        help="The standard deviation (in ms) of the delay between turns in "
        "a session.",
    )

    session_group.add_argument(
        "--session-turns-mean",
        type=int,
        help="The mean number of turns per session.",
    )

    session_group.add_argument(
        "--session-turns-stddev",
        type=int,
        help="The standard deviation of the number of turns per session.",
    )


def _add_tokenizer_args(parser):
    tokenizer_group = parser.add_argument_group("Tokenizer")

    tokenizer_group.add_argument(
        "--tokenizer",
        type=str,
        help="The HuggingFace tokenizer to use to interpret token metrics "
        "from prompts and responses. The value can be the name of a tokenizer "
        "or the filepath of the tokenizer. The default value is the model "
        "name.",
    )
    tokenizer_group.add_argument(
        "--tokenizer-revision",
        type=str,
        help="The specific model version to use. It can be a branch name, "
        "tag name, or commit ID.",
    )
    tokenizer_group.add_argument(
        "--tokenizer-trust-remote-code",
        action="store_true",
        help="Allow custom tokenizer to be downloaded and executed. "
        "This carries security risks and should only be used "
        "for repositories you trust. This is only necessary for custom "
        "tokenizers stored in HuggingFace Hub. ",
    )


def _parse_template_args(subparsers) -> argparse.ArgumentParser:
    template = subparsers.add_parser(
        ic.Subcommand.TEMPLATE.value,
        description="Subcommand to generate a template YAML file for profiling.",
    )
    _add_template_args(template)
    _add_other_args(template)
    template.set_defaults(func=template_handler)
    return template


def _parse_profile_args(subparsers) -> argparse.ArgumentParser:
    profile = subparsers.add_parser(
        ic.Subcommand.PROFILE.value,
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
        ic.Subcommand.ANALYZE.value,
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


def _parse_config_args(subparsers) -> argparse.ArgumentParser:
    config = subparsers.add_parser(
        ic.Subcommand.CONFIG.value,
        description="Subcommmand that indicates a config file is being used.",
    )

    _add_config_args(config)
    # This should be the superset of all possible args
    _add_analyze_args(config)
    _add_audio_input_args(config)
    _add_endpoint_args(config)
    _add_image_input_args(config)
    _add_input_args(config)
    _add_other_args(config)
    _add_output_args(config)
    _add_profile_args(config)
    _add_session_args(config)
    _add_tokenizer_args(config)

    config.set_defaults(func=config_handler)
    return config


def _parse_process_export_files_args(subparsers) -> argparse.ArgumentParser:
    process_export_files = subparsers.add_parser(
        ic.Subcommand.PROCESS.value,
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
    _ = _parse_config_args(subparsers)
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
    if not args.subcommand:
        return args

    if args.subcommand == ic.Subcommand.CONFIG.value:
        pass
    elif args.subcommand == ic.Subcommand.PROFILE.value:
        args = _check_goodput_args(args)
    elif args.subcommand == ic.Subcommand.ANALYZE.value:
        args = _process_sweep_args(args)
        args = _check_goodput_args(args)
    elif args.subcommand == ic.Subcommand.TEMPLATE.value:
        pass
    elif args.subcommand == ic.Subcommand.PROCESS.value:
        pass
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")

    return args


### Entrypoint ###
def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    argv = sys.argv
    parser = init_parsers()
    passthrough_index = get_passthrough_args_index(argv)
    extra_args = argv[passthrough_index + 1 :]

    args = parser.parse_args(argv[1:passthrough_index])
    args = refine_args(parser, args)

    return args, extra_args
