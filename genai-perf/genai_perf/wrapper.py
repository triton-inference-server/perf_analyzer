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

import subprocess
from argparse import Namespace
from typing import List, Optional

import genai_perf.logging as logging
import genai_perf.utils as utils
from genai_perf.constants import DEFAULT_GRPC_URL
from genai_perf.inputs.input_constants import DEFAULT_INPUT_DATA_JSON
from genai_perf.inputs.inputs import OutputFormat
from genai_perf.telemetry_data.triton_telemetry_data_collector import (
    TelemetryDataCollector,
)

logger = logging.getLogger(__name__)


class Profiler:
    @staticmethod
    def add_protocol_args(args: Namespace) -> List[str]:
        cmd = []
        if args.service_kind == "triton":
            cmd += ["-i", "grpc", "--streaming"]
            if args.u is None:  # url
                cmd += ["-u", f"{DEFAULT_GRPC_URL}"]
            if args.output_format == OutputFormat.TENSORRTLLM:
                cmd += ["--shape", "max_tokens:1", "--shape", "text_input:1"]
        elif args.service_kind == "openai":
            cmd += ["-i", "http"]
        return cmd

    @staticmethod
    def add_inference_load_args(args: Namespace) -> List[str]:
        cmd = []
        if args.concurrency:
            cmd += ["--concurrency-range", f"{args.concurrency}"]
        elif args.request_rate:
            cmd += ["--request-rate-range", f"{args.request_rate}"]
        return cmd

    @staticmethod
    def build_cmd(args: Namespace, extra_args: Optional[List[str]] = None) -> List[str]:
        skip_args = [
            "artifact_dir",
            "backend",
            "batch_size_image",
            "batch_size_text",
            "concurrency",
            "endpoint_type",
            "extra_inputs",
            "formatted_model_name",
            "func",
            "generate_plots",
            "goodput",
            "image_format",
            "image_height_mean",
            "image_height_stddev",
            "image_width_mean",
            "image_width_stddev",
            "input_dataset",
            "input_file",
            "input_format",
            "model",
            "model_selection_strategy",
            "num_prompts",
            "output_format",
            "output_tokens_mean",
            "output_tokens_mean_deterministic",
            "output_tokens_stddev",
            "prompt_source",
            "random_seed",
            "request_rate",
            "server_metrics_url",
            # The 'streaming' passed in to this script is to determine if the
            # LLM response should be streaming. That is different than the
            # 'streaming' that PA takes, which means something else (and is
            # required for decoupled models into triton).
            "streaming",
            "subcommand",
            "synthetic_input_files",
            "synthetic_input_tokens_mean",
            "synthetic_input_tokens_stddev",
            "tokenizer",
            "tokenizer_trust_remote_code",
            "tokenizer_revision",
        ]

        utils.remove_file(args.profile_export_file)

        cmd = [
            f"perf_analyzer",
            f"-m",
            f"{args.formatted_model_name}",
            f"--async",
            f"--input-data",
            f"{args.artifact_dir / DEFAULT_INPUT_DATA_JSON}",
        ]
        cmd += Profiler.add_protocol_args(args)
        cmd += Profiler.add_inference_load_args(args)

        for arg, value in vars(args).items():
            if arg in skip_args:
                pass
            elif value is None:
                pass
            elif value is False:
                pass
            elif value is True:
                if len(arg) == 1:
                    cmd += [f"-{arg}"]
                else:
                    cmd += [f"--{arg}"]
            # GAP needs to call PA using triton_c_api service kind when running
            # against tensorrtllm engine.
            elif arg == "service_kind" and value == "tensorrtllm_engine":
                cmd += ["--service-kind", "triton_c_api", "--streaming"]
            else:
                if len(arg) == 1:
                    cmd += [f"-{arg}", f"{value}"]
                else:
                    arg = utils.convert_option_name(arg)
                    cmd += [f"--{arg}", f"{value}"]

        if extra_args is not None:
            for arg in extra_args:
                cmd += [f"{arg}"]
        return cmd

    @staticmethod
    def run(
        args: Namespace,
        extra_args: Optional[List[str]],
        telemetry_data_collector: Optional[TelemetryDataCollector] = None,
    ) -> None:
        try:
            if telemetry_data_collector is not None:
                telemetry_data_collector.start()
            cmd = Profiler.build_cmd(args, extra_args)
            logger.info(f"Running Perf Analyzer : '{' '.join(cmd)}'")
            if args and args.verbose:
                subprocess.run(cmd, check=True, stdout=None)
            else:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        finally:
            if telemetry_data_collector is not None:
                telemetry_data_collector.stop()
