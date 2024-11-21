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

from argparse import Namespace
from typing import Tuple

import genai_perf.logging as logging
from genai_perf.checkpoint.checkpoint import Checkpoint
from genai_perf.config.generate.genai_perf_config import GenAIPerfConfig
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.generate.search_parameters import SearchParameters
from genai_perf.config.generate.sweep_objective_generator import SweepObjectiveGenerator
from genai_perf.config.input.config_command import ConfigCommand, Range, Subcommand
from genai_perf.config.run.run_config import RunConfig
from genai_perf.exceptions import GenAIPerfException
from genai_perf.measurements.run_config_measurement import RunConfigMeasurement
from genai_perf.subcommand.common import (
    calculate_metrics,
    create_artifacts_dirs,
    create_config_options,
    create_telemetry_data_collector,
    generate_inputs,
    run_perf_analyzer,
)
from genai_perf.tokenizer import get_tokenizer
from genai_perf.types import GpuRecords, ModelObjectiveParameters
from genai_perf.wrapper import Profiler

logger = logging.getLogger(__name__)


def analyze_handler(args: Namespace) -> None:
    """
    Handles `analyze` subcommand workflow
    """
    #
    # Setup
    config = _setup_config(args)

    model_name = config.model_names[0]
    model_search_parameters = {
        model_name: SearchParameters(config=config, subcommand=Subcommand.ANALYZE)
    }
    sweep_objective_generator = SweepObjectiveGenerator(config, model_search_parameters)
    telemetry_data_collector = create_telemetry_data_collector(args)
    checkpoint = Checkpoint(config)
    results = checkpoint.results

    #
    # Sweep Loop
    for count, objectives in enumerate(sweep_objective_generator.get_objectives()):
        #
        # Create GAP/PA Configs
        genai_perf_config = GenAIPerfConfig(
            config=config, args=args, model_objective_parameters=objectives
        )

        # The GAP/PA Configs will (for now) modify the CLI args
        # based on the objective being swept
        gap_obj_args = genai_perf_config.get_obj_args()

        perf_analyzer_config = PerfAnalyzerConfig(
            model_name=model_name,
            args=gap_obj_args,
            config=config,
            model_objective_parameters=objectives,
        )
        obj_args = perf_analyzer_config.get_obj_args()

        #
        # Create Input/Artifacts
        input_config_options = create_config_options(obj_args)
        create_artifacts_dirs(obj_args)
        tokenizer = get_tokenizer(
            obj_args.tokenizer,
            obj_args.tokenizer_trust_remote_code,
            obj_args.tokenizer_revision,
        )
        generate_inputs(input_config_options)

        #
        # Run PA
        run_perf_analyzer(
            args=obj_args,
            perf_analyzer_config=perf_analyzer_config,
            telemetry_data_collector=telemetry_data_collector,
        )

        #
        # Extract Perf Metrics
        infer_mode, load_level = _determine_infer_mode_and_load_level(
            obj_args, objectives, model_name
        )
        data_parser = calculate_metrics(obj_args, tokenizer)
        perf_stats = data_parser.get_statistics(infer_mode, load_level)
        perf_metrics = perf_stats.create_records()

        #
        # Extract Telemetry Metrics
        # FIXME: Once I'm able to collect telemetry records will need
        # to write a method to hook this up
        # telemetry_stats = (
        #     telemetry_data_collector.get_statistics()
        #     if telemetry_data_collector
        #     else None
        # )
        gpu_metrics: GpuRecords = {}

        #
        # Create RunConfigMeasurement
        run_config_measurement = RunConfigMeasurement(gpu_metrics)
        run_config_measurement.add_perf_metrics(model_name, perf_metrics)

        #
        # Create RunConfig
        run_config_name = model_name + "_run_config_" + str(count)
        run_config = RunConfig(
            name=run_config_name,
            genai_perf_config=genai_perf_config,
            perf_analyzer_config=perf_analyzer_config,
            measurement=run_config_measurement,
        )

        #
        # Add to results and write checkpoint
        results.add_run_config(run_config)
        checkpoint.create_checkpoint_object()


def _setup_config(args: Namespace) -> ConfigCommand:
    config = ConfigCommand(model_names=args.model)

    if args.sweep_list:
        config.analyze.sweep_parameters = {args.sweep_type: args.sweep_list}
    else:
        config.analyze.sweep_parameters = {
            args.sweep_type: Range(min=args.sweep_min, max=args.sweep_max)
        }

    return config


def _determine_infer_mode_and_load_level(
    args: Namespace, objectives: ModelObjectiveParameters, model_name: str
) -> Tuple[str, str]:
    if args.sweep_type == "concurrency":
        infer_mode = "concurrency"
        load_level = (
            f"{objectives[model_name][infer_mode].get_value_based_on_category()}"
        )
    elif args.sweep_type == "request_rate":
        infer_mode = "request_rate"
        load_level = (
            f"{float(objectives[model_name][infer_mode].get_value_based_on_category())}"
        )
    elif args.sweep_type == "input_sequence_length" or args.sweep_type == "num_prompts":
        if args.concurrency:
            infer_mode = "concurrency"
            load_level = f"{args.concurrency}"
        elif args.request_rate:
            infer_mode = "request_rate"
            load_level = f"{args.request_rate}"
        else:
            infer_mode = "concurrency"
            load_level = "1"
    else:
        raise GenAIPerfException("Cannot determine infer_mode/load_level")

    return infer_mode, load_level
