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

import csv
import os
from argparse import Namespace
from typing import List, Tuple

import genai_perf.logging as logging
from genai_perf.checkpoint.checkpoint import Checkpoint
from genai_perf.config.generate.genai_perf_config import GenAIPerfConfig
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.generate.search_parameters import SearchParameters
from genai_perf.config.generate.sweep_objective_generator import SweepObjectiveGenerator
from genai_perf.config.input.config_command import ConfigCommand, Range, Subcommand
from genai_perf.config.run.run_config import RunConfig
from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.measurements.run_config_measurement import RunConfigMeasurement
from genai_perf.metrics.telemetry_statistics import TelemetryStatistics
from genai_perf.record.types.energy_consumption_p99 import GpuEnergyConsumptionP99
from genai_perf.record.types.gpu_memory_used_p99 import GpuMemoryUsedP99
from genai_perf.record.types.gpu_power_limit_avg import GPUPowerLimitAvg
from genai_perf.record.types.gpu_power_usage_p99 import GPUPowerUsageP99
from genai_perf.record.types.gpu_utilization_p99 import GPUUtilizationP99
from genai_perf.record.types.input_sequence_length_p99 import InputSequenceLengthP99
from genai_perf.record.types.inter_token_latency_p99 import InterTokenLatencyP99
from genai_perf.record.types.output_sequence_length_p99 import OutputSequenceLengthP99
from genai_perf.record.types.output_token_throughput_avg import OutputTokenThroughputAvg
from genai_perf.record.types.request_latency_p99 import RequestLatencyP99
from genai_perf.record.types.request_throughput_avg import RequestThroughputAvg
from genai_perf.record.types.time_to_first_token_p99 import TimeToFirstTokenP99
from genai_perf.record.types.total_gpu_memory_avg import GPUTotalMemoryAvg
from genai_perf.subcommand.common import (
    calculate_metrics,
    create_artifacts_dirs,
    create_config_options,
    create_telemetry_data_collectors,
    generate_inputs,
    merge_telemetry_metrics,
    run_perf_analyzer,
)
from genai_perf.tokenizer import get_tokenizer
from genai_perf.types import ModelObjectiveParameters

logger = logging.getLogger(__name__)


###########################################################################
# Analyze Handler
###########################################################################
def analyze_handler(args: Namespace) -> None:
    """
    Handles `analyze` subcommand workflow
    """
    analyze = Analyze(args)
    analyze.sweep()
    analyze.report()


###########################################################################
# Analyze Class
###########################################################################
class Analyze:
    """
    Contains all the methods needed to run the analyze subcommand
    """

    PERF_METRICS_HEADER = [
        TimeToFirstTokenP99.header(),
        InterTokenLatencyP99.header(),
        RequestLatencyP99.header(),
        OutputSequenceLengthP99.header(),
        OutputTokenThroughputAvg.header(),
        RequestThroughputAvg.header(),
    ]
    PERF_METRICS_TAGS = [
        TimeToFirstTokenP99.tag,
        InterTokenLatencyP99.tag,
        RequestLatencyP99.tag,
        OutputSequenceLengthP99.tag,
        OutputTokenThroughputAvg.tag,
        RequestThroughputAvg.tag,
    ]

    GPU_METRICS_HEADER = [
        "Config Name",
        "GPU",
        GPUPowerUsageP99.header(),
        GpuEnergyConsumptionP99.header(),
        GPUUtilizationP99.header(),
        GpuMemoryUsedP99.header(),
        GPUPowerLimitAvg.header(),
        GPUTotalMemoryAvg.header(),
    ]
    GPU_METRICS_TAGS = [
        GPUPowerUsageP99.tag,
        GpuEnergyConsumptionP99.tag,
        GPUUtilizationP99.tag,
        GpuMemoryUsedP99.tag,
        GPUPowerLimitAvg.tag,
        GPUTotalMemoryAvg.tag,
    ]

    def __init__(self, args: Namespace) -> None:
        self._args = args
        self._config = self._setup_config(args)
        self._model_name = self._config.model_names[0]
        self._model_search_parameters = {
            self._model_name: SearchParameters(
                config=self._config, subcommand=Subcommand.ANALYZE
            )
        }
        self._sweep_objective_generator = SweepObjectiveGenerator(
            self._config, self._model_search_parameters
        )
        self._telemetry_data_collectors = create_telemetry_data_collectors(args)
        self._checkpoint = Checkpoint(self._config)
        self._results = self._checkpoint.results

    def _setup_config(self, args: Namespace) -> ConfigCommand:
        config = ConfigCommand(model_names=args.model)
        sweep_type = self._map_args_to_config_sweep_type(args.sweep_type)

        if args.sweep_list:
            config.analyze.sweep_parameters = {sweep_type: args.sweep_list}
        else:
            config.analyze.sweep_parameters = {
                sweep_type: Range(min=args.sweep_min, max=args.sweep_max)
            }

        return config

    def _map_args_to_config_sweep_type(self, sweep_type: str) -> str:
        # The CLI arg sweep type name doesn't have a 1:1 mapping to
        # what was implemented in the config
        if sweep_type == "batch_size":
            return "runtime_batch_size"
        else:
            return sweep_type

    ###########################################################################
    # Sweep Methods
    ###########################################################################
    def sweep(self) -> None:
        """
        Sweeps over the objectives
        """
        for count, objectives in enumerate(
            self._sweep_objective_generator.get_objectives()
        ):
            #
            # Create GAP/PA Configs
            genai_perf_config = GenAIPerfConfig(
                config=self._config,
                args=self._args,
                model_objective_parameters=objectives,
            )

            # The GAP/PA Configs will (for now) modify the CLI args
            # based on the objective being swept
            gap_obj_args = genai_perf_config.get_obj_args()

            perf_analyzer_config = PerfAnalyzerConfig(
                model_name=self._model_name,
                args=gap_obj_args,
                config=self._config,
                model_objective_parameters=objectives,
            )
            obj_args = perf_analyzer_config.get_obj_args()

            #
            # Check if this configuration has already been profiled (is in the checkpoint)
            representation = RunConfig(
                genai_perf_config=genai_perf_config,
                perf_analyzer_config=perf_analyzer_config,
            ).representation()

            run_config_found = self._results.found_representation(representation)
            run_config_name = self._results.get_run_config_name_based_on_representation(
                self._model_name, representation
            )

            if not run_config_found:
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
                    telemetry_data_collectors=self._telemetry_data_collectors,
                )

                #
                # Extract Perf Metrics
                infer_mode, load_level = self._determine_infer_mode_and_load_level(
                    obj_args, objectives, self._model_name
                )
                data_parser = calculate_metrics(obj_args, tokenizer)
                perf_stats = data_parser.get_statistics(infer_mode, load_level)
                perf_metrics = perf_stats.create_records()

                #
                # Extract Telemetry Metrics
                telemetry_metrics_list = [
                    collector.get_metrics()
                    for collector in self._telemetry_data_collectors
                ]

                merged_telemetry_metrics = merge_telemetry_metrics(
                    telemetry_metrics_list
                )
                merged_telemetry_stats = TelemetryStatistics(merged_telemetry_metrics)
                gpu_metrics = merged_telemetry_stats.create_records()

                #
                # Create output CSV in artifact directory
                OutputReporter(
                    perf_stats, merged_telemetry_stats, obj_args
                ).report_output()

                #
                # Create RunConfigMeasurement
                run_config_measurement = RunConfigMeasurement(gpu_metrics)
                run_config_measurement.add_perf_metrics(self._model_name, perf_metrics)

                #
                # Create RunConfig
                run_config = RunConfig(
                    name=run_config_name,
                    genai_perf_config=genai_perf_config,
                    perf_analyzer_config=perf_analyzer_config,
                    measurement=run_config_measurement,
                )

                #
                # Add to results and write checkpoint
                self._results.add_run_config(run_config)
                self._checkpoint.create_checkpoint_object()

            else:
                objective = list(objectives[self._model_name])[0]
                value = list(objectives[self._model_name].values())[0].value
                logger.info(
                    f"{run_config_name}:{objective}{value} found in checkpoint - skipping profiling..."
                )

    ###########################################################################
    # Report Methods
    ###########################################################################
    def report(self) -> None:
        """
        Creates a CSV report based on checkpointed results
        """
        filename = os.getcwd() + "/analyze_export_genai_perf.csv"
        logger.info(f"Generating {filename}")

        with open(filename, mode="w", newline="") as f:
            csv_writer = csv.writer(f)

            #
            # Perf Metrics
            self._write_perf_metrics_header(csv_writer)
            self._write_perf_metrics_body(csv_writer)
            csv_writer.writerow("")

            #
            # GPU Metrics
            self._write_gpu_metrics_header(csv_writer)
            self._write_gpu_metrics_body(csv_writer)

    ###########################################################################
    # Report - Perf Metrics Methods
    ###########################################################################
    def _create_stimulus_header(self) -> List[str]:
        infer_type = self._determine_infer_type()
        infer_header = "Concurrency" if infer_type == "concurrency" else "Request Rate"

        stimulus_header = ["Config Name", infer_header, "ISL", "Num Dataset Entries"]

        return stimulus_header

    def _write_perf_metrics_header(self, csv_writer) -> None:
        stimulus_header = self._create_stimulus_header()
        csv_writer.writerow(stimulus_header + Analyze.PERF_METRICS_HEADER)

    def _write_perf_metrics_body(self, csv_writer) -> None:
        for run_config in self._results.run_configs:
            infer_value = run_config.perf_analyzer_config.get_inference_value()

            isl = int(
                run_config.get_model_perf_metric_value(
                    self._model_name, InputSequenceLengthP99.tag
                )
            )
            num_dataset_entries = run_config.genai_perf_config.input.num_dataset_entries

            metrics = []
            for tag in Analyze.PERF_METRICS_TAGS:
                metric = run_config.get_model_perf_metric_value(self._model_name, tag)
                metrics.append(f"{metric:.2f}")

            row = [run_config.name, infer_value, isl, num_dataset_entries] + metrics
            csv_writer.writerow(row)

    ###########################################################################
    # Report - GPU Metrics Methods
    ###########################################################################
    def _write_gpu_metrics_header(self, csv_writer) -> None:
        csv_writer.writerow(Analyze.GPU_METRICS_HEADER)

    def _write_gpu_metrics_body(self, csv_writer) -> None:
        gpus = self._get_list_of_gpus()

        for gpu in gpus:
            for run_config in self._results.run_configs:
                gpu_metrics = []
                for tag in Analyze.GPU_METRICS_TAGS:
                    gpu_metric = run_config.get_gpu_metric_value(gpu, tag)
                    gpu_metrics.append(f"{gpu_metric:.2f}")

                row = [run_config.name, gpu]
                row.extend(gpu_metrics)
                csv_writer.writerow(row)

    def _get_list_of_gpus(self) -> List[str]:
        gpu_power_usage = self._results.run_configs[0].get_gpu_metric(
            GPUPowerUsageP99.tag
        )

        gpus = []
        if gpu_power_usage:
            gpus = list(gpu_power_usage.keys())

        return gpus

    ###########################################################################
    # Inference Determination Methods
    ###########################################################################
    def _determine_infer_mode_and_load_level(
        self, args: Namespace, objectives: ModelObjectiveParameters, model_name: str
    ) -> Tuple[str, str]:
        if args.sweep_type == "concurrency":
            infer_mode = "concurrency"
            load_level = (
                f"{objectives[model_name][infer_mode].get_value_based_on_category()}"
            )
        elif args.sweep_type == "request_rate":
            infer_mode = "request_rate"
            load_level = f"{float(objectives[model_name][infer_mode].get_value_based_on_category())}"
        elif (
            args.sweep_type == "input_sequence_length"
            or args.sweep_type == "num_dataset_entries"
            or args.sweep_type == "batch_size"
        ):
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

    def _determine_infer_type(self):
        if self._args.sweep_type == "concurrency":
            infer_type = "concurrency"
        elif self._args.sweep_type == "request_rate":
            infer_type = "request_rate"
        else:
            if self._args.concurrency:
                infer_type = "concurrency"
            elif self._args.request_rate:
                infer_type = "request_rate"
            else:
                infer_type = "concurrency"

        return infer_type
