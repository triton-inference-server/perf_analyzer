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

import os
import subprocess  # nosec
from typing import Dict, List, Optional, Tuple

import genai_perf.logging as logging
from genai_perf.checkpoint.checkpoint import Checkpoint
from genai_perf.config.generate.genai_perf_config import GenAIPerfConfig
from genai_perf.config.generate.objective_parameter import (
    ObjectiveCategory,
    ObjectiveParameter,
)
from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.generate.search_parameter import SearchUsage
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.run.results import Results
from genai_perf.config.run.run_config import RunConfig
from genai_perf.constants import DEFAULT_DCGM_METRICS_URL
from genai_perf.exceptions import GenAIPerfException
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.inputs.input_constants import OutputFormat, PromptSource
from genai_perf.inputs.inputs import Inputs
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.measurements.run_config_measurement import RunConfigMeasurement
from genai_perf.metrics import Statistics
from genai_perf.metrics.telemetry_metrics import TelemetryMetrics
from genai_perf.metrics.telemetry_statistics import TelemetryStatistics
from genai_perf.profile_data_parser import (
    ImageRetrievalProfileDataParser,
    LLMProfileDataParser,
    ProfileDataParser,
)
from genai_perf.telemetry_data.dcgm_telemetry_data_collector import (
    DCGMTelemetryDataCollector,
)
from genai_perf.telemetry_data.telemetry_data_collector import TelemetryDataCollector
from genai_perf.tokenizer import Tokenizer, get_tokenizer
from genai_perf.types import GpuRecords, ModelObjectiveParameters, PerfRecords
from genai_perf.utils import remove_file

logger = logging.getLogger(__name__)


class Subcommand:
    def __init__(
        self, config: ConfigCommand, extra_args: Optional[List[str]] = None
    ) -> None:
        # These fields are constant throughout the GAP run
        self._config = config
        self._extra_args = extra_args
        self._model_name = (
            self._config.model_names[0] if self._config.model_names else ""
        )
        self._telemetry_data_collectors = self._create_telemetry_data_collectors()

        if config.output.enable_checkpointing:
            self._checkpoint: Optional[Checkpoint] = Checkpoint(self._config)
            self._results = self._checkpoint.results
        else:
            self._checkpoint = None
            self._results = Results()

        # Will only initialize the tokenizer if necessary (no match in checkpoint)
        # to save time
        self._tokenizer: Optional[Tokenizer] = None

        # These fields can change (based on objectives), vary from run to run
        # and are used by multiple methods
        self._data_parser: Optional[ProfileDataParser] = None

    ###########################################################################
    # Perf Analyzer Methods
    ###########################################################################
    def _run_perf_analyzer(
        self,
        perf_analyzer_config: PerfAnalyzerConfig,
    ) -> None:
        try:
            for collector in self._telemetry_data_collectors:
                if collector:
                    collector.start()

            remove_file(perf_analyzer_config.get_profile_export_file())
            cmd = perf_analyzer_config.create_command()
            logger.info(f"Running Perf Analyzer : '{' '.join(cmd)}'")

            if self._config.verbose or self._config.perf_analyzer.verbose:
                subprocess.run(cmd, check=True, stdout=None)  # nosec
            else:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)  # nosec
        finally:
            for collector in self._telemetry_data_collectors:
                if collector:
                    collector.stop()

    ###########################################################################
    # Config Methods
    ###########################################################################
    def _is_config_present_in_results(
        self,
        genai_perf_config: GenAIPerfConfig,
        perf_analyzer_config: PerfAnalyzerConfig,
    ) -> bool:
        representation = self._create_representation(
            genai_perf_config, perf_analyzer_config
        )
        representation_found = self._results.found_representation(representation)

        return representation_found

    def _get_run_config_name(
        self,
        genai_perf_config: GenAIPerfConfig,
        perf_analyzer_config: PerfAnalyzerConfig,
    ) -> str:
        representation = self._create_representation(
            genai_perf_config, perf_analyzer_config
        )
        run_config_name = self._results.get_run_config_name_based_on_representation(
            self._model_name, representation
        )

        return run_config_name

    ###########################################################################
    # Checkpoint Methods
    ###########################################################################
    def _add_results_to_checkpoint(
        self,
        genai_perf_config: GenAIPerfConfig,
        perf_analyzer_config: PerfAnalyzerConfig,
        objectives: ModelObjectiveParameters,
    ) -> None:
        run_config = self._create_run_config(
            genai_perf_config, perf_analyzer_config, objectives
        )
        self._results.add_run_config(run_config)

        if self._config.output.enable_checkpointing:
            self._checkpoint.create_checkpoint_object()  # type: ignore

    def _found_config_in_checkpoint(
        self,
        genai_perf_config: GenAIPerfConfig,
        perf_analyzer_config: PerfAnalyzerConfig,
        objectives: ModelObjectiveParameters,
    ) -> None:
        obj_list = []
        for name, parameter in objectives[self._model_name].items():
            obj_list.append(f"{name}{parameter.get_value_based_on_category()}")

        obj_str = "-".join(obj_list)
        run_config_name = self._get_run_config_name(
            genai_perf_config, perf_analyzer_config
        )
        logger.info(
            f"{run_config_name}:{obj_str} found in checkpoint. Results in '{perf_analyzer_config.get_artifact_directory()}/' - skipping profiling..."
        )

    ###########################################################################
    # Create Methods
    ###########################################################################
    def _create_artifact_directory(
        self, perf_analyzer_config: PerfAnalyzerConfig
    ) -> None:
        os.makedirs(perf_analyzer_config.get_artifact_directory(), exist_ok=True)

    def _create_plot_directory(self, perf_analyzer_config: PerfAnalyzerConfig) -> None:
        if self._config.output.generate_plots:
            plot_dir = perf_analyzer_config.get_artifact_directory() / "plots"
            os.makedirs(plot_dir, exist_ok=True)

    def _create_tokenizer(self) -> None:
        if self._tokenizer:
            return

        logger.info(f"Creating tokenizer for: {self._config.tokenizer.name}")
        self._tokenizer = get_tokenizer(self._config)

    def _create_inputs_config(
        self, perf_analyzer_config: PerfAnalyzerConfig
    ) -> InputsConfig:
        inputs_config = InputsConfig(
            config=self._config,
            tokenizer=self._tokenizer,  # type: ignore
            output_directory=perf_analyzer_config.get_artifact_directory(),
        )

        return inputs_config

    def _create_telemetry_data_collectors(
        self,
    ) -> List[Optional[TelemetryDataCollector]]:
        if self._config.endpoint.service_kind == "triton":
            logger.warning(
                "GPU metrics are no longer collected from Triton's /metrics endpoint.\n"
                "Telemetry is now collected exclusively from the DCGM-Exporter /metrics endpoint.\n"
                "If you're using Triton, please ensure DCGM-Exporter is running.\n"
            )
        server_metrics_urls = self._config.endpoint.server_metrics_urls or [
            DEFAULT_DCGM_METRICS_URL
        ]
        telemetry_collectors: List[Optional[TelemetryDataCollector]] = []

        for url in map(str.strip, server_metrics_urls):
            collector = DCGMTelemetryDataCollector(url)
            if collector.is_url_reachable():
                telemetry_collectors.append(collector)
            else:
                logger.warning(f"Skipping unreachable metrics URL: {url}")

        return telemetry_collectors

    def _create_objectives_based_on_stimulus(self) -> ModelObjectiveParameters:
        objectives: ModelObjectiveParameters = {self._model_name: {}}
        if (
            self._config.perf_analyzer.get_field("stimulus").is_set_by_user
            or self._config.input.prompt_source != PromptSource.PAYLOAD
        ):
            for key, value in self._config.perf_analyzer.stimulus.items():
                objectives[self._model_name][key] = ObjectiveParameter(
                    SearchUsage.RUNTIME_PA, ObjectiveCategory.INTEGER, value
                )

        return objectives

    def _create_genai_perf_config(
        self, objectives: ModelObjectiveParameters
    ) -> GenAIPerfConfig:
        genai_perf_config = GenAIPerfConfig(
            config=self._config,
            model_objective_parameters=objectives,
        )

        return genai_perf_config

    def _create_perf_analyzer_config(
        self, objectives: ModelObjectiveParameters
    ) -> PerfAnalyzerConfig:
        perf_analyzer_config = PerfAnalyzerConfig(
            config=self._config,
            model_objective_parameters=objectives,
            extra_args=self._extra_args,
        )

        return perf_analyzer_config

    def _create_representation(
        self,
        genai_perf_config: GenAIPerfConfig,
        perf_analyzer_config: PerfAnalyzerConfig,
    ) -> str:
        run_config = RunConfig(
            genai_perf_config=genai_perf_config,
            perf_analyzer_config=perf_analyzer_config,
        )
        representation = run_config.representation()

        return representation

    def _create_telemetry_metrics_list(self) -> List[TelemetryMetrics]:
        telemetry_metrics_list = [
            collector.get_metrics()  # type: ignore
            for collector in self._telemetry_data_collectors
        ]

        return telemetry_metrics_list

    def _create_merged_telemetry_stats(self) -> TelemetryStatistics:
        telemetry_metrics_list = self._create_telemetry_metrics_list()
        merged_telemetry_metrics = self._merge_telemetry_metrics(telemetry_metrics_list)
        telemetry_stats = TelemetryStatistics(merged_telemetry_metrics)

        return telemetry_stats

    def _create_run_config(
        self,
        genai_perf_config: GenAIPerfConfig,
        perf_analyzer_config: PerfAnalyzerConfig,
        objectives: ModelObjectiveParameters,
    ) -> RunConfig:
        run_config_name = self._get_run_config_name(
            genai_perf_config, perf_analyzer_config
        )
        run_config_measurement = self._create_run_config_measurement(
            perf_analyzer_config, objectives
        )

        run_config = RunConfig(
            name=run_config_name,
            genai_perf_config=genai_perf_config,
            perf_analyzer_config=perf_analyzer_config,
            measurement=run_config_measurement,
        )

        return run_config

    def _create_run_config_measurement(
        self,
        perf_analyzer_config: PerfAnalyzerConfig,
        objectives: ModelObjectiveParameters,
    ) -> RunConfigMeasurement:
        gpu_metrics = self._create_gpu_metrics()
        perf_metrics = self._create_perf_metrics(perf_analyzer_config, objectives)

        run_config_measurement = RunConfigMeasurement(gpu_metrics)
        run_config_measurement.add_perf_metrics(self._model_name, perf_metrics)

        return run_config_measurement

    def _create_gpu_metrics(self) -> GpuRecords:
        merged_telemetry_stats = self._create_merged_telemetry_stats()
        gpu_metrics = merged_telemetry_stats.create_records()

        return gpu_metrics

    def _create_perf_metrics(
        self,
        perf_analyzer_config: PerfAnalyzerConfig,
        objectives: ModelObjectiveParameters,
    ) -> PerfRecords:
        perf_stats = self._create_perf_stats(perf_analyzer_config, objectives)
        perf_metrics = perf_stats.create_records()

        return perf_metrics

    def _create_perf_stats(
        self,
        perf_analyzer_config: PerfAnalyzerConfig,
        objectives: ModelObjectiveParameters,
    ) -> Statistics:
        infer_mode, load_level = self._determine_infer_mode_and_load_level(objectives)
        perf_stats = self._data_parser.get_statistics(infer_mode, load_level)  # type: ignore

        return perf_stats

    def _create_session_stats(
        self,
        perf_analyzer_config: PerfAnalyzerConfig,
        objectives: ModelObjectiveParameters,
    ) -> Dict[str, Statistics]:
        session_stats = self._data_parser.get_session_statistics()  # type: ignore

        return session_stats

    ###########################################################################
    # Metrics/Statistics Methods
    ###########################################################################
    def _calculate_metrics(
        self,
        perf_analyzer_config: PerfAnalyzerConfig,
    ) -> ProfileDataParser:
        if self._config.endpoint.output_format == OutputFormat.TEMPLATE:
            return ProfileDataParser(
                perf_analyzer_config.get_profile_export_file(),
                goodput_constraints=self._config.input.goodput,
            )
        elif self._config.endpoint.type in [
            "embeddings",
            "nvclip",
            "rankings",
            "dynamic_grpc",
        ]:
            return ProfileDataParser(
                perf_analyzer_config.get_profile_export_file(),
                goodput_constraints=self._config.input.goodput,
            )
        elif self._config.endpoint.type == "image_retrieval":
            return ImageRetrievalProfileDataParser(
                perf_analyzer_config.get_profile_export_file(),
                goodput_constraints=self._config.input.goodput,
            )
        else:
            return LLMProfileDataParser(
                filename=perf_analyzer_config.get_profile_export_file(),
                tokenizer=self._tokenizer,  # type: ignore
                goodput_constraints=self._config.input.goodput,
            )

    def _merge_telemetry_metrics(
        self,
        metrics_list: List[TelemetryMetrics],
    ) -> TelemetryMetrics:
        """
        Merges multiple TelemetryMetrics objects into a single one.
        """

        merged_metrics = TelemetryMetrics()

        for metrics in metrics_list:
            for metric in TelemetryMetrics.TELEMETRY_METRICS:
                metric_key = metric.name
                metric_dict = getattr(merged_metrics, metric_key)
                source_dict = getattr(metrics, metric_key)

                for gpu_id, values in source_dict.items():
                    metric_dict[gpu_id].extend(values)
        return merged_metrics

    def _set_data_parser(self, perf_analyzer_config: PerfAnalyzerConfig) -> None:
        self._data_parser = self._calculate_metrics(perf_analyzer_config)

    ###########################################################################
    # Inputs Methods
    ###########################################################################
    def _generate_inputs(self, perf_analyzer_config: PerfAnalyzerConfig) -> None:
        inputs_config = self._create_inputs_config(perf_analyzer_config)
        inputs = Inputs(inputs_config)
        inputs.create_inputs()

    ###########################################################################
    # Outputs Methods
    ###########################################################################
    def _add_output_to_artifact_directory(
        self,
        perf_analyzer_config: PerfAnalyzerConfig,
        objectives: ModelObjectiveParameters,
    ) -> None:
        perf_stats = self._create_perf_stats(perf_analyzer_config, objectives)
        merged_telemetry_stats = self._create_merged_telemetry_stats()
        session_stats = self._create_session_stats(perf_analyzer_config, objectives)

        OutputReporter(
            perf_stats,
            merged_telemetry_stats,
            self._config,
            perf_analyzer_config,
            session_stats,
        ).report_output()

    ###########################################################################
    # Inference Determination Methods
    ###########################################################################
    def _determine_infer_mode_and_load_level(
        self,
        objectives: ModelObjectiveParameters,
    ) -> Tuple[str, str]:
        if self._config.analyze.any_field_set_by_user():
            infer_mode, load_level = (
                self._determine_infer_mode_and_load_level_based_on_objectives(
                    objectives
                )
            )
        else:
            infer_mode, load_level = (
                self._determine_infer_mode_and_load_level_based_on_stimulus()
            )

        return infer_mode, load_level

    def _determine_infer_mode_and_load_level_based_on_objectives(
        self, objectives: ModelObjectiveParameters
    ) -> Tuple[str, str]:
        if "concurrency" in self._config.analyze.sweep_parameters:
            infer_mode = "concurrency"
            load_level = f"{objectives[self._model_name][infer_mode].get_value_based_on_category()}"
        elif "request_rate" in self._config.analyze.sweep_parameters:
            infer_mode = "request_rate"
            load_level = f"{float(objectives[self._model_name][infer_mode].get_value_based_on_category())}"
        elif (
            "input_sequence_length" in self._config.analyze.sweep_parameters
            or "num_dataset_entries" in self._config.analyze.sweep_parameters
            or "batch_size" in self._config.analyze.sweep_parameters
        ):
            infer_mode, load_level = (
                self._determine_infer_mode_and_load_level_based_on_stimulus()
            )
        else:
            raise GenAIPerfException("Cannot determine infer_mode/load_level")

        return infer_mode, load_level

    def _determine_infer_mode_and_load_level_based_on_stimulus(self) -> Tuple[str, str]:
        if "session_concurrency" in self._config.perf_analyzer.stimulus:
            # [TPA-985] Profile export file should have a session concurrency mode
            infer_mode = "request_rate"
            load_level = "0.0"
        # When using fixed schedule mode, infer mode is not set.
        # Setting to default values to avoid an error.
        elif "fixed_schedule" in self._config.perf_analyzer.stimulus:
            infer_mode = "request_rate"
            load_level = "0.0"
        elif "concurrency" in self._config.perf_analyzer.stimulus:
            infer_mode = "concurrency"
            load_level = f'{self._config.perf_analyzer.stimulus["concurrency"]}'
        elif "request_rate" in self._config.perf_analyzer.stimulus:
            infer_mode = "request_rate"
            load_level = f'{self._config.perf_analyzer.stimulus["request_rate"]}'
        else:
            raise GenAIPerfException("Cannot determine infer_mode/load_level")

        return infer_mode, load_level

    def _determine_infer_type(self):
        if "concurrency" in self._config.analyze.sweep_parameters:
            infer_type = "concurrency"
        elif "request_rate" in self._config.analyze.sweep_parameters:
            infer_type = "request_rate"
        else:
            if "request_rate" in self._config.perf_analyzer.stimulus:
                infer_type = "request_rate"
            else:
                infer_type = "concurrency"

        return infer_type

    def _get_num_dataset_entries(self, run_config: RunConfig) -> int:
        if "num_dataset_entries" in self._config.analyze.sweep_parameters:
            num_dataset_entries = run_config.genai_perf_config.get_parameters()[
                "num_dataset_entries"
            ]
        else:
            num_dataset_entries = run_config.genai_perf_config.get_parameters()[
                "input"
            ]["num_dataset_entries"]

        return num_dataset_entries
