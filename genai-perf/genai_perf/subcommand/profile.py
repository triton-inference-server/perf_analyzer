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

from typing import List, Optional

from genai_perf.config.generate.perf_analyzer_config import PerfAnalyzerConfig
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.export_data.output_reporter import OutputReporter
from genai_perf.plots.plot_config_parser import PlotConfigParser
from genai_perf.plots.plot_manager import PlotManager
from genai_perf.subcommand.subcommand import Subcommand


###########################################################################
# Profile Handler
###########################################################################
def profile_handler(config: ConfigCommand, extra_args: Optional[List[str]]) -> None:
    """
    Handles `profile` subcommand workflow
    """
    profile = Profile(config, extra_args)
    profile.profile()

    if config.output.generate_plots:
        profile.create_plots()


###########################################################################
# Profile Class
###########################################################################
class Profile(Subcommand):
    """
    Contains all the methods needed to run the profile subcommand
    """

    def __init__(self, config: ConfigCommand, extra_args: Optional[List[str]]) -> None:
        super().__init__(config, extra_args)

    def profile(self) -> None:
        """
        Profiles the model based on the user's stimulus
        """
        objectives = self._create_objectives_based_on_stimulus()
        genai_perf_config = self._create_genai_perf_config(objectives)
        perf_analyzer_config = self._create_perf_analyzer_config(objectives)

        if self._is_config_present_in_results(genai_perf_config, perf_analyzer_config):
            self._found_config_in_checkpoint(
                genai_perf_config, perf_analyzer_config, objectives
            )
        else:
            # Pre-amble
            self._create_tokenizer()
            self._create_artifact_directory(perf_analyzer_config)
            self._create_plot_directory(perf_analyzer_config)
            self._generate_inputs(perf_analyzer_config)

            # Profile using Perf Analyzer
            self._run_perf_analyzer(perf_analyzer_config)

            # Post-amble
            self._set_data_parser(perf_analyzer_config)
            self._add_results_to_checkpoint(
                genai_perf_config, perf_analyzer_config, objectives
            )
            self._add_output_to_artifact_directory(perf_analyzer_config, objectives)

    def create_plots(self) -> None:
        # TMA-1911: support plots CLI option
        plot_dir = self._config.output.artifact_directory / "plots"
        PlotConfigParser.create_init_yaml_config(
            filenames=[self._config.output.profile_export_file],  # single run
            output_dir=plot_dir,
        )
        config_parser = PlotConfigParser(plot_dir / "config.yaml")
        plot_configs = config_parser.generate_configs(self._config)
        plot_manager = PlotManager(plot_configs)
        plot_manager.generate_plots()
