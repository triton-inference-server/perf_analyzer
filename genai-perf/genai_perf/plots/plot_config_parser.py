#!/usr/bin/env python3
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

from pathlib import Path
from typing import List, Union

import genai_perf.logging as logging

# Skip type checking to avoid mypy error
# Issue: https://github.com/python/mypy/issues/10632
import yaml  # type: ignore
from genai_perf.metrics import Statistics
from genai_perf.plots.plot_config import PlotConfig, PlotType, ProfileRunData
from genai_perf.profile_data_parser import LLMProfileDataParser
from genai_perf.tokenizer import (
    DEFAULT_TOKENIZER,
    DEFAULT_TOKENIZER_REVISION,
    get_tokenizer,
)
from genai_perf.utils import load_yaml, scale

logger = logging.getLogger(__name__)


class PlotConfigParser:
    """Parses YAML configuration file to generate PlotConfigs."""

    def __init__(self, filename: Path) -> None:
        self._filename = filename

    def generate_configs(
        self,
        tokenizer: str = DEFAULT_TOKENIZER,
        tokenizer_trust_remote_code: bool = False,
        tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
    ) -> List[PlotConfig]:
        """Load YAML configuration file and convert to PlotConfigs."""
        logger.info(
            f"Generating plot configurations by parsing {self._filename}. "
            "This may take a few seconds.",
        )
        configs = load_yaml(self._filename)

        plot_configs = []
        for _, config in configs.items():
            # Collect profile run data
            profile_data: List[ProfileRunData] = []
            for filepath in config["paths"]:
                stats = self._get_statistics(
                    filepath, tokenizer, tokenizer_trust_remote_code, tokenizer_revision
                )
                profile_data.append(
                    ProfileRunData(
                        name=self._get_run_name(Path(filepath)),
                        x_metric=self._get_metric(stats, config["x_metric"]),
                        y_metric=self._get_metric(stats, config["y_metric"]),
                    )
                )

            plot_configs.append(
                PlotConfig(
                    title=config["title"],
                    data=profile_data,
                    x_label=config["x_label"],
                    y_label=config["y_label"],
                    width=config["width"],
                    height=config["height"],
                    type=self._get_plot_type(config["type"]),
                    output=Path(config["output"]),
                )
            )

        return plot_configs

    def _get_statistics(
        self,
        filepath: str,
        tokenizer: str,
        tokenizer_trust_remote_code: bool = False,
        tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
    ) -> Statistics:
        """Extract a single profile run data."""
        data_parser = LLMProfileDataParser(
            filename=Path(filepath),
            tokenizer=get_tokenizer(
                tokenizer, tokenizer_trust_remote_code, tokenizer_revision
            ),
        )
        load_info = data_parser.get_profile_load_info()

        # TMA-1904: Remove single experiment assumption
        assert len(load_info) == 1
        infer_mode, load_level = load_info[0]
        stats = data_parser.get_statistics(infer_mode, load_level)
        return stats

    def _get_run_name(self, filepath: Path) -> str:
        """Construct a profile run name."""
        if filepath.parent.name:
            return filepath.parent.name + "/" + filepath.stem
        return filepath.stem

    def _get_metric(self, stats: Statistics, name: str) -> List[Union[int, float]]:
        if not name:  # no metric
            return []
        elif name == "inter_token_latencies":
            itls = stats.metrics.data[name]
            return [scale(x, (1 / 1e6)) for x in itls]  # ns to ms
        elif name == "token_positions":
            chunked_itls = getattr(stats.metrics, "_chunked_inter_token_latencies")
            token_positions: List[Union[int, float]] = []
            for request_itls in chunked_itls:
                token_positions += list(range(1, len(request_itls) + 1))
            return token_positions
        elif name == "time_to_first_tokens":
            ttfts = stats.metrics.data[name]
            return [scale(x, (1 / 1e6)) for x in ttfts]  # ns to ms
        elif name == "time_to_second_tokens":
            ttsts = stats.metrics.data[name]
            return [scale(x, (1 / 1e6)) for x in ttsts]  # ns to ms
        elif name == "request_latencies":
            req_latencies = stats.metrics.data[name]
            return [scale(x, (1 / 1e6)) for x in req_latencies]  # ns to ms

        return stats.metrics.data[name]

    def _get_plot_type(self, plot_type: str) -> PlotType:
        """Returns the plot type as PlotType object."""
        if plot_type == "scatter":
            return PlotType.SCATTER
        elif plot_type == "box":
            return PlotType.BOX
        elif plot_type == "heatmap":
            return PlotType.HEATMAP
        else:
            raise ValueError(
                "Unknown plot type encountered while parsing YAML configuration. "
                "Plot type must be either 'scatter', 'box', or 'heatmap'."
            )

    @staticmethod
    def create_init_yaml_config(filenames: List[Path], output_dir: Path) -> None:
        config_str = f"""
        plot1:
          title: Time to First Token
          x_metric: ""
          y_metric: time_to_first_tokens
          x_label: Time to First Token (ms)
          y_label: ""
          width: {1200 if len(filenames) > 1 else 700}
          height: 450
          type: box
          paths: {[str(f) for f in filenames]}
          output: {output_dir}

        plot2:
          title: Request Latency
          x_metric: ""
          y_metric: request_latencies
          x_label: Request Latency (ms)
          y_label: ""
          width: {1200 if len(filenames) > 1 else 700}
          height: 450
          type: box
          paths: {[str(f) for f in filenames]}
          output: {output_dir}

        plot3:
          title: Distribution of Input Sequence Lengths to Output Sequence Lengths
          x_metric: input_sequence_lengths
          y_metric: output_sequence_lengths
          x_label: Input Sequence Length
          y_label: Output Sequence Length
          width: {1200 if len(filenames) > 1 else 700}
          height: 450
          type: heatmap
          paths: {[str(f) for f in filenames]}
          output: {output_dir}

        plot4:
          title: Time to First Token vs Input Sequence Lengths
          x_metric: input_sequence_lengths
          y_metric: time_to_first_tokens
          x_label: Input Sequence Length
          y_label: Time to First Token (ms)
          width: {1200 if len(filenames) > 1 else 700}
          height: 450
          type: scatter
          paths: {[str(f) for f in filenames]}
          output: {output_dir}

        plot5:
          title: Token-to-Token Latency vs Output Token Position
          x_metric: token_positions
          y_metric: inter_token_latencies
          x_label: Output Token Position
          y_label: Token-to-Token Latency (ms)
          width: {1200 if len(filenames) > 1 else 700}
          height: 450
          type: scatter
          paths: {[str(f) for f in filenames]}
          output: {output_dir}
        """

        filepath = output_dir / "config.yaml"
        logger.info(f"Creating initial YAML configuration file to {filepath}")
        config = yaml.safe_load(config_str)
        with open(str(filepath), "w") as f:
            yaml.dump(config, f, sort_keys=False)
