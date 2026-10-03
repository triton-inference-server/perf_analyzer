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
from pathlib import Path

import pytest

from genai_perf.plots.plot_config import PlotConfig, PlotType, ProfileRunData
from genai_perf.plots.plot_manager import PlotManager


class TestPlotManager:
    @pytest.fixture
    def box_plot_config(self, tmp_path: Path):
        output_path = tmp_path / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        return PlotConfig(
            title="Test Box Plot",
            data=[ProfileRunData(name="test", x_metric=[1, 2, 3], y_metric=[1, 2, 3])],
            x_label="X Axis",
            y_label="Y Axis",
            height=1000,
            width=1000,
            output=output_path,
            type=PlotType.BOX,
        )

    @pytest.fixture
    def scatter_plot_config(self, tmp_path: Path):
        output_path = tmp_path / "output"
        output_path.mkdir(parents=True, exist_ok=True)
        return PlotConfig(
            title="Test Scatter Plot",
            data=[ProfileRunData(name="test", x_metric=[1, 2, 3], y_metric=[1, 2, 3])],
            x_label="X Axis",
            y_label="Y Axis",
            height=1000,
            width=1000,
            output=output_path,
            type=PlotType.SCATTER,
        )

    @pytest.fixture
    def plot_manager(
        self, box_plot_config: PlotConfig, scatter_plot_config: PlotConfig
    ):
        return PlotManager([box_plot_config, scatter_plot_config])

    def test_plot_manager_generated_plots_are_all_present(
        self,
        box_plot_config: PlotConfig,
        scatter_plot_config: PlotConfig,
        plot_manager: PlotManager,
    ):
        plot_manager.generate_plots()
        assert Path(box_plot_config.output / "test_box_plot.html").exists()
        assert Path(box_plot_config.output / "test_box_plot.jpeg").exists()
        assert Path(scatter_plot_config.output / "test_scatter_plot.html").exists()
        assert Path(scatter_plot_config.output / "test_scatter_plot.jpeg").exists()
