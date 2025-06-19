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
