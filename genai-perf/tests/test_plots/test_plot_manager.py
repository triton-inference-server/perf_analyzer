import pytest
from genai_perf.plots.plot_manager import PlotManager
from genai_perf.plots.box_plot import BoxPlot
from genai_perf.plots.scatter_plot import ScatterPlot
from genai_perf.plots.plot_config import PlotConfig, PlotType


class TestPlotManager:
    @pytest.fixture
    def plot_config(self, tmp_path):
        return PlotConfig(
            title="Test Plot",
            data=[{"x": [1, 2, 3], "y": [1, 2, 3]}],
            x_label="X Axis",
            y_label="Y Axis",
            height=1000,
            width=1000,
            output=tmp_path / "output",
            type=PlotType.BOX,
        )

    @pytest.fixture
    def plot_manager(self, plot_config):
        return PlotManager(plot_config)

    @pytest.fixture
    def sample_plots(self, plot_config):
        return {
            "box": BoxPlot(
                {"data": [[1, 2, 3], [4, 5, 6]], "labels": ["A", "B"]}, plot_config
            ),
            "scatter": ScatterPlot({"x": [1, 2, 3], "y": [1, 2, 3]}, plot_config),
        }

    def test_plot_manager_initialization(self, plot_manager):
        assert plot_manager.plots == {}

    def test_add_plot(self, plot_manager, sample_plots):
        for name, plot in sample_plots.items():
            plot_manager.add_plot(name, plot)
        assert len(plot_manager.plots) == len(sample_plots)

    def test_get_plot(self, plot_manager, sample_plots):
        plot_manager.add_plot("test", sample_plots["box"])
        assert plot_manager.get_plot("test") == sample_plots["box"]

    def test_get_nonexistent_plot(self, plot_manager):
        with pytest.raises(KeyError):
            plot_manager.get_plot("nonexistent")

    def test_remove_plot(self, plot_manager, sample_plots):
        plot_manager.add_plot("test", sample_plots["box"])
        plot_manager.remove_plot("test")
        assert "test" not in plot_manager.plots

    def test_clear_plots(self, plot_manager, sample_plots):
        for name, plot in sample_plots.items():
            plot_manager.add_plot(name, plot)
        plot_manager.clear_plots()
        assert len(plot_manager.plots) == 0

    def test_plot_names(self, plot_manager, sample_plots):
        for name, plot in sample_plots.items():
            plot_manager.add_plot(name, plot)
        assert set(plot_manager.plot_names()) == set(sample_plots.keys())
