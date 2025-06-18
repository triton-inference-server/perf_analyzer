import pytest
from genai_perf.plots.scatter_plot import ScatterPlot
from genai_perf.plots.plot_config import PlotConfig


class TestScatterPlot:
    @pytest.fixture
    def scatter_plot_data(self):
        return {
            "x": [1, 2, 3, 4, 5],
            "y": [10, 20, 30, 40, 50],
            "sizes": [100, 200, 300, 400, 500],
            "colors": ["red", "blue", "green", "yellow", "purple"],
        }

    @pytest.fixture
    def scatter_plot_config(self):
        return PlotConfig(
            title="Scatter Plot Test",
            x_label="X Axis",
            y_label="Y Axis",
            show_grid=True,
            show_legend=True,
        )

    def test_scatter_plot_creation(self, scatter_plot_data, scatter_plot_config):
        plot = ScatterPlot(scatter_plot_data, scatter_plot_config)
        assert plot.data == scatter_plot_data
        assert plot.config == scatter_plot_config

    def test_scatter_plot_minimal_data(self):
        data = {"x": [1, 2, 3], "y": [1, 2, 3]}
        plot = ScatterPlot(data, PlotConfig())
        assert plot.data == data

    def test_scatter_plot_data_validation(self):
        with pytest.raises(ValueError):
            ScatterPlot({"x": [1, 2, 3]})  # Missing y data

    def test_scatter_plot_optional_data_validation(self):
        data = {
            "x": [1, 2, 3],
            "y": [1, 2, 3],
            "sizes": [100, 200],  # Mismatched length
        }
        with pytest.raises(ValueError):
            ScatterPlot(data)
