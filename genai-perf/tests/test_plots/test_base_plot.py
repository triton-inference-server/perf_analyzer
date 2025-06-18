import pytest
from genai_perf.plots.base_plot import BasePlot
from genai_perf.plots.plot_config import PlotConfig, PlotType


class TestBasePlot:
    @pytest.fixture
    def sample_data(self):
        return {"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]}

    @pytest.fixture
    def plot_config(self, tmp_path):
        return PlotConfig(
            title="Test Plot",
            x_label="X Axis",
            y_label="Y Axis",
            height=1000,
            width=1000,
            output=tmp_path / "output",
            type=PlotType.BOX,
        )

    def test_base_plot_initialization(self, sample_data, plot_config):
        plot = BasePlot(sample_data, plot_config)
        assert plot.data == sample_data
        assert plot.config == plot_config

    def test_base_plot_validation(self):
        with pytest.raises(ValueError):
            BasePlot(None)

    def test_base_plot_empty_data(self):
        with pytest.raises(ValueError):
            BasePlot({})

    def test_base_plot_missing_required_data(self):
        with pytest.raises(ValueError):
            BasePlot({"x": [1, 2, 3]})  # Missing y data

    def test_base_plot_data_length_mismatch(self):
        with pytest.raises(ValueError):
            BasePlot({"x": [1, 2, 3], "y": [1, 2]})
