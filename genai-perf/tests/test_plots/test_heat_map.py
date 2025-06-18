import pytest
import numpy as np
from genai_perf.plots.heat_map import HeatMap
from genai_perf.plots.plot_config import PlotConfig


class TestHeatMap:
    @pytest.fixture
    def heat_map_data(self):
        return {
            "data": np.random.rand(5, 5),
            "x_labels": ["A", "B", "C", "D", "E"],
            "y_labels": ["1", "2", "3", "4", "5"],
        }

    @pytest.fixture
    def heat_map_config(self):
        return PlotConfig(
            title="Heat Map Test",
            x_label="X Axis",
            y_label="Y Axis",
            show_grid=True,
            color_map="viridis",
        )

    def test_heat_map_creation(self, heat_map_data, heat_map_config):
        plot = HeatMap(heat_map_data, heat_map_config)
        assert plot.data == heat_map_data
        assert plot.config == heat_map_config

    def test_heat_map_data_validation(self):
        with pytest.raises(ValueError):
            HeatMap({"data": np.array([])})

    def test_heat_map_labels_mismatch(self):
        with pytest.raises(ValueError):
            HeatMap(
                {
                    "data": np.random.rand(3, 3),
                    "x_labels": ["A", "B"],  # Missing label
                    "y_labels": ["1", "2", "3"],
                }
            )

    def test_heat_map_invalid_color_map(self):
        with pytest.raises(ValueError):
            config = PlotConfig(color_map="invalid_map")
            HeatMap(
                {
                    "data": np.random.rand(3, 3),
                    "x_labels": ["A", "B", "C"],
                    "y_labels": ["1", "2", "3"],
                },
                config,
            )
