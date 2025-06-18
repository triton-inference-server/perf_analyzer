import pytest
from genai_perf.plots.box_plot import BoxPlot
from genai_perf.plots.plot_config import PlotConfig


class TestBoxPlot:
    @pytest.fixture
    def box_plot_data(self):
        return {
            "data": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]],
            "labels": ["Group A", "Group B", "Group C"],
        }

    @pytest.fixture
    def box_plot_config(self):
        return PlotConfig(
            title="Box Plot Test",
            x_label="Groups",
            y_label="Values",
            show_grid=True,
            show_outliers=True,
        )

    def test_box_plot_creation(self, box_plot_data, box_plot_config):
        plot = BoxPlot(box_plot_data, box_plot_config)
        assert plot.data == box_plot_data
        assert plot.config == box_plot_config

    def test_box_plot_data_validation(self):
        with pytest.raises(ValueError):
            BoxPlot({"data": [], "labels": []})

    def test_box_plot_labels_mismatch(self):
        with pytest.raises(ValueError):
            BoxPlot(
                {
                    "data": [[1, 2, 3], [4, 5, 6]],
                    "labels": ["Group A"],  # Missing label
                }
            )

    def test_box_plot_empty_data_groups(self):
        with pytest.raises(ValueError):
            BoxPlot({"data": [[], [1, 2, 3]], "labels": ["Group A", "Group B"]})
