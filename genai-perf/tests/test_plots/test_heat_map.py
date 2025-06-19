from unittest.mock import patch

import pytest

from genai_perf.plots.exceptions import EmptyDataError
from genai_perf.plots.heat_map import HeatMap
from genai_perf.plots.plot_config import ProfileRunData


@pytest.fixture
def profile_run_data_list():
    return [
        ProfileRunData(name="run1", x_metric=[1, 2, 3], y_metric=[4, 5, 6]),
        ProfileRunData(name="run2", x_metric=[1, 2, 3], y_metric=[7, 8, 9]),
    ]


def test_heat_map_init(profile_run_data_list):
    plot = HeatMap(profile_run_data_list)
    assert plot._profile_data == profile_run_data_list


@patch("genai_perf.plots.base_plot.BasePlot._generate_parquet")
@patch("genai_perf.plots.base_plot.BasePlot._generate_graph_file")
def test_heat_map_create_plot(
    mock_gen_graph, mock_gen_parquet, profile_run_data_list, tmp_path
):
    plot = HeatMap(profile_run_data_list)
    plot.create_plot(
        graph_title="Test Title",
        x_label="X",
        y_label="Y",
        width=800,
        height=600,
        filename_root="testfile",
        output_dir=tmp_path,
    )
    assert mock_gen_parquet.called
    assert mock_gen_graph.call_count == 2
    html_call = [c for c in mock_gen_graph.call_args_list if c[0][2].endswith(".html")]
    jpeg_call = [c for c in mock_gen_graph.call_args_list if c[0][2].endswith(".jpeg")]
    assert html_call and jpeg_call


def test_heat_map_create_dataframe(profile_run_data_list):
    plot = HeatMap(profile_run_data_list)
    df = plot._create_dataframe("X", "Y")
    assert list(df.columns) == ["X", "Y", "Run Name"]
    assert len(df) == 2
    assert df["Run Name"].tolist() == ["run1", "run2"]
    assert df["Y"].tolist() == [[4, 5, 6], [7, 8, 9]]


@patch("genai_perf.plots.base_plot.BasePlot._generate_parquet")
@patch("genai_perf.plots.base_plot.BasePlot._generate_graph_file")
def test_heat_map_create_plot_empty_data(mock_gen_graph, mock_gen_parquet, tmp_path):
    with pytest.raises(EmptyDataError) as exc:
        plot = HeatMap([])
        plot.create_plot(
            graph_title="Empty",
            x_label="X",
            y_label="Y",
            width=700,
            height=450,
            filename_root="emptyfile",
            output_dir=tmp_path,
        )
    assert "Data is empty" in str(exc.value)
