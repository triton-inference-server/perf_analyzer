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
