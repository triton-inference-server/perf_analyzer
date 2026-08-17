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

# Skip type checking to avoid mypy error
# Issue: https://github.com/python/mypy/issues/10632
import yaml  # type: ignore
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.plots.plot_config import PlotType
from genai_perf.plots.plot_config_parser import PlotConfigParser
import io
import tempfile
import shutil
import os
import pytest
from unittest import mock


class TestPlotConfigParser:
    yaml_config = """
    plot1:
      title: TTFT vs ITL
      x_metric: time_to_first_tokens
      y_metric: inter_token_latencies
      x_label: TTFT (ms)
      y_label: ITL (ms)
      width: 1000
      height: 3000
      type: box
      paths:
        - run1/concurrency32.json
        - run2/concurrency32.json
        - run3/concurrency32.json
      output: test_output_1

    plot2:
      title: Input Sequence Length vs Output Sequence Length
      x_metric: input_sequence_lengths
      y_metric: output_sequence_lengths
      x_label: Input Sequence Length
      y_label: Output Sequence Length
      width: 1234
      height: 5678
      type: scatter
      paths:
        - run4/concurrency1.json
      output: test_output_2
    """

    def test_generate_configs(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "genai_perf.plots.plot_config_parser.load_yaml",
            lambda _: yaml.safe_load(self.yaml_config),
        )
        monkeypatch.setattr(PlotConfigParser, "_get_statistics", lambda *_: {})
        monkeypatch.setattr(PlotConfigParser, "_get_metric", lambda *_: [1, 2, 3])

        config_parser = PlotConfigParser(Path("test_config.yaml"))
        config = ConfigCommand({"model_name": "test_model"})
        plot_configs = config_parser.generate_configs(config)

        assert len(plot_configs) == 2
        pc1, pc2 = plot_configs

        # plot config 1
        assert pc1.title == "TTFT vs ITL"
        assert pc1.x_label == "TTFT (ms)"
        assert pc1.y_label == "ITL (ms)"
        assert pc1.width == 1000
        assert pc1.height == 3000
        assert pc1.type == PlotType.BOX
        assert pc1.output == Path("test_output_1")

        assert len(pc1.data) == 3  # profile run data
        prd1, prd2, prd3 = pc1.data
        assert prd1.name == "run1/concurrency32"
        assert prd2.name == "run2/concurrency32"
        assert prd3.name == "run3/concurrency32"
        for prd in pc1.data:
            assert prd.x_metric == [1, 2, 3]
            assert prd.y_metric == [1, 2, 3]

        # plot config 2
        assert pc2.title == "Input Sequence Length vs Output Sequence Length"
        assert pc2.x_label == "Input Sequence Length"
        assert pc2.y_label == "Output Sequence Length"
        assert pc2.width == 1234
        assert pc2.height == 5678
        assert pc2.type == PlotType.SCATTER
        assert pc2.output == Path("test_output_2")

        assert len(pc2.data) == 1  # profile run data
        prd = pc2.data[0]
        assert prd.name == "run4/concurrency1"
        assert prd.x_metric == [1, 2, 3]
        assert prd.y_metric == [1, 2, 3]


class DummyStats:
    def __init__(self, data=None, chunked=None):
        self.metrics = mock.Mock()
        self.metrics.data = data or {}
        if chunked is not None:
            setattr(self.metrics, "_chunked_inter_token_latencies", chunked)


class DummyConfig:
    pass


def test_get_run_name_with_parent():
    parser = PlotConfigParser(Path("/foo/bar/baz.json"))
    assert parser._get_run_name(Path("/foo/bar/baz.json")) == "bar/baz"


def test_get_run_name_without_parent():
    parser = PlotConfigParser(Path("baz.json"))
    assert parser._get_run_name(Path("baz.json")) == "baz"


def test_get_metric_empty():
    parser = PlotConfigParser(Path("dummy"))
    stats = DummyStats({})
    assert parser._get_metric(stats, "") == []


def test_get_metric_inter_token_latencies():
    parser = PlotConfigParser(Path("dummy"))
    stats = DummyStats({"inter_token_latencies": [1000000, 2000000]})
    # Should scale from ns to ms
    result = parser._get_metric(stats, "inter_token_latencies")
    assert result == [1.0, 2.0]


def test_get_metric_token_positions():
    parser = PlotConfigParser(Path("dummy"))
    stats = DummyStats({"token_positions": []}, chunked=[[1, 2, 3], [1, 2]])
    result = parser._get_metric(stats, "token_positions")
    assert result == [1, 2, 3, 1, 2]


def test_get_metric_time_to_first_tokens():
    parser = PlotConfigParser(Path("dummy"))
    stats = DummyStats({"time_to_first_tokens": [1000000, 2000000]})
    result = parser._get_metric(stats, "time_to_first_tokens")
    assert result == [1.0, 2.0]


def test_get_metric_time_to_second_tokens():
    parser = PlotConfigParser(Path("dummy"))
    stats = DummyStats({"time_to_second_tokens": [1000000, 3000000]})
    result = parser._get_metric(stats, "time_to_second_tokens")
    assert result == [1.0, 3.0]


def test_get_metric_request_latencies():
    parser = PlotConfigParser(Path("dummy"))
    stats = DummyStats({"request_latencies": [1000000, 4000000]})
    result = parser._get_metric(stats, "request_latencies")
    assert result == [1.0, 4.0]


def test_get_metric_fallback():
    parser = PlotConfigParser(Path("dummy"))
    stats = DummyStats({"foo": [42, 43]})
    result = parser._get_metric(stats, "foo")
    assert result == [42, 43]


def test_get_plot_type_valid():
    parser = PlotConfigParser(Path("dummy"))
    assert parser._get_plot_type("scatter") == PlotType.SCATTER
    assert parser._get_plot_type("box") == PlotType.BOX
    assert parser._get_plot_type("heatmap") == PlotType.HEATMAP


def test_get_plot_type_invalid():
    parser = PlotConfigParser(Path("dummy"))
    with pytest.raises(ValueError):
        parser._get_plot_type("invalid_type")


def test_get_statistics_calls_parser(monkeypatch):
    parser = PlotConfigParser(Path("dummy"))
    dummy_stats = object()
    dummy_parser = mock.Mock()
    dummy_parser.get_profile_load_info.return_value = [("mode", "level")]
    dummy_parser.get_statistics.return_value = dummy_stats
    monkeypatch.setattr(
        "genai_perf.plots.plot_config_parser.LLMProfileDataParser",
        lambda **kwargs: dummy_parser,
    )
    monkeypatch.setattr(
        "genai_perf.plots.plot_config_parser.get_tokenizer", lambda config: "tok"
    )
    result = parser._get_statistics("foo.json", DummyConfig())
    assert result is dummy_stats
    dummy_parser.get_profile_load_info.assert_called_once()
    dummy_parser.get_statistics.assert_called_once_with("mode", "level")


def test_get_statistics_assert(monkeypatch):
    parser = PlotConfigParser(Path("dummy"))
    dummy_parser = mock.Mock()
    dummy_parser.get_profile_load_info.return_value = []
    monkeypatch.setattr(
        "genai_perf.plots.plot_config_parser.LLMProfileDataParser",
        lambda **kwargs: dummy_parser,
    )
    monkeypatch.setattr(
        "genai_perf.plots.plot_config_parser.get_tokenizer", lambda config: "tok"
    )
    with pytest.raises(AssertionError):
        parser._get_statistics("foo.json", DummyConfig())


def test_create_init_yaml_config(tmp_path):
    files = [tmp_path / "a.json", tmp_path / "b.json"]
    for f in files:
        f.write_text("{}")
    output_dir = tmp_path
    PlotConfigParser.create_init_yaml_config(files, output_dir)
    config_path = output_dir / "config.yaml"
    assert config_path.exists()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Check structure and some keys
    assert "plot1" in config
    assert config["plot1"]["title"] == "Time to First Token"
    assert config["plot1"]["paths"] == [str(f) for f in files]
    assert config["plot1"]["output"] == str(output_dir)
    assert "plot5" in config
    assert config["plot5"]["type"] == "scatter"


def test_generate_configs_integration(monkeypatch, tmp_path):
    # Create a minimal YAML config file
    yaml_config = {
        "plot1": {
            "title": "Test Plot",
            "x_metric": "foo",
            "y_metric": "bar",
            "x_label": "X",
            "y_label": "Y",
            "width": 100,
            "height": 200,
            "type": "scatter",
            "paths": [str(tmp_path / "run1.json")],
            "output": str(tmp_path),
        }
    }
    config_path = tmp_path / "test.yaml"
    with open(config_path, "w") as f:
        yaml.dump(yaml_config, f)
    # Patch dependencies
    monkeypatch.setattr(
        "genai_perf.plots.plot_config_parser.load_yaml", lambda _: yaml_config
    )
    monkeypatch.setattr(PlotConfigParser, "_get_statistics", lambda *_: None)
    monkeypatch.setattr(PlotConfigParser, "_get_metric", lambda *_: [1, 2])
    parser = PlotConfigParser(config_path)
    config = ConfigCommand({"model_name": "test_model"})
    plot_configs = parser.generate_configs(config)
    assert len(plot_configs) == 1
    pc = plot_configs[0]
    assert pc.title == "Test Plot"
    assert pc.x_label == "X"
    assert pc.y_label == "Y"
    assert pc.width == 100
    assert pc.height == 200
    assert pc.type == PlotType.SCATTER
    assert pc.output == tmp_path
    assert len(pc.data) == 1
    prd = pc.data[0]
    assert prd.x_metric == [1, 2]
    assert prd.y_metric == [1, 2]
