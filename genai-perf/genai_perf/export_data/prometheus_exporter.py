# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict

from genai_perf.export_data.exporter_config import ExporterConfig
from genai_perf.logging import logging
from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """
    Exports metrics in a Prometheus-compatible format via an HTTP endpoint.
    """

    def __init__(self, config: ExporterConfig):
        self._stats = config.stats
        self._metrics = config.metrics
        self._args = config.args
        if not hasattr(self._args, "prometheus_port"):
            raise ValueError("Prometheus port is not provided.")
        self._port = (
            self._args.prometheus_port
            if hasattr(self._args, "prometheus_port")
            else 8000
        )

        self._prom_metrics: Dict[str, Any] = {}
        self._initialize_metrics()

    def _initialize_metrics(self):
        """
        Initialize Prometheus metrics based on available stats and metrics.
        """
        for metric in self._metrics.request_metrics and self._metrics.system_metrics:
            metric_name = metric.name.replace(".", "_")
            self._prom_metrics[metric_name] = Gauge(
                metric_name,
                f"Metric: {metric_name} ({metric.unit})",
            )

    def export(self) -> None:
        """
        Start Prometheus HTTP server and expose metrics.
        """
        start_http_server(self._port)
        self._update_metrics()
        logger.info(
            f"Prometheus metrics available at http://localhost:{self._port}/metrics"
        )

    def _update_metrics(self):
        """
        Update Prometheus metrics with current stats.
        """
        for metric_name, prom_metric in self._prom_metrics.items():
            if metric_name in self._stats:
                prom_metric.set(self._stats[metric_name]["avg"])
