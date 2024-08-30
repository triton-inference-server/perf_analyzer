#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import time
from abc import ABC, abstractmethod
from threading import Event, Thread
from typing import Optional

import requests
from genai_perf.metrics import TelemetryMetrics, TelemetryStatistics


class TelemetryDataCollector(ABC):
    def __init__(
        self, server_metrics_url: str, collection_interval: float = 1.0  # in seconds
    ) -> None:
        self._server_metrics_url = server_metrics_url
        self._collection_interval = collection_interval
        self._metrics = TelemetryMetrics()
        self._stop_event = Event()
        self._thread: Optional[Thread] = None

    def is_url_reachable(self) -> bool:
        timeout_seconds = 5
        if self._server_metrics_url:
            try:
                response = requests.get(
                    self._server_metrics_url, timeout=timeout_seconds
                )
                return response.status_code == requests.codes.ok
            except requests.RequestException:
                return False
        return True

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = Thread(target=self._collect_metrics)
            self._thread.start()

    def stop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()

    def get_statistics(self) -> TelemetryStatistics:
        telemetry_stats = TelemetryStatistics(self._metrics)
        return telemetry_stats

    def _fetch_metrics(self) -> str:
        response = requests.get(self._server_metrics_url)
        response.raise_for_status()
        return response.text

    @abstractmethod
    def _process_and_update_metrics(self, metrics_data: str) -> None:
        """This method should be implemented by subclasses."""
        pass

    def _collect_metrics(self) -> None:
        """Continuously collect telemetry metrics for every second"""
        while not self._stop_event.is_set():
            metrics_data = self._fetch_metrics()
            self._process_and_update_metrics(metrics_data)
            time.sleep(self._collection_interval)

    @property
    def metrics(self) -> TelemetryMetrics:
        return self._metrics

    @property
    def metrics_url(self) -> str:
        return self._server_metrics_url
