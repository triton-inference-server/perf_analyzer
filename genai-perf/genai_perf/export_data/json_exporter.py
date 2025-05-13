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


import json
import os
from enum import Enum
from typing import Dict, Union

import genai_perf.logging as logging
from genai_perf.export_data import telemetry_data_exporter_util as telem_utils
from genai_perf.export_data.exporter_config import ExporterConfig

logger = logging.getLogger(__name__)


class JsonExporter:
    """
    A class to export the statistics and arg values in a json format.
    """

    def __init__(self, config: ExporterConfig) -> None:
        self._stats: Dict = config.stats
        self._telemetry_stats: Dict[str, Dict[str, Union[str, Dict[str, float]]]] = (
            config.telemetry_stats
        )
        self._session_stats: Dict = config.session_stats
        self._config = config.config
        self._args = self._config.to_json_dict()
        self._output_dir = config.perf_analyzer_config.get_artifact_directory()
        self._export_data: Dict = {}

        self._merge_stats_and_args()
        self._add_session_stats()

    def export(self) -> None:
        prefix = os.path.splitext(
            os.path.basename(self._config.output.profile_export_file)
        )[0]
        filename = self._output_dir / f"{prefix}_genai_perf.json"
        logger.info(f"Generating {filename}")
        with open(str(filename), "w") as f:
            f.write(json.dumps(self._export_data, indent=2))

    def _merge_stats_and_args(self) -> None:
        self._export_data = dict(self._stats)
        telem_utils.merge_telemetry_stats_json(self._telemetry_stats, self._export_data)
        self._export_data.update({"input_config": self._args})

    def _add_session_stats(self) -> None:
        if self._session_stats:
            self._export_data.update({"sessions": self._session_stats})
