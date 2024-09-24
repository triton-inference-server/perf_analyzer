# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import unittest
from unittest.mock import patch

from genai_perf.measurements.model_config_measurement import (
    ModelConfigMeasurement,
    ModelConfigMeasurementDefaults,
)
from genai_perf.record.types.perf_latency_p99 import PerfLatencyP99
from genai_perf.record.types.perf_throughput import PerfThroughput
from genai_perf.record.types.time_to_first_token_avg import TimeToFirstTokenAvg
from genai_perf.utils import checkpoint_encoder


class TestModelConfigMeasurement(unittest.TestCase):
    ###########################################################################
    # Setup & Teardown
    ###########################################################################
    def setUp(self):
        NotImplemented

    def tearDown(self):
        patch.stopall()


if __name__ == "__main__":
    unittest.main()
