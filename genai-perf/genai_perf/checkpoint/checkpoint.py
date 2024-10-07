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

from typing import Any, Dict

from genai_perf.config.input.config_command import ConfigCommand


class Checkpoint:
    """
    Contains the methods necessary for reading and writing GenAI-Perf
    state to a file so that stateful subcommands (such as Optimize or
    Analyze) can resume or continue (ex: running Analyze then Visualize)
    """

    def __init__(self, config: ConfigCommand):
        pass

    ###########################################################################
    # Read/Write Methods
    ###########################################################################
    def write_to_checkpoint(self) -> None:
        pass

    # @classmethod
    # def read_from_checkpoint(cls) -> Dict[str, Any]:
    #     pass
