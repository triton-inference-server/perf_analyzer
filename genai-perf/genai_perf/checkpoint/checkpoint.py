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
import os
from dataclasses import dataclass

from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.run.results import Results
from genai_perf.exceptions import GenAIPerfException
from genai_perf.types import CheckpointObject


@dataclass(frozen=True)
class CheckpointDefaults:
    FILENAME = "checkpoint.json"


@dataclass
class Checkpoint:
    """
    Contains the methods necessary for reading and writing GenAI-Perf
    state to a file so that stateful subcommands (such as Optimize or
    Analyze) can resume or continue (ex: running Analyze then Visualize)
    """

    config: ConfigCommand

    # Every top-level class that needs to store state is passed in
    results: Results

    def __post_init__(self):
        self._create_class_from_checkpoint()

    ###########################################################################
    # Read/Write Methods
    ###########################################################################
    def create_checkpoint_object(self) -> None:
        state_dict = {"Results": self.results.create_checkpoint_object()}

        checkpoint_file_path = self._create_checkpoint_file_path()
        with open(checkpoint_file_path, "w") as checkpoint_file:
            json.dump(state_dict, checkpoint_file, default=checkpoint_encoder)

    def _create_class_from_checkpoint(self) -> None:
        checkpoint_file_path = self._create_checkpoint_file_path()

        if os.path.isfile(checkpoint_file_path):
            self.checkpoint_exists = True
            try:
                with open(checkpoint_file_path, "r") as checkpoint_file:
                    checkpoint_json = json.load(checkpoint_file)
                    self._state: CheckpointObject = {
                        "Results": Results.create_class_from_checkpoint(
                            checkpoint_json["Results"]
                        )
                    }

            except EOFError:
                raise (
                    GenAIPerfException(
                        f"Checkpoint file {checkpoint_file} is"
                        " empty or corrupted. Delete it and rerun GAP"
                    )
                )
        else:
            self.checkpoint_exists = False
            self._state = {}

    def _create_checkpoint_file_path(self) -> str:
        checkpoint_file_path = os.path.join(
            self.config.checkpoint_directory, CheckpointDefaults.FILENAME
        )

        return checkpoint_file_path


###########################################################################
# Encoder
###########################################################################
def checkpoint_encoder(obj):
    if isinstance(obj, bytes):
        return obj.decode("utf-8")
    elif hasattr(obj, "create_checkpoint_object"):
        return obj.create_checkpoint_object()
    else:
        return obj.__dict__
