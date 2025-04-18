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

from typing import List, Optional

from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.inputs.input_constants import Subcommand
from genai_perf.subcommand.analyze import analyze_handler
from genai_perf.subcommand.process_export_files import process_export_files_handler
from genai_perf.subcommand.profile import profile_handler


###########################################################################
# Config Subcommand Handler
###########################################################################
def config_handler(config: ConfigCommand, extra_args: Optional[List[str]]) -> None:
    """
    Handles `config` subcommand workflow
    """
    if config.subcommand == Subcommand.PROFILE:
        profile_handler(config, extra_args)
    elif config.subcommand == Subcommand.ANALYZE:
        analyze_handler(config, extra_args)
    elif config.subcommand == Subcommand.PROCESS:
        process_export_files_handler(config, extra_args)
    else:
        raise ValueError(f"User Config: {config.subcommand} handler not found.")
