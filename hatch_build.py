# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sysconfig

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Tag the perf_analyzer wheel as platform-specific.

    The wheel ships the compiled perf_analyzer executable
    (perf_analyzer/bin/perf_analyzer), staged in by the CMake build rather
    than produced by the build backend. Hatchling therefore sees only data
    files and would emit "py3-none-any", so x86_64 and aarch64 builds collide
    on one filename and pip may install a binary for the wrong architecture.

    The binary is a standalone executable, not a CPython extension, so the
    tag is py3-none-<build platform> (e.g. linux_x86_64) rather than an
    interpreter-specific one. Release builds then run `auditwheel repair`,
    which derives the manylinux tag from the binary's glibc symbol
    requirements.
    """

    def initialize(self, version, build_data):
        if self.target_name != "wheel":
            return
        plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{plat}"
