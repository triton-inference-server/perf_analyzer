#!/usr/bin/env python3
# mypy: ignore-errors
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


import os
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

# Global constants
PA_ABSLT_PATH = os.environ.get("PA_ABSLT_PATH", os.getcwd())
PA_TEMPLATES_ABSLT_PATH = os.path.join(PA_ABSLT_PATH, "templates")

# Change working directory to perf_analyzer/templates.
os.chdir(PA_TEMPLATES_ABSLT_PATH)

# read the yaml file
with open("template_vars.yaml") as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# create the jinja2 environment
env = Environment(loader=FileSystemLoader("."), autoescape=True)
for file in data.keys():
    if "template" not in data[file]:
        continue

    template = env.get_template(data[file]["template"])
    file_vars = data["General"].copy()

    if file in data:
        file_vars.update(data[file])

    # render the template with the data and print the output
    output = template.render(file_vars)

    # grab the path to the output directory
    output_dir = os.path.join(
        Path(data[file]["output_dir"]), Path(data[file]["filename"])
    )

    # write the output to a file
    with open(output_dir, "w") as file:
        file.write(output)
    file_vars.clear()

# Change working directory to perf_analyzer.
os.chdir(PA_ABSLT_PATH)
