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


from typing import Any, Dict, List

from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.synthetic_image_generator import SyntheticImageGenerator
from genai_perf.inputs.synthetic_prompt_generator import SyntheticPromptGenerator


class SyntheticDataRetriever:
    """
    A data retriever class that handles generation of synthetic data.
    """

    def __init__(self, config):
        self.config = config

    def retrieve_data(self) -> List[Dict[str, Any]]:
        synthetic_dataset = []
        for _ in range(self.config.num_prompts):
            prompt = SyntheticPromptGenerator.create_synthetic_prompt(
                self.config.tokenizer,
                self.config.prompt_tokens_mean,
                self.config.prompt_tokens_stddev,
            )
            data = {"text_input": prompt}

            if self.config.output_format == OutputFormat.OPENAI_VISION:
                image = SyntheticImageGenerator.create_synthetic_image(
                    image_width_mean=self.config.image_width_mean,
                    image_width_stddev=self.config.image_width_stddev,
                    image_height_mean=self.config.image_height_mean,
                    image_height_stddev=self.config.image_height_stddev,
                    image_format=self.config.image_format,
                )
                data["image"] = image

            synthetic_dataset.append(data)
        return synthetic_dataset
