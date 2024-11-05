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


from typing import List

from genai_perf.inputs.input_constants import DEFAULT_SYNTHETIC_FILENAME
from genai_perf.inputs.retrievers.base_input_retriever import BaseInputRetriever
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.inputs.retrievers.synthetic_image_generator import (
    SyntheticImageGenerator,
)
from genai_perf.inputs.retrievers.synthetic_prompt_generator import (
    SyntheticPromptGenerator,
)


class SyntheticDataRetriever(BaseInputRetriever):
    """
    A data retriever class that handles generation of synthetic data.
    """

    def retrieve_data(self) -> GenericDataset:
        files = self.config.synthetic_input_filenames or [DEFAULT_SYNTHETIC_FILENAME]
        synthetic_dataset = GenericDataset(files_data={})

        for file in files:
            data_rows: List[DataRow] = []

            for _ in range(self.config.num_prompts):
                row = DataRow(texts=[], images=[])
                prompt = SyntheticPromptGenerator.create_synthetic_prompt(
                    self.config.tokenizer,
                    self.config.prompt_tokens_mean,
                    self.config.prompt_tokens_stddev,
                )
                for _ in range(self.config.batch_size_text):
                    row.texts.append(prompt)

                for _ in range(self.config.batch_size_image):
                    image = SyntheticImageGenerator.create_synthetic_image(
                        image_width_mean=self.config.image_width_mean,
                        image_width_stddev=self.config.image_width_stddev,
                        image_height_mean=self.config.image_height_mean,
                        image_height_stddev=self.config.image_height_stddev,
                        image_format=self.config.image_format,
                    )
                    row.images.append(image)

                data_rows.append(row)

            file_data = FileData(file, data_rows)

            synthetic_dataset.files_data[file] = file_data

        return synthetic_dataset
