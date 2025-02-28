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


import uuid
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
from genai_perf.utils import sample_bounded_normal_int


class SyntheticDataRetriever(BaseInputRetriever):
    def retrieve_data(self) -> GenericDataset:
        files = self.config.synthetic_input_filenames or [DEFAULT_SYNTHETIC_FILENAME]
        synthetic_dataset = GenericDataset(files_data={})

        use_prefix_prompts = self.config.num_prefix_prompts > 0
        if use_prefix_prompts:
            self._initialize_prefix_prompts()

        for file in files:
            data_rows = self._generate_data_rows(use_prefix_prompts)
            synthetic_dataset.files_data[file] = FileData(data_rows)

        return synthetic_dataset

    def _initialize_prefix_prompts(self) -> None:
        SyntheticPromptGenerator.create_prefix_prompts_pool(
            self.config.tokenizer,
            self.config.num_prefix_prompts,
            self.config.prefix_prompt_length,
        )

    def _generate_data_rows(self, use_prefix_prompts: bool) -> List[DataRow]:
        if self.config.num_sessions > 0:
            return self._generate_multi_turn_sessions(use_prefix_prompts)
        return self._generate_stateless_entries(use_prefix_prompts)

    def _generate_multi_turn_sessions(self, use_prefix_prompts: bool) -> List[DataRow]:
        data_rows = []

        for _ in range(self.config.num_sessions):
            num_turns = sample_bounded_normal_int(
                self.config.session_turns_mean,
                self.config.session_turns_stddev,
                lower=1,
            )
            session_id = str(uuid.uuid4())

            session_delay = 0
            for turn_idx in range(num_turns):
                is_first_turn = turn_idx == 0
                row = self._create_data_row(
                    use_prefix_prompts and is_first_turn, session_id
                )

                if turn_idx < num_turns - 1:
                    session_delay = sample_bounded_normal_int(
                        self.config.session_turn_delay_mean,
                        self.config.session_turn_delay_stddev,
                        lower=0,
                    )
                    row.payload_metadata["delay"] = session_delay

                data_rows.append(row)

        return data_rows

    def _generate_stateless_entries(self, use_prefix_prompts: bool) -> List[DataRow]:
        data_rows = []

        for _ in range(self.config.num_dataset_entries):
            row = self._create_data_row(use_prefix_prompts)
            row.images = [
                self._generate_image() for _ in range(self.config.batch_size_image)
            ]
            data_rows.append(row)

        return data_rows

    def _create_data_row(
        self, use_prefix_prompts: bool, session_id: str = ""
    ) -> DataRow:
        row = DataRow(texts=[], images=[], optional_data={}, payload_metadata={})

        if session_id:
            row.optional_data["session_id"] = session_id

        row.texts = [
            self._generate_prompt(use_prefix_prompts)
            for _ in range(self.config.batch_size_text)
        ]
        return row

    def _generate_prompt(self, use_prefix_prompts: bool) -> str:
        prompt = SyntheticPromptGenerator.create_synthetic_prompt(
            self.config.tokenizer,
            self.config.prompt_tokens_mean,
            self.config.prompt_tokens_stddev,
        )
        if use_prefix_prompts:
            prefix_prompt = SyntheticPromptGenerator.get_random_prefix_prompt()
            return f"{prefix_prompt} {prompt}"
        return prompt

    def _generate_image(self):
        return SyntheticImageGenerator.create_synthetic_image(
            image_width_mean=self.config.image_width_mean,
            image_width_stddev=self.config.image_width_stddev,
            image_height_mean=self.config.image_height_mean,
            image_height_stddev=self.config.image_height_stddev,
            image_format=self.config.image_format,
        )
