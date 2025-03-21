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
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers import (
    SyntheticAudioGenerator,
    SyntheticImageGenerator,
    SyntheticPromptGenerator,
)
from genai_perf.inputs.retrievers.base_input_retriever import BaseInputRetriever
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)
from genai_perf.utils import sample_bounded_normal_int


class SyntheticDataRetriever(BaseInputRetriever):

    def __init__(self, inputs_config: InputsConfig):
        super().__init__(inputs_config)
        self._include_image: bool = (
            self.config.input.image.width.mean > 0
            and self.config.input.image.height.mean > 0
        )
        self._include_audio: bool = self.config.input.audio.length.mean > 0

    def retrieve_data(self) -> GenericDataset:
        files = self.config.input.synthetic_files or [DEFAULT_SYNTHETIC_FILENAME]
        synthetic_dataset = GenericDataset(files_data={})

        use_prefix_prompts = self.config.input.prefix_prompt.num > 0
        if use_prefix_prompts:
            self._initialize_prefix_prompts()

        for file in files:
            data_rows = self._generate_data_rows(use_prefix_prompts)
            synthetic_dataset.files_data[file] = FileData(data_rows)

        return synthetic_dataset

    def _initialize_prefix_prompts(self) -> None:
        SyntheticPromptGenerator.create_prefix_prompts_pool(
            self.tokenizer,
            self.config.input.prefix_prompt.num,
            self.config.input.prefix_prompt.length,
        )

    def _generate_data_rows(self, use_prefix_prompts: bool) -> List[DataRow]:
        if self.config.input.sessions.num > 0:
            return self._generate_multi_turn_sessions(use_prefix_prompts)
        return self._generate_stateless_entries(use_prefix_prompts)

    def _generate_multi_turn_sessions(self, use_prefix_prompts: bool) -> List[DataRow]:
        data_rows = []

        for _ in range(self.config.input.sessions.num):
            num_turns = sample_bounded_normal_int(
                self.config.input.sessions.turns.mean,
                self.config.input.sessions.turns.stddev,
                lower=1,
            )
            session_id = str(uuid.uuid4())

            session_delay = 0
            for turn_idx in range(num_turns):
                is_first_turn = turn_idx == 0
                row = self._create_data_row(session_id)
                row.texts = self._generate_prompts(use_prefix_prompts and is_first_turn)

                if turn_idx < num_turns - 1:
                    session_delay = sample_bounded_normal_int(
                        self.config.input.sessions.turn_delay.mean,
                        self.config.input.sessions.turn_delay.stddev,
                        lower=0,
                    )
                    row.payload_metadata["delay"] = session_delay

                data_rows.append(row)

        return data_rows

    def _generate_stateless_entries(self, use_prefix_prompts: bool) -> List[DataRow]:
        data_rows = []

        for _ in range(self.config.input.num_dataset_entries):
            row = self._create_data_row()
            row.texts = self._generate_prompts(use_prefix_prompts)
            row.images = self._generate_images()
            row.audios = self._generate_audios()
            data_rows.append(row)

        return data_rows

    def _create_data_row(self, session_id: str = "") -> DataRow:
        row = DataRow()

        if session_id:
            row.payload_metadata["session_id"] = session_id
        return row

    def _generate_prompts(self, use_prefix_prompts: bool) -> List[str]:
        prompts = []
        for _ in range(self.config.input.batch_size):
            prompt = SyntheticPromptGenerator.create_synthetic_prompt(
                self.tokenizer,
                self.config.input.synthetic_tokens.mean,
                self.config.input.synthetic_tokens.stddev,
            )
            if use_prefix_prompts:
                prefix_prompt = SyntheticPromptGenerator.get_random_prefix_prompt()
                prompt = f"{prefix_prompt} {prompt}"

            prompts.append(prompt)
        return prompts

    def _generate_images(self) -> List[str]:
        """
        Generate synthetic images if the image width and height are specified.
        """
        images = []
        if self._include_image:
            for _ in range(self.config.input.image.batch_size):
                images.append(
                    SyntheticImageGenerator.create_synthetic_image(
                        image_width_mean=self.config.input.image.width.mean,
                        image_width_stddev=self.config.input.image.width.stddev,
                        image_height_mean=self.config.input.image.height.mean,
                        image_height_stddev=self.config.input.image.height.stddev,
                        image_format=self.config.input.image.format,
                    )
                )
        return images

    def _generate_audios(self) -> List[str]:
        """
        Generate synthetic audios if the audio length is specified.
        """
        audios = []
        if self._include_audio:
            for _ in range(self.config.input.audio.batch_size):
                audios.append(
                    SyntheticAudioGenerator.create_synthetic_audio(
                        self.config.input.audio
                    )
                )
        return audios
