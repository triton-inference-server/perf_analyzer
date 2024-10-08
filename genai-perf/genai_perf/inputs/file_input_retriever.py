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

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from genai_perf.inputs.input_constants import DEFAULT_BATCH_SIZE, OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.utils import load_json_str


class FileInputRetriever:
    """
    A input retriever class that handles input data provided by the user through
    file and directories.
    """

    def __init__(self, config: InputsConfig) -> None:
        self.config = config

    # TODO: match return type to retriever interface
    def retrieve_data(self) -> Dict[str, Any]:
        if self.config.output_format == OutputFormat.RANKINGS:
            queries_filename = self.config.input_filename / "queries.jsonl"
            passages_filename = self.config.input_filename / "passages.jsonl"
            return self._read_rankings_input_files(queries_filename, passages_filename)
        else:
            return self._get_input_dataset_from_file()

    def _read_rankings_input_files(
        self,
        queries_filename: Path,
        passages_filename: Path,
    ) -> Dict[str, Any]:

        def __key_exists(line: Dict):
            """Validation function that checks if 'text' key exists."""
            if "text" not in line:
                raise ValueError("Each data entry must have 'text' key name.")
            return line

        with open(queries_filename, "r") as file:
            queries = [load_json_str(line, func=__key_exists) for line in file]

        with open(passages_filename, "r") as file:
            passages = [load_json_str(line, func=__key_exists) for line in file]

        if len(queries) < 1:
            raise ValueError("Queries file must have at least one entry.")
        if len(passages) < 1:
            raise ValueError("Passages file must have at least one entry.")
        if self.config.batch_size_text > len(passages):
            raise ValueError(
                "Batch size cannot be larger than the number of available passages"
            )

        dataset_json: Dict[str, Any] = {}
        dataset_json["features"] = [{"name": "input"}]
        dataset_json["rows"] = []

        for _ in range(self.config.num_prompts):
            data = {
                "query": random.choice(queries),
                "passages": random.sample(passages, self.config.batch_size_text),
            }
            dataset_json["rows"].append({"row": data})
        return dataset_json

    def _get_input_dataset_from_file(self) -> Dict[str, Any]:
        """
        Returns
        -------
        Dict
            The dataset in the required format with the prompts and/or images
            read from the file.
        """
        self._verify_file()
        prompts, images = self._get_prompts_from_input_file()
        if self.config.batch_size_image > len(images):
            raise ValueError(
                "Batch size for images cannot be larger than the number of available images"
            )
        if self.config.batch_size_text > len(prompts):
            raise ValueError(
                "Batch size for texts cannot be larger than the number of available texts"
            )

        dataset_json: Dict[str, Any] = {}
        dataset_json["features"] = [{"name": "text"}]
        dataset_json["rows"] = []

        if (
            self.config.batch_size_text == DEFAULT_BATCH_SIZE
            and self.config.batch_size_image == DEFAULT_BATCH_SIZE
        ):
            for prompt, image in zip(prompts, images):
                content = {}
                if prompt is not None:
                    content["text"] = prompt
                if image is not None:
                    content["image"] = image
                dataset_json["rows"].append({"row": content})
        else:
            for _ in range(self.config.num_prompts):
                content_array = []
                sampled_image_indices = random.sample(
                    range(len(prompts)), self.config.batch_size_image
                )
                sampled_text_indices = random.sample(
                    range(len(prompts)), self.config.batch_size_text
                )

                sampled_images = [images[i] for i in sampled_image_indices]
                sampled_texts = [prompts[i] for i in sampled_text_indices]

                max_samples = max(len(sampled_texts), len(sampled_images))
                num_sampled_images = len(sampled_images)
                num_sampled_texts = len(sampled_texts)

                for i in range(max_samples):
                    content = {}
                    if i < num_sampled_images:
                        content["image"] = sampled_images[i]
                    if i < num_sampled_texts:
                        content["text"] = sampled_texts[i]

                    content_array.append(content)

                dataset_json["rows"].append({"row": content_array})

        return dataset_json

    def _verify_file(self) -> None:
        if not self.config.input_filename.exists():
            raise FileNotFoundError(
                f"The file '{self.config.input_filename}' does not exist."
            )

    def _get_prompts_from_input_file(self) -> Tuple[List[str], List[str]]:
        """
        Reads the input prompts from a JSONL file and returns a list of prompts.

        Returns
        -------
        Tuple[List[str], List[str]]
            A list of prompts and images read from the file.
        """
        prompts = []
        images = []
        with open(self.config.input_filename, mode="r", newline=None) as file:
            for line in file:
                if line.strip():
                    data = load_json_str(line)
                    # None if not provided
                    prompt = data.get("text")
                    prompt_alt = data.get("text_input")
                    if prompt and prompt_alt:
                        raise ValueError(
                            "Each data entry must have only one of 'text_input' or 'text' key name."
                        )
                    prompt = prompt if prompt else prompt_alt
                    image = data.get("image")
                    prompts.append(prompt.strip() if prompt else prompt)
                    images.append(image.strip() if image else image)
        return prompts, images
