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

from typing import Dict, List

from genai_perf import utils
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import OutputFormat, PromptSource
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.file_input_retriever import FileInputRetriever
from genai_perf.inputs.retrievers.generic_dataset import GenericDataset
from genai_perf.inputs.retrievers.synthetic_data_retriever import SyntheticDataRetriever
from genai_perf.inputs.retrievers.synthetic_image_generator import ImageFormat
from PIL import Image
from requests import Response


class InputRetrieverFactory:
    def __init__(self, config: InputsConfig):
        self.config = config

    def get_input_data(self) -> GenericDataset:
        """
        Retrieve and convert the dataset based on the input type.

        Returns
        -------
        Dict:
            The generic dataset JSON
        """

        input_data: GenericDataset = None
        if self.config.output_format in [
            OutputFormat.OPENAI_EMBEDDINGS,
            OutputFormat.RANKINGS,
            OutputFormat.IMAGE_RETRIEVAL,
        ]:
            # TODO: remove once the factory fully integrates retrievers
            file_retriever = FileInputRetriever(self.config)
            input_data = file_retriever.retrieve_data()

            if self.config.output_format == OutputFormat.IMAGE_RETRIEVAL:
                input_data = self._encode_images_in_input_dataset(input_data)

        else:
            if self.config.input_type == PromptSource.SYNTHETIC:
                synthetic_retriever = SyntheticDataRetriever(self.config)
                input_data = synthetic_retriever.retrieve_data()
            elif self.config.input_type == PromptSource.FILE:
                # TODO: remove once the factory fully integrates retrievers
                file_retriever = FileInputRetriever(self.config)
                input_data = file_retriever.retrieve_data()
                self._encode_images_in_input_dataset(input_data)
            else:
                raise GenAIPerfException("Input source is not recognized.")

        return input_data

    def _convert_input_synthetic_or_file_dataset_to_generic_json(
        self, dataset: Dict
    ) -> Dict[str, List[Dict]]:
        generic_dataset_json = self._convert_dataset_to_generic_input_json(dataset)

        return generic_dataset_json

    def _encode_images_in_input_dataset(self, input_file_dataset: GenericDataset):
        for file_data in input_file_dataset.files_data.values():
            for row in file_data.rows:
                for i, image in enumerate(row.images):
                    if image is not None:
                        payload = self._encode_image(image)
                        row.images[i] = payload

    def _encode_image(self, filename: str) -> str:
        img = Image.open(filename)
        if img is None:
            raise GenAIPerfException(f"Failed to open image '{filename}'.")
        if img.format is None:
            raise GenAIPerfException(
                f"Failed to determine image format of '{filename}'."
            )

        if img.format.lower() not in utils.get_enum_names(ImageFormat):
            raise GenAIPerfException(
                f"Unsupported image format '{img.format}' of "
                f"the image '{filename}'."
            )

        img_base64 = utils.encode_image(img, img.format)
        payload = f"data:image/{img.format.lower()};base64,{img_base64}"
        return payload

    def _check_for_error_in_json_of_dataset(self, dataset_json: Dict) -> None:
        if "error" in dataset_json:
            raise GenAIPerfException(dataset_json["error"])
