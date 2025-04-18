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


from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.inputs.converters import ImageRetrievalConverter
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.retrievers.generic_dataset import (
    DataRow,
    FileData,
    GenericDataset,
)


class TestImageRetrievalConverter:
    @staticmethod
    def create_generic_dataset() -> GenericDataset:
        return GenericDataset(
            files_data={
                "file1": FileData(
                    rows=[
                        DataRow(images=["test_image_1", "test_image_2"]),
                    ],
                )
            }
        )

    @staticmethod
    def create_generic_dataset_with_payload_parameters() -> GenericDataset:
        optional_data = {"session_id": "abcd"}
        return GenericDataset(
            files_data={
                "file1": FileData(
                    rows=[
                        DataRow(
                            images=["test_image_1", "test_image_2"],
                            optional_data=optional_data,
                            payload_metadata={"timestamp": 0},
                        )
                    ]
                )
            }
        )

    def test_convert_default(self):
        """
        Test Image Retrieval request payload
        """
        generic_dataset = self.create_generic_dataset()

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.output_format = OutputFormat.IMAGE_RETRIEVAL

        image_retrieval_converter = ImageRetrievalConverter(config)
        result = image_retrieval_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": [
                                {"type": "image_url", "url": "test_image_1"},
                                {"type": "image_url", "url": "test_image_2"},
                            ]
                        }
                    ]
                },
            ]
        }

        assert result == expected_result

    def test_convert_with_payload_parameters(self):
        generic_dataset = self.create_generic_dataset_with_payload_parameters()

        config = ConfigCommand({"model_name": "test_model"})
        config.endpoint.output_format = OutputFormat.IMAGE_RETRIEVAL

        image_retrieval_converter = ImageRetrievalConverter(config)
        result = image_retrieval_converter.convert(generic_dataset)

        expected_result = {
            "data": [
                {
                    "payload": [
                        {
                            "input": [
                                {"type": "image_url", "url": "test_image_1"},
                                {"type": "image_url", "url": "test_image_2"},
                            ],
                            "session_id": "abcd",
                        }
                    ],
                    "timestamp": [0],
                },
            ]
        }

        assert result == expected_result
