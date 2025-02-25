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


from pathlib import Path
from typing import Any, Dict, Tuple, Union

from genai_perf.inputs.retrievers.base_input_retriever import BaseInputRetriever
from genai_perf.inputs.retrievers.generic_dataset import (
    FileData,
    GenericDataset,
    ImageData,
    TextData,
)


class BaseFileInputRetriever(BaseInputRetriever):
    """
    A base input retriever class that defines file input methods.
    """

    def _verify_file(self, filename: Path) -> None:
        """
        Verifies that the file exists.

        Args
        ----------
        filename : Path
            The file path to verify.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not filename.exists():
            raise FileNotFoundError(f"The file '{filename}' does not exist.")

    def _get_content_from_input_file(
        self, filename: Path
    ) -> Union[Tuple[TextData, ImageData], Dict[str, Any]]:
        """
        Reads the content from a JSONL file and returns lists of each content type.

        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_input_dataset_from_file(self, filename: Path) -> FileData:
        """
        Retrieves the dataset from a specific JSONL file.

        """

        raise NotImplementedError("This method should be implemented by subclasses.")

    def retrieve_data(self) -> GenericDataset:
        """
        Retrieves the dataset from a file or directory.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
