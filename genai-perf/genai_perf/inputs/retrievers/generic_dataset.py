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

from dataclasses import dataclass, field
from typing import Any, Dict, List, TypeAlias, Union

Filename: TypeAlias = str
TextData: TypeAlias = List[str]
ImageData: TypeAlias = List[str]
AudioData: TypeAlias = List[str]
InputData: TypeAlias = Union[TextData, ImageData, AudioData]
OptionalData: TypeAlias = Dict[str, Any]
PayloadMetadata: TypeAlias = Dict[str, Any]
DataRowField: TypeAlias = Union[InputData, OptionalData, PayloadMetadata]
DataRowDict: TypeAlias = Dict[str, DataRowField]
GenericDatasetDict: TypeAlias = Dict[Filename, List[DataRowDict]]


@dataclass
class DataRow:
    texts: TextData = field(default_factory=list)
    images: ImageData = field(default_factory=list)
    audios: AudioData = field(default_factory=list)
    optional_data: Dict[str, Any] = field(default_factory=dict)
    payload_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> DataRowDict:
        """
        Converts the DataRow object to a dictionary.
        """
        datarow_dict: DataRowDict = {}

        if self.texts:
            datarow_dict["texts"] = self.texts
        if self.images:
            datarow_dict["images"] = self.images
        if self.audios:
            datarow_dict["audios"] = self.audios
        if self.optional_data:
            datarow_dict["optional_data"] = self.optional_data
        if self.payload_metadata:
            datarow_dict["payload_metadata"] = self.payload_metadata
        return datarow_dict


@dataclass
class FileData:
    rows: List[DataRow]

    def to_list(self) -> List[DataRowDict]:
        """
        Converts the FileData object to a list.
        Output format example for two payloads from a file:
        [
            {
                'texts': ['text1', 'text2'],
                'images': ['image1', 'image2'],
                'audios': ['audio1', 'audio2'],
                'optional_data': {},
                'payload_metadata': {},
            },
            {
                'texts': ['text3', 'text4'],
                'images': ['image3', 'image4'],
                'audios': ['audio3', 'audio4'],
                'optional_data': {},
                'payload_metadata': {},
            },
        ]
        """
        return [row.to_dict() for row in self.rows]


@dataclass
class GenericDataset:
    files_data: Dict[str, FileData]

    def to_dict(self) -> GenericDatasetDict:
        """
        Converts the entire DataStructure object to a dictionary.
        Output format example for one payload from two files:
        {
            'file_0': [
                {
                    'texts': ['text1', 'text2'],
                    'images': ['image1', 'image2'],
                    'audios': ['audio1', 'audio2'],
                    'optional_data': {},
                    'payload_metadata': {},
                },
            ],
            'file_1': [
                {
                    'texts': ['text1', 'text2'],
                    'images': ['image1', 'image2'],
                    'audios': ['audio1', 'audio2'],
                    'optional_data': {},
                    'payload_metadata': {},
                },
            ],
        }
        """
        return {
            filename: file_data.to_list()
            for filename, file_data in self.files_data.items()
        }
