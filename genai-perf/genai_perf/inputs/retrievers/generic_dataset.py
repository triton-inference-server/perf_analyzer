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

from dataclasses import dataclass, field
from typing import Dict, List, TypeAlias

Filename: TypeAlias = str
TypeOfData: TypeAlias = str
ListOfData: TypeAlias = List[str]
DataRowDict: TypeAlias = Dict[TypeOfData, ListOfData]
GenericDatasetDict: TypeAlias = Dict[Filename, List[DataRowDict]]


@dataclass
class DataRow:
    texts: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)

    def to_dict(self) -> DataRowDict:
        """
        Converts the DataRow object to a dictionary.
        """
        return {"texts": self.texts, "images": self.images}


@dataclass
class FileData:
    filename: str
    rows: List[DataRow]

    def to_dict(self) -> Dict[Filename, List[DataRowDict]]:
        """
        Converts the FileData object to a dictionary.
        Output format example for two payloads from a file:
        {
            'file_0': [
                {'texts': ['text1', 'text2'], 'images': ['image1', 'image2']},
                {'texts': ['text3', 'text4'], 'images': ['image3', 'image4']}
            ]
        }
        """
        return {self.filename: [row.to_dict() for row in self.rows]}


@dataclass
class GenericDataset:
    files_data: Dict[str, FileData]

    def to_dict(self) -> GenericDatasetDict:
        """
        Converts the entire DataStructure object to a dictionary.
        Output format example for one payload from two files:
        {
            {
            'file_0': [{'texts': ['text1', 'text2'], 'images': ['image1', 'image2']}],
            'file_1': [{'texts': ['text1', 'text2'], 'images': ['image1', 'image2']}]
        }
        }
        """
        return {
            filename: file_data.to_dict()[filename]
            for filename, file_data in self.files_data.items()
        }
