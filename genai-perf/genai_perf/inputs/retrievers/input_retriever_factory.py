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

from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.input_constants import PromptSource
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.inputs.retrievers.base_input_retriever import BaseInputRetriever
from genai_perf.inputs.retrievers.file_input_retriever import FileInputRetriever
from genai_perf.inputs.retrievers.synthetic_data_retriever import SyntheticDataRetriever


class InputRetrieverFactory:
    """
    Factory class to create the input retriever to get the input data
    based on the input source.
    """

    @staticmethod
    def create(config: InputsConfig) -> BaseInputRetriever:
        retrievers = {
            PromptSource.SYNTHETIC: SyntheticDataRetriever,
            PromptSource.FILE: FileInputRetriever,
        }
        input_type = config.input_type
        if input_type not in retrievers:
            raise GenAIPerfException(f"Input source '{input_type}' is not recognized.")
        return retrievers[input_type](config)
