# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import io
from typing import TYPE_CHECKING, List

# Use TYPE_CHECKING to import BatchEncoding only during static type checks
if TYPE_CHECKING:
    from transformers import BatchEncoding

from genai_perf.exceptions import GenAIPerfException

DEFAULT_TOKENIZER = "hf-internal-testing/llama-tokenizer"
DEFAULT_TOKENIZER_REVISION = "main"


class Tokenizer:
    """
    A small wrapper class around Huggingface Tokenizer
    """

    def __init__(self) -> None:
        """
        Initialize the tokenizer with default values
        """

        # default tokenizer parameters for __call__, encode, decode methods
        self._call_args = {"add_special_tokens": False}
        self._encode_args = {"add_special_tokens": False}
        self._decode_args = {"skip_special_tokens": True}

    def set_tokenizer(self, name: str, trust_remote_code: bool, revision: str):
        """
        Downloading the tokenizer from Huggingface.co or local filesystem
        """
        try:
            # Silence tokenizer warning on import and first use
            with contextlib.redirect_stdout(
                io.StringIO()
            ) as stdout, contextlib.redirect_stderr(io.StringIO()):
                from transformers import AutoTokenizer
                from transformers import logging as token_logger

                token_logger.set_verbosity_error()
                tokenizer = AutoTokenizer.from_pretrained(
                    name, trust_remote_code=trust_remote_code, revision=revision
                )
        except Exception as e:
            raise GenAIPerfException(e)
        self._tokenizer = tokenizer

    def __call__(self, text, **kwargs) -> "BatchEncoding":
        self._call_args.update(kwargs)
        return self._tokenizer(text, **self._call_args)

    def encode(self, text, **kwargs) -> List[int]:
        self._encode_args.update(kwargs)
        return self._tokenizer.encode(text, **self._encode_args)

    def decode(self, token_ids, **kwargs) -> str:
        self._decode_args.update(kwargs)
        return self._tokenizer.decode(token_ids, **self._decode_args)

    def __repr__(self) -> str:
        return self._tokenizer.__repr__()


def get_empty_tokenizer() -> Tokenizer:
    """
    Return a Tokenizer without a tokenizer set
    """
    return Tokenizer()


def get_tokenizer(
    tokenizer_model: str,
    trust_remote_code: bool = False,
    tokenizer_revision: str = DEFAULT_TOKENIZER_REVISION,
) -> Tokenizer:
    """
    Return tokenizer for the given model name
    """
    tokenizer = Tokenizer()
    tokenizer.set_tokenizer(tokenizer_model, trust_remote_code, tokenizer_revision)
    return tokenizer
