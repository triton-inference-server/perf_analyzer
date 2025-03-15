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

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.tokenizer import (
    DEFAULT_TOKENIZER,
    DEFAULT_TOKENIZER_REVISION,
    get_tokenizer,
)


class TestTokenizer:
    def _create_tokenizer_config(
        self, name, trust_remote_code=False, revision=DEFAULT_TOKENIZER_REVISION
    ):
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = name
        config.tokenizer.trust_remote_code = trust_remote_code
        config.tokenizer.revision = revision

        return config

    def test_default_tokenizer(self):
        config = self._create_tokenizer_config(name=DEFAULT_TOKENIZER)
        get_tokenizer(config)

    def test_non_default_tokenizer(self):
        config = self._create_tokenizer_config(name="gpt2")
        get_tokenizer(config)

    def test_default_tokenizer_all_args(self):
        config = self._create_tokenizer_config(
            name=DEFAULT_TOKENIZER,
            trust_remote_code=False,
            revision=DEFAULT_TOKENIZER_REVISION,
        )
        get_tokenizer(config)

    def test_non_default_tokenizer_all_args(self):
        config = self._create_tokenizer_config(
            name="gpt2",
            trust_remote_code=False,
            revision="11c5a3d5811f50298f278a704980280950aedb10",
        )
        get_tokenizer(config)

    def test_default_args(self):
        config = self._create_tokenizer_config(name=DEFAULT_TOKENIZER)
        tokenizer = get_tokenizer(config)

        # There are 3 special tokens in the default tokenizer
        #  - <unk>: 0  (unknown)
        #  - <s>: 1  (beginning of sentence)
        #  - </s>: 2  (end of sentence)
        special_tokens = list(tokenizer._tokenizer.added_tokens_encoder.keys())
        special_token_ids = list(tokenizer._tokenizer.added_tokens_encoder.values())

        # special tokens are disabled by default
        text = "This is test."
        tokens = tokenizer(text)["input_ids"]
        assert all([s not in tokens for s in special_token_ids])

        tokens = tokenizer.encode(text)
        assert all([s not in tokens for s in special_token_ids])

        output = tokenizer.decode(tokens)
        assert all([s not in output for s in special_tokens])

        # check special tokens is enabled
        text = "This is test."
        tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        assert any([s in tokens for s in special_token_ids])

        tokens = tokenizer.encode(text, add_special_tokens=True)
        assert any([s in tokens for s in special_token_ids])

        output = tokenizer.decode(tokens, skip_special_tokens=False)
        assert any([s in output for s in special_tokens])
