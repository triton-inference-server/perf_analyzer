# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from contextlib import nullcontext as does_not_raise

import pytest
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.exceptions import GenAIPerfException
from genai_perf.inputs.retrievers.synthetic_prompt_generator import (
    SyntheticPromptGenerator,
)
from genai_perf.tokenizer import get_tokenizer


class TestSyntheticPromptGenerator:

    def test_synthetic_prompt_default(self):
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "gpt2"
        tokenizer = get_tokenizer(config)
        _ = SyntheticPromptGenerator.create_synthetic_prompt(tokenizer)

    def test_synthetic_prompt_zero_token(self):
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "gpt2"
        tokenizer = get_tokenizer(config)
        prompt = SyntheticPromptGenerator.create_synthetic_prompt(
            tokenizer=tokenizer,
            prompt_tokens_mean=0,
            prompt_tokens_stddev=0,
        )

        assert prompt == ""
        assert len(tokenizer.encode(prompt)) == 0

    def test_synthetic_prompt_nonzero_tokens(self):
        prompt_tokens = 123
        tolerance = 2
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "gpt2"
        tokenizer = get_tokenizer(config)
        prompt = SyntheticPromptGenerator.create_synthetic_prompt(
            tokenizer=tokenizer,
            prompt_tokens_mean=prompt_tokens,
            prompt_tokens_stddev=0,
        )
        assert len(tokenizer.encode(prompt)) <= 123 + tolerance
        assert len(tokenizer.encode(prompt)) >= 123 - tolerance

    @pytest.mark.parametrize(
        "test_num_tokens, context",
        [
            (12, does_not_raise()),
            (9, pytest.raises(GenAIPerfException)),
            (16, pytest.raises(GenAIPerfException)),
        ],
    )
    def test_generate_prompt_with_token_reuse(self, test_num_tokens, context):
        config = ConfigCommand({"model_name": "test_model"})
        config.tokenizer.name = "gpt2"
        tokenizer = get_tokenizer(config)
        with context:
            _ = SyntheticPromptGenerator._generate_prompt_with_token_reuse(
                tokenizer=tokenizer,
                num_tokens=test_num_tokens,
                prompt_hash_list=[1, 2, 3],
                block_size=5,
            )
