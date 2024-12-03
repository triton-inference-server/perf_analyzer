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

import os
import pathlib
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List

from genai_perf.tokenizer import Tokenizer


class SyntheticPromptGenerator:
    _tokenized_corpus = None
    _corpus_length = 0

    @classmethod
    def create_synthetic_prompt(
        cls,
        tokenizer: Tokenizer,
        prompt_tokens_mean: int = 550,
        prompt_tokens_stddev: int = 250,
    ) -> str:
        """
        Generate a synthetic prompt with a specific number of tokens.

        Args:
            tokenizer: Tokenizer instance.
            prompt_tokens_mean: Mean number of tokens in the prompt.
            prompt_tokens_stddev: Standard deviation for the number of tokens in the prompt.

        Returns:
            A synthetic prompt as a string.
        """
        if cls._tokenized_corpus is None:
            cls._initialize_corpus(tokenizer)

        num_prompt_tokens = max(
            1, int(random.gauss(prompt_tokens_mean, prompt_tokens_stddev))
        )

        return cls._generate_prompt(tokenizer, num_prompt_tokens)

    @classmethod
    def _initialize_corpus(cls, tokenizer: Tokenizer):
        """
        Load and tokenize the corpus once, storing it for reuse.

        Args:
            tokenizer: Tokenizer for tokenizing the corpus.
        """
        corpus_path = pathlib.Path(__file__).parent / "sonnets.txt"

        with open(corpus_path, "r") as f:
            lines = f.readlines()

        def tokenize_chunk(chunk):
            return tokenizer.encode(" ".join(chunk))

        num_threads = os.cpu_count()
        if num_threads is None:
            num_threads = 4
        chunk_size = len(lines) // num_threads
        chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            tokenized_chunks = list(executor.map(tokenize_chunk, chunks))

        cls._tokenized_corpus = [token for chunk in tokenized_chunks for token in chunk]
        cls._corpus_length = len(cls._tokenized_corpus)

    @classmethod
    def _generate_prompt(cls, tokenizer: Tokenizer, num_tokens: int) -> str:
        """
        Generate a prompt containing exactly `num_tokens` using the preloaded tokenized corpus.

        Args:
            tokenizer: Tokenizer for decoding tokens.
            num_tokens: Number of tokens required in the prompt.

        Returns:
            A synthetic prompt as a string.

        Raises:
            ValueError: If the tokenized corpus is not initialized
        """
        if not cls._tokenized_corpus:
            raise ValueError("Tokenized corpus is not initialized.")

        start_idx = random.randrange(cls._corpus_length)

        end_idx = start_idx + num_tokens
        prompt_tokens = cls._tokenized_corpus[start_idx:end_idx]
        if end_idx > cls._corpus_length:
            prompt_tokens += cls._tokenized_corpus[: end_idx - cls._corpus_length]

        return tokenizer.decode(prompt_tokens)
