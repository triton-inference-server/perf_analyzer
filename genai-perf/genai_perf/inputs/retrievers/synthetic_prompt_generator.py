# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from genai_perf.inputs.input_constants import DEFAULT_CORPUS_FILE
from genai_perf.logging import logging
from genai_perf.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class SyntheticPromptGenerator:
    _tokenized_corpus = None
    _corpus_length = 0
    _prefix_prompts: List[str] = []
    logger = logging.getLogger(__name__)

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
    def _initialize_corpus(
        cls, tokenizer: Tokenizer, corpus_file: str = DEFAULT_CORPUS_FILE
    ) -> None:
        """
        Load and tokenize the corpus once, storing it for reuse.

        Args:
            tokenizer: Tokenizer for tokenizing the corpus.
        """
        corpus_path = pathlib.Path(__file__).parent / corpus_file

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
        if num_tokens > cls._corpus_length:
            logger.warning(
                f"Requested prompt length {num_tokens} is longer than the corpus. "
                f"Returning a prompt of length {cls._corpus_length}."
            )

        start_idx = random.randrange(cls._corpus_length)

        end_idx = start_idx + num_tokens
        prompt_tokens = cls._tokenized_corpus[start_idx:end_idx]
        if end_idx > cls._corpus_length:
            prompt_tokens += cls._tokenized_corpus[: end_idx - cls._corpus_length]

        return tokenizer.decode(prompt_tokens)

    @classmethod
    def create_prefix_prompts_pool(
        cls, tokenizer: Tokenizer, num_prompts: int, prompt_length: int
    ) -> None:
        """
        Generate a pool of prefix prompts.

        Args:
            tokenizer: Tokenizer instance.
            num_prompts: Number of prefix prompts to generate.
            prompt_length: Number of tokens per prefix prompt.
        """
        if cls._tokenized_corpus is None:
            cls._initialize_corpus(tokenizer)

        cls._prefix_prompts = [
            cls._generate_prompt(tokenizer, prompt_length) for _ in range(num_prompts)
        ]

    @classmethod
    def get_random_prefix_prompt(cls) -> str:
        """
        Fetch a random prefix prompt from the pool.

        Returns:
            A random prefix prompt.
        """
        return random.choice(cls._prefix_prompts)
