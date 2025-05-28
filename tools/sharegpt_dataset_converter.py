#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import json
import sys

from transformers import AutoTokenizer


def configure_tokenizer(tokenizer_name):
    """Initialize and configure the Hugging Face tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Tokenizer error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def process_conversation(conversation, tokenizer):
    """Extract human prompt and calculate GPT response tokens"""
    if len(conversation) < 2:
        return None

    human_msg = (
        conversation[0].get("value", "")
        if conversation[0].get("from") == "human"
        else ""
    )
    gpt_msg = (
        conversation[1].get("value", "") if conversation[1].get("from") == "gpt" else ""
    )

    if not human_msg or not gpt_msg:
        return None

    tokens = tokenizer.encode(gpt_msg, add_special_tokens=True)
    return {"text": human_msg, "output_length": len(tokens)}


def convert_dataset(args):
    """Main conversion logic with tokenizer integration"""
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            sharegpt_data = json.load(f)

        tokenizer = configure_tokenizer(args.tokenizer)
        converted_count = 0
        skipped_count = 0

        with open(args.output, "w", encoding="utf-8") as outfile:
            for item in sharegpt_data:
                conversations = item.get("conversations", [])[:2]
                processed = process_conversation(conversations, tokenizer)

                if processed:
                    outfile.write(json.dumps(processed) + "\n")
                    converted_count += 1
                else:
                    skipped_count += 1

        print(f"Successfully converted {converted_count} entries")
        print(f"Skipped {skipped_count} invalid/malformed entries")
        return True

    except Exception as e:
        print(f"Conversion error: {str(e)}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert ShareGPT dataset to GenAI-Perf format with accurate token counting"
    )
    parser.add_argument("--input", required=True, help="Input ShareGPT JSON file")
    parser.add_argument("--output", required=True, help="Output JSONL filename")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help='Hugging Face tokenizer name (e.g. "meta-llama/Meta-Llama-3-8B-Instruct")',
    )

    args = parser.parse_args()

    if not convert_dataset(args):
        sys.exit(1)


if __name__ == "__main__":
    main()
