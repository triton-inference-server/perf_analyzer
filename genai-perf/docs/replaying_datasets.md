<!--
Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Replaying a dataset

This guide explains how you can replay a dataset using GenAI-Perf.
This leverages a new option and an expanded input file format.

## Requirements
1. New option: `--input-file payload:<path_to_file>`
2. New payload file format

## Payload file format

The payload file format has a couple possibilities to describe the dataset. Utilzing jsonl,
a user can define a dataset via the following:

1. Timestamp and text prompt

```bash
// input.jsonl
{timestamp: 0, "text": "What is in this image?"}
{timestamp: 2, "text": "What breed is the dog in the image?"}
{timestamp: 9, "text": "What kind of bicycle is that?"}
...
```

The requests enumerated in the file will be sent at the defined timestamps with the
corresponding text.

2. Timestamp and hash_ids


```bash
// input.jsonl
{timestamp: 0, "hash_ids": [0, 1, 2, 4]}
{timestamp: 2, "hash_ids": [0, 5, 8, 10]}
{timestamp: 9, "hash_ids": [4, 5, 6, 7]}
...
```
The requests enumerated will be sent at the defined timestamps.
The hash_ids can define blocks of 512 tokens that will be synthetically
generated. Previous hash_ids will be cached, ensuring that repeated hash_ids
send the same data in a payload.


