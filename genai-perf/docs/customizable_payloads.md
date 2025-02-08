<!--
Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Use Custom Payload Formats

With GenAI-Perf, you can customize how input data is formatted into payloads
using templates. This allows you to define how payloads are
structured when sent to an inference server.

These provide less customizability than a
[customizable frontend](customizable_frontends.md), which is used when more
endpoint-specific logic is necessary. Templates are used when
the only change necessary is specifying a custom payload schema.

## Table of Contents

- [Use a Named Template (Predefined)](#predefined)
- [Use a Custom Template (Fully Customizable)](#custom)

## Use a Named Template <a id="predefined"></a>

GenAI-Perf provides common built-in templates to simplify request formatting.
You can find these in `NAMED_TEMPLATES` in the class
[template_converter](../genai_perf/inputs/converters/template_converter.py).

One such template is `nv-embedqa`, which structures requests for
embedding-based models.

### Example: Use the nv-embedqa Named Template

Run the following command:

```bash
genai-perf profile \
  --model MY_MODEL \
  --tokenizer MY_MODEL \
  --num-payloads 2 \
  --extra-inputs payload_template:nv-embedqa
```

#### Result
GenAI-Perf will use the nv-embedqa template to format the input data.

After conversion, the `inputs.json` file would look similar to this:

```json
{
    "data": [
        {"text": ["example1"]},
        {"text": ["example2"]}
    ]
}
```

## Use a Custom Template <a id="custom"></a>

Sometimes, you may have a custom payload format.
Custom templates allow you to benchmark using GenAI-Perf.

Here is an example template:
```
    {% for item in texts %}
        {"custom_key": [{{ item|tojson }}]}{% if not loop.last %},{% endif %}
    {% endfor %}
```

This tells GenAI-Perf to format input texts like:

```json
[
    {"custom_key": ["example1"]},
    {"custom_key": ["example2"]}
]
```

If you the above template is saved in `custom_template.jinja`,
you can run the command:

```bash
genai-perf profile \
  --model MY_MODEL \
  --tokenizer MY_MODEL \
  --num-payloads 2 \
  --extra-inputs payload_template:custom_template.jinja
```
