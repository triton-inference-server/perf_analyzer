<!--
Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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

# Support a New API With Customizable Frontends

This guide explains how you can add support for benchmarking a new API.
The main requirement is that the endpoint uses request and response
formats that are compatible with the OpenAI frontend.

The primary logic involves adding a new converter to the genai-perf repository
under `genai_perf/inputs/converters/`. Converters allow the system to handle
different request formats. Please follow these steps below to add a converter
for your custom API.

## Create the Converter Class
Create a new file in the `genai_perf/inputs/converters/` directory, such as
`new_converter.py`.

Define the converter class in this file. Your class should inherit from
BaseConverter, which is located in
`genai_perf/inputs/converters/base_converter.py`. You can reference existing
converters for more code examples.

```python
from genai_perf.inputs.converters.base_converter import BaseConverter
from genai_perf.inputs.input_constants import OutputFormat
from genai_perf.inputs.inputs_config import InputsConfig
from genai_perf.exceptions import GenAIPerfException
from typing import Any, Dict

class NewConverter(BaseConverter):
    def check_config(self, config: InputsConfig) -> None:
        # If applicable, any configuration checks go here
        # Else, omit this function

    def convert(
        self, generic_dataset: GenericDataset, config: InputsConfig
    ) -> Dict[Any, Any]:
        request_body: Dict[str, Any] = {"data": []}

        for file_data in generic_dataset.files_data.values():
            for index, row in enumerate(file_data.rows):
                # Select a model name via the specified model selection
                # strategy
                model_name = self._select_model_name(config, index)

                # Populate the request body
                payload = {
                    "model": model_name,
                    "input": row.texts,
                }

                self._add_request_params(payload, config)
                request_body["data"].append({"payload": [payload]})

        return request_body
```

## Update `__init__.py`

In `genai_perf/inputs/converters/__init__.py`, import your new converter class:

```python
from .new_converter import NewConverter
```

Then, add the new converter class to the `__all__` list to make it available
for use:

```python
__all__ = [
    # Existing converters
    "NewConverter",
]
```

## Create the New Output Format

In `genai_perf/inputs/input_constants`, go to the enum OutputFormat.
Add the name of your new endpoint, so that the endpoint name is detected
by the parser.

```python
class OutputFormat(Enum):
    # Existing output formats
    NEW_ENDPOINT = auto()
```
## Map the Output Format to the Converter

Open `genai_perf/inputs/converters/output_format_converter_factory.py.`
Add a mapping for your new converter in the converters dictionary, linking
the appropriate OutputFormat value to your converter class:

```python
converters = {
    # Existing mappings
    OutputFormat.NEW_FORMAT: NewConverter,
}
```

## Update the Metrics Parser

GenAI-Perf needs to know which metrics format your API uses. Go to
`genai_perf/main.py`. If your endpoint is not an LLM endpoint, add it to the
list of endpoint types that use a ProfileDataParser.

If you find that GenAI-Perf does not correctly read your response format, it
may be necessary to create a new profile data parser. If so, go to
`genai_perf/profile_data_parser/image_retrieval_profile_data_parser.py` as an
example for how to create a new data parser. Add a parser in the same
directory, then add an if/else branch to the calculate_metrics function to
use the custom parser for that endpoint type.

## Test the Converter

After implementing your converter, you can run it against your server to
ensure it works:

```bash
genai-perf profile -m TEST_MODEL --endpoint NEW-ENDPOINT
```

You can also write unit tests to ensure it works as expected.
To do so, create a test file in the tests directory.
You can reference existing converter tests named `test_**_converter.py`.
To run the test, run `pytest tests/test_new_converter.py`, replacing the
file name with the name of the file you created.
