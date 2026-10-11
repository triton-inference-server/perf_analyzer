# Support custom image-retrieval endpoints in GenAI-Perf

## Summary

GenAI-Perf fails while processing results from valid image-retrieval endpoints
when the endpoint is not the hard-coded `v1/infer` route and the response body
does not contain the literal string `image_retrieval`.

This affects NVIDIA Retriever NIMs that use routes such as:

- `v1/ocr`
- `v1/page-elements`
- `v1/table-structure`

The request workload completes and Perf Analyzer writes a valid profile export,
but GenAI-Perf then exits with:

```text
RuntimeError: Unknown OpenAI response format.
```

The proposed change makes `ImageRetrievalProfileDataParser` trust the output
format already selected by `--endpoint-type image_retrieval`. It does not add an
allowlist of Retriever endpoint names and does not depend on a product-specific
response schema.

## User-visible failure

A representative invocation is:

```bash
genai-perf profile \
  --model nvidia/nemotron-ocr-v2 \
  --service-kind openai \
  --endpoint-type image_retrieval \
  --endpoint v1/ocr \
  --batch-size-image 1 \
  --input-file /datasets/images.jsonl \
  --request-count 300 \
  --concurrency 1 \
  --url http://localhost:8000
```

Perf Analyzer successfully sends requests to the NIM and creates
`profile_export.json`. The failure occurs afterward, when GenAI-Perf tries to
calculate and export metrics.

A simplified OCR response in the profile export looks like:

```json
{
  "model": "nvidia/nemotron-ocr-v2",
  "data": [
    {
      "index": 0,
      "text_detections": []
    }
  ],
  "usage": {
    "images_size_mb": 1.0
  }
}
```

Page-elements and table-structure NIMs use the same image-retrieval request
shape but return their own task-specific objects in `data`.

## Root cause

GenAI-Perf already has authoritative format information at configuration time:

```text
--endpoint-type image_retrieval
        |
        v
OutputFormat.IMAGE_RETRIEVAL
        |
        v
ImageRetrievalProfileDataParser
```

Despite selecting the specialized parser, its base constructor calls
`ProfileDataParser._get_profile_metadata()`. For an OpenAI service, that method
tries to infer the response format again.

The inference recognizes image retrieval only when either:

1. the endpoint is exactly `v1/infer`, or
2. the serialized response contains the substring `image_retrieval`.

The OCR, page-elements, and table-structure routes satisfy neither condition.
Consequently, format inference raises `Unknown OpenAI response format` even
though the user explicitly supplied the correct endpoint type and GenAI-Perf
already selected the correct parser.

This is a result-processing problem, not an HTTP compatibility problem. The
requests reach the custom endpoint and the NIM returns successful responses
before the error occurs.

## Proposed change

Override metadata handling in `ImageRetrievalProfileDataParser` so an OpenAI
profile handled by that class is assigned `ResponseFormat.IMAGE_RETRIEVAL`:

```python
def _get_profile_metadata(self, data: dict) -> None:
    self._service_kind = data["service_kind"]
    if self._service_kind == "openai":
        self._response_format = ResponseFormat.IMAGE_RETRIEVAL
    else:
        super()._get_profile_metadata(data)
```

This is safe within the existing dispatch model because GenAI-Perf instantiates
`ImageRetrievalProfileDataParser` only after the configuration has selected the
image-retrieval output format.

The change intentionally does not inspect the endpoint name or response body.
An invalid route will still fail during the request. A valid custom route can
return any task-specific response because image metrics are calculated from:

- request and response timestamps, and
- the number of `image_url` entries in the request payload.

The image-retrieval metric parser does not need fields from the response body.

## Scope

The proposed fix covers all OpenAI-style endpoints used with
`--endpoint-type image_retrieval`, including:

| Endpoint | Workload |
| --- | --- |
| `v1/infer` | Existing generic image retrieval |
| `v1/ocr` | OCR |
| `v1/page-elements` | Page element detection |
| `v1/table-structure` | Table structure detection |
| Future custom routes | Any route using the image-retrieval request format |

It is not overfit to OCR and does not require adding new endpoint names when a
Retriever NIM introduces another image task.

The change is deliberately limited to image retrieval. Other endpoint types
continue to use their existing metadata behavior.

## Why CLI changes are insufficient

There is no supported GenAI-Perf 0.0.11 option that bypasses the second format
inference step.

- `--endpoint-type image_retrieval` is already present and correctly selects the
  specialized parser.
- Changing `--endpoint` to `v1/infer` would send the request to the wrong NIM
  route.
- Adding the string `image_retrieval` to benchmark data or relying on it to
  appear in a model response would be brittle and could alter the workload.
- GenAI-Perf 0.0.11 cannot reprocess the profile export through a dedicated
  export-processing subcommand after manually changing its endpoint metadata.

The defect therefore needs a code change in GenAI-Perf.

## Validation

The issue and proposed fix were tested with GenAI-Perf 0.0.11 and the OCR 2.0.0
release-candidate NIM.

1. The unmodified public command completed the load but failed during profile
   parsing with `Unknown OpenAI response format`.
2. A private compatibility wrapper that forces the image-retrieval response
   format completed successfully, confirming the failure location.
3. The proposed public parser change was mounted over the stock GenAI-Perf
   0.0.11 installation and run against the same NIM.
4. The patched run completed without the private wrapper and generated both
   `profile_export_genai_perf.json` and `profile_export_genai_perf.csv`.
5. A regression test using endpoint `v1/ocr` and an OCR-shaped response passes,
   along with the existing `v1/infer` tests.

The candidate implementation and regression test are currently in:

- `genai-perf/genai_perf/profile_data_parser/image_retrieval_profile_data_parser.py`
- `genai-perf/tests/test_data_parser/test_image_retrieval_profile_data_parser.py`

## More general robustness direction

The underlying design issue is that response format is known when GenAI-Perf
builds its configuration, but later code tries to recover it from endpoint names
and response substrings. That creates a growing list of special cases for custom
endpoints.

A broader follow-up could make output format explicit throughout result
processing:

1. Derive the expected response format from the configured endpoint type.
2. Pass it directly to `ProfileDataParser`, or persist `endpoint_type` or
   `response_format` in the profile export metadata.
3. Use endpoint and response sniffing only as a compatibility fallback for old
   profile exports that lack explicit format metadata.

That design would make custom embeddings, rankings, multimodal, and generative
endpoints more robust as well. The proposed image-parser override is a small,
backward-compatible correction that follows this principle without requiring a
profile-export schema change.
