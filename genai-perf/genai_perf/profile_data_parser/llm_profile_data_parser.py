#!/usr/bin/env python3

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

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NoReturn, Tuple, TypeAlias

import orjson
from genai_perf.constants import DEFAULT_LRU_CACHE_SIZE, EMPTY_RESPONSE_TOKEN
from genai_perf.exceptions import GenAIPerfException
from genai_perf.logging import logging
from genai_perf.metrics import LLMMetrics, Statistics
from genai_perf.profile_data_parser.profile_data_parser import (
    ProfileDataParser,
    ResponseFormat,
)
from genai_perf.tokenizer import Tokenizer
from genai_perf.utils import (
    load_json_str,
    not_data_sse_field,
    remove_sse_prefix,
    sse_error_occurred,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)

SessionMetrics: TypeAlias = Dict[str, Dict[str, List[float | int]]]


class LLMProfileDataParser(ProfileDataParser):
    """A class that calculates and aggregates all the LLM performance statistics
    across the Perf Analyzer profile results.

    The LLMProfileDataParser class parses profile export JSON file, collects the
    core LLM performance metrics, and calculates summary statistics for each
    different Perf Analyzer runs/experiments.

    Example:

      >>> ... # run Perf Analyzer with concurrency level 10
      >>>
      >>> from transformers import AutoTokenizer
      >>>
      >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
      >>> pd = LLMProfileDataParser(
      >>>     filename="profile_export.json",
      >>>     tokenizer=tokenizer,
      >>> )
      >>> stats = pd.get_statistics(infer_mode="concurrency", level=10)
      >>>
      >>> print(stats)  # output: Statistics(avg_time_to_first_token=...)
      >>> stats.pretty_print()  # Output: time_to_first_token_s: ...
    """

    def __init__(
        self,
        filename: Path,
        tokenizer: Tokenizer,
        goodput_constraints: Dict[str, float] = {},
    ) -> None:
        self._tokenizer = tokenizer
        self._session_metrics: SessionMetrics = defaultdict(lambda: defaultdict(list))
        super().__init__(filename, goodput_constraints)

    def _parse_profile_data(self, data: dict) -> None:
        """Parse through the entire profile data to collect statistics."""
        self._profile_results = {}
        for experiment in data["experiments"]:
            infer_mode = experiment["experiment"]["mode"]
            load_level = experiment["experiment"]["value"]
            requests = experiment["requests"]

            llm_metrics = self._parse_requests(requests)

            # aggregate and calculate statistics
            statistics = Statistics(llm_metrics)
            self._profile_results[(infer_mode, str(load_level))] = statistics

            # calculate per-session statistics
            for session_id, session_metric in self._session_metrics.items():
                metrics = LLMMetrics.from_dict(session_metric)
                self._session_statistics[session_id] = Statistics(metrics)

    def _parse_requests(self, requests: dict) -> LLMMetrics:
        """Parse each requests in profile export data to extract key metrics."""
        min_req_timestamp, max_res_timestamp = float("inf"), 0
        request_latencies: List[int] = []
        time_to_first_tokens: List[int] = []
        time_to_second_tokens: List[int] = []
        inter_token_latencies: List[int] = []
        output_token_throughputs_per_user: List[float] = []
        input_sequence_lengths: List[int] = []
        output_sequence_lengths: List[int] = []
        chunked_inter_token_latency: List[int] = []
        chunked_inter_token_latencies: List[List[int]] = []

        logger.info(f"Parsing total {len(requests)} requests.")
        for request in tqdm(
            requests,
            desc="Progress: ",
            unit="requests",
            total=len(requests),
            miniters=len(requests) // 100,
        ):
            req_timestamp = request["timestamp"]
            req_inputs = request["request_inputs"]
            res_timestamps = request["response_timestamps"]
            res_outputs = request["response_outputs"]

            self._preprocess_response(res_timestamps, res_outputs)

            # Skip requests with empty response. This happens sometimes when the
            # model returns a single response with empty string.
            if not res_timestamps:
                continue

            # Cache length
            num_responses = len(res_timestamps)

            # Cache frequently used appends
            request_latencies_append = request_latencies.append
            ttft_list_append = time_to_first_tokens.append
            ttst_list_append = time_to_second_tokens.append
            itl_append = inter_token_latencies.append
            tps_user_append = output_token_throughputs_per_user.append
            in_len_append = input_sequence_lengths.append
            out_len_append = output_sequence_lengths.append
            chunked_itls_append = chunked_inter_token_latencies.append

            # track entire benchmark duration
            min_req_timestamp = min(min_req_timestamp, req_timestamp)
            max_res_timestamp = max(max_res_timestamp, res_timestamps[-1])

            # request latencies
            req_latency_ns = res_timestamps[-1] - req_timestamp
            request_latencies_append(req_latency_ns)  # nanosec

            # time to first token
            ttft = res_timestamps[0] - req_timestamp
            ttft_list_append(ttft)

            # time to second token (if available)
            if num_responses > 1:
                ttst = res_timestamps[1] - res_timestamps[0]
                ttst_list_append(ttst)

            # number of input tokens
            input_seq_len = self._get_input_token_count(req_inputs)
            in_len_append(input_seq_len)

            # number of output tokens
            output_token_counts, total_output_token = self._get_output_token_counts(
                res_outputs
            )
            out_len_append(total_output_token)

            if num_responses > 1 and total_output_token > 1:
                # inter token latencies
                inter_token_latency = round(
                    (req_latency_ns - ttft) / (total_output_token - 1)
                )
                itl_append(inter_token_latency)

                # output token throughput per user (TPS/user)
                inter_token_latency_s = inter_token_latency / 1e9
                tps_user_append(1 / inter_token_latency_s)

            # The new ITL calculation above loses all token-level ITL information
            # and as a result breaks ITL vs token position visualization. Keep
            # the old version of inter token latency as a WAR to preserve the
            # visualization.
            chunked_inter_token_latency = []
            for (t1, _), (t2, n2) in self._pairwise(
                zip(res_timestamps, output_token_counts)
            ):
                # TMA-1676: handle empty first/last responses
                # if the latter response has zero token (e.g. empty string),
                # then set it default to one for the sake of inter token latency
                # calculation and to avoid divide by zero.
                num_token = 1 if n2 == 0 else n2
                chunked_inter_token_latency.append(round((t2 - t1) / num_token))
            chunked_itls_append(chunked_inter_token_latency)

            # (per-session) calculate llm metrics
            if "session_id" in req_inputs:
                session_metric = self._session_metrics[req_inputs["session_id"]]
                session_metric["request_latencies"].append(req_latency_ns)
                session_metric["time_to_first_tokens"].append(ttft)
                if num_responses > 1:
                    session_metric["time_to_second_tokens"].append(ttst)
                if num_responses > 1 and total_output_token > 1:
                    session_metric["inter_token_latencies"].append(inter_token_latency)
                    session_metric["output_token_throughputs_per_user"].append(
                        1 / inter_token_latency_s
                    )
                session_metric["input_sequence_lengths"].append(input_seq_len)
                session_metric["output_sequence_lengths"].append(total_output_token)
                # collect request and last response timestamps each session
                session_metric["req_timestamps"].append(req_timestamp)
                session_metric["last_res_timestamps"].append(res_timestamps[-1])

        # request & output token throughput
        benchmark_duration = (max_res_timestamp - min_req_timestamp) / 1e9  # to seconds
        request_throughputs, output_token_throughputs = (
            self._calculate_throughput_metrics(
                requests, output_sequence_lengths, benchmark_duration
            )
        )

        llm_metrics = LLMMetrics(
            request_throughputs,
            request_latencies,
            time_to_first_tokens,
            time_to_second_tokens,
            inter_token_latencies,
            output_token_throughputs,
            output_token_throughputs_per_user,
            output_sequence_lengths,
            input_sequence_lengths,
            chunked_inter_token_latencies,
        )

        self._postprocess_session_metrics()

        if self._goodput_constraints:
            goodput_val = self._calculate_goodput(benchmark_duration, llm_metrics)
            llm_metrics.request_goodputs = goodput_val

        return llm_metrics

    def _calculate_throughput_metrics(
        self,
        requests: dict,
        output_sequence_lengths: List[int],
        benchmark_duration: float,
    ) -> Tuple[List[float], List[float]]:
        """Calculate request throughput and output token throughput."""
        request_throughputs = [len(requests) / benchmark_duration]
        output_token_throughputs = [sum(output_sequence_lengths) / benchmark_duration]

        return request_throughputs, output_token_throughputs

    def _postprocess_session_metrics(self) -> None:
        """Postprocess session metrics to calculate request & output token throughput."""
        for metric in self._session_metrics.values():
            init_t = min(metric["req_timestamps"])
            last_t = max(metric["last_res_timestamps"])
            session_latency = (last_t - init_t) / 1e9  # to seconds
            num_requests = len(metric["req_timestamps"])
            num_tokens = sum(metric["output_sequence_lengths"])
            metric["request_throughputs"].append(num_requests / session_latency)
            metric["output_token_throughputs"].append(num_tokens / session_latency)
            # clean up
            del metric["req_timestamps"]
            del metric["last_res_timestamps"]

    def _pairwise(self, iterable):
        """Generate pairs of consecutive elements from the given iterable."""
        iterable = list(iterable)
        return zip(iterable, iterable[1:])

    def _preprocess_response(
        self, res_timestamps: List[int], res_outputs: List[Dict[str, str]]
    ) -> None:
        """Helper function to preprocess responses of a request."""
        if (
            self._service_kind == "openai"
            and "huggingface" not in self._response_format.name.lower()
        ):
            # Sometimes streamed chunks are returned in a splintered fashion.
            # This forces a merge with the previous chunk if error detected.
            for i in reversed(range(1, len(res_outputs))):
                response = res_outputs[i]["response"]

                # skip non-data events
                if not_data_sse_field(response):
                    continue

                if not response.startswith("data: "):
                    prefix_idx = response.find("data: ")
                    if "data:" not in res_outputs[i - 1]["response"]:
                        raise GenAIPerfException(
                            "Detected a splintered SSE response but the "
                            "previous response does not contain proper SSE "
                            "prefix to continue the fix."
                        )

                    # When 'data: ' prefix is not found in the current
                    # response, append it with the previous response
                    # (assuming that the prev response contains it).
                    if prefix_idx == -1:
                        res_outputs[i - 1]["response"] += response
                        res_outputs[i]["response"] = ""
                    else:
                        res_outputs[i - 1]["response"] += response[0:prefix_idx]
                        res_outputs[i]["response"] = response[prefix_idx:]

            # PA sometimes receives multiple SSE responses at once (as a single
            # response). Handle these responses by merging into a single response.
            for i in range(len(res_outputs)):
                responses = res_outputs[i]["response"].strip().split("\n\n")

                # Check if any error event occurred.
                for r in responses:
                    if sse_error_occurred(r):
                        raise GenAIPerfException(
                            f"Detected an error event in the SSE response: {r}"
                        )

                if len(responses) > 1:
                    data = load_json_str(remove_sse_prefix(responses[0]))
                    if self._response_format == ResponseFormat.TRITON_GENERATE:
                        merged_text = "".join(
                            [self._extract_generate_text_output(r) for r in responses]
                        )
                        data["text_output"] = merged_text
                    elif self._response_format == ResponseFormat.HUGGINGFACE_GENERATE:
                        merged_text = "".join(
                            [
                                self._extract_huggingface_generate_text_output(r)
                                for r in responses
                            ]
                        )
                        if isinstance(data, list) and len(data) > 0:
                            data[0]["generated_text"] = merged_text  # type: ignore
                    else:
                        merged_text = "".join(
                            [self._extract_text_output(r) for r in responses]
                        )
                        if self._response_format == ResponseFormat.OPENAI_COMPLETIONS:
                            data["choices"][0]["text"] = merged_text
                        else:
                            data["choices"][0]["delta"]["content"] = merged_text
                    res_outputs[i] = {"response": orjson.dumps(data).decode("utf-8")}
                elif self._is_empty_response(responses[0]):
                    res_outputs[i]["response"] = ""

            # Remove responses without any content
            indices_to_remove = []
            for idx, out in enumerate(res_outputs):
                if not out["response"] or self._is_empty_response(out["response"]):
                    indices_to_remove.append(idx)
            indices_to_remove.sort(reverse=True)
            for index in indices_to_remove:
                res_timestamps.pop(index)
                res_outputs.pop(index)

    def _get_input_token_count(self, req_inputs: dict) -> int:
        """Deserialize the request input and return tokenized inputs."""
        if self._service_kind == "triton":
            input_text = req_inputs["text_input"]
        elif self._service_kind == "triton_c_api":
            return len(req_inputs["input_ids"])  # no tokenizer required
        elif self._service_kind == "openai":
            input_text = self._get_input_payload(req_inputs)
        else:
            raise ValueError(f"Unknown service kind: '{self._service_kind}'.")

        return self._cached_encode(input_text)

    @lru_cache(maxsize=DEFAULT_LRU_CACHE_SIZE)
    def _cached_encode(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def _get_input_payload(self, req_inputs: dict) -> str:
        """Deserialize the request input payload."""
        payload = load_json_str(req_inputs["payload"])
        if self._response_format == ResponseFormat.TRITON_GENERATE:
            return " ".join(payload["text_input"])
        elif self._response_format == ResponseFormat.HUGGINGFACE_GENERATE:
            return payload["inputs"]
        elif self._response_format == ResponseFormat.OPENAI_CHAT_COMPLETIONS:
            return " ".join(m["content"] for m in payload["messages"])
        elif self._response_format == ResponseFormat.OPENAI_COMPLETIONS:
            return " ".join(payload["prompt"])
        elif self._response_format == ResponseFormat.OPENAI_MULTIMODAL:
            content = payload["messages"][0]["content"]
            return " ".join(c["text"] for c in content if c["type"] == "text")
        else:
            raise ValueError("Failed to parse request input in profile export file.")

    def _get_output_token_counts(
        self, res_outputs: List[Dict]
    ) -> Tuple[List[int], int]:
        if self._service_kind == "triton":
            output_texts = [r["text_output"] for r in res_outputs]
        elif self._service_kind == "triton_c_api":
            token_ids = []
            for r in res_outputs:
                if isinstance(r["output_ids"], list):
                    token_ids += r["output_ids"]
                elif isinstance(r["output_ids"], int):
                    token_ids.append(r["output_ids"])
                else:
                    token_ids.append(EMPTY_RESPONSE_TOKEN)
            return token_ids, len(token_ids)
        elif self._service_kind == "openai":
            output_texts = [
                self._extract_text_output(r["response"]) for r in res_outputs
            ]
        else:
            raise ValueError(f"Unknown service kind: '{self._service_kind}'.")

        full_text_token_count = self._cached_encode("".join(output_texts))
        output_token_counts = self._get_output_token_counts_batch(output_texts)
        return output_token_counts, full_text_token_count

    def _get_output_token_counts_batch(self, texts: List[str]) -> List[int]:
        # Exclamation mark forces tokenizers to use consistent prefix
        forced_texts = ["!" + t for t in texts]
        encoded = self._tokenizer(forced_texts)
        input_ids: List[List[int]] = encoded["input_ids"]  # type: ignore
        return [len(ids[1:]) for ids in input_ids]

    def _get_tensorrtllm_engine_token_counts(
        self, res_outputs: List[Dict]
    ) -> Tuple[List[int], int]:
        token_ids = []
        for r in res_outputs:
            if isinstance(r["output_ids"], list):
                token_ids += r["output_ids"]
            elif isinstance(r["output_ids"], int):
                token_ids.append(r["output_ids"])
            else:
                # for the empty first/last responses
                token_ids.append(EMPTY_RESPONSE_TOKEN)
        return token_ids, len(token_ids)

    def _get_triton_output_tokens(self, res_outputs: List[Dict]) -> List[str]:
        """Return a list of Triton response texts."""
        return [r["text_output"] for r in res_outputs]

    def _get_text_output_tokens(self, res_outputs: List[Dict]) -> List[str]:
        """Return a list of LLM response texts."""
        output_texts = []
        for output in res_outputs:
            text = self._extract_text_output(output["response"])
            output_texts.append(text)
        return output_texts

    def _extract_text_output(self, response: str) -> str:
        """Extract text output based on response format."""
        if not response or not_data_sse_field(response):
            return ""

        response = remove_sse_prefix(response)
        if response == "[DONE]":
            return ""

        # Use a mapping to select the appropriate extraction method
        extraction_methods = {
            ResponseFormat.TRITON_GENERATE: self._extract_generate_text_output,
            ResponseFormat.HUGGINGFACE_GENERATE: self._extract_huggingface_generate_text_output,
            ResponseFormat.OPENAI_CHAT_COMPLETIONS: self._extract_openai_chat_text_output,
            ResponseFormat.OPENAI_COMPLETIONS: self._extract_openai_completion_text_output,
            ResponseFormat.OPENAI_MULTIMODAL: self._extract_openai_chat_text_output,
        }
        extract_method = extraction_methods.get(
            self._response_format, self._throw_unknown_response_format_error
        )
        return extract_method(response) or ""  # prevent null output

    def _get_response_output_tokens(self, output_texts: List[str]) -> List[List[int]]:
        """Return a list of response output tokens."""
        # Exclamation mark trick forces the llama tokenization to consistently
        # start each output with a specific token which allows us to safely skip
        # the first token of every tokenized output and get only the ones that
        # are returned by the model
        encodings = self._tokenizer(["!" + txt for txt in output_texts])
        return [out[1:] for out in encodings.data["input_ids"]]

    def _extract_huggingface_generate_text_output(self, response: str) -> str:
        data = load_json_str(response)

        if isinstance(data, list):
            if len(data) > 0:
                first_item = data[0]  # type: ignore
                if isinstance(first_item, dict):
                    return first_item.get("generated_text", "")
                else:
                    raise ValueError(
                        f"Unknown HuggingFace response type in list: {type(first_item)}"
                    )
        else:
            raise ValueError(f"Unknown HuggingFace response type: {type(data)}")

        return ""

    def _extract_openai_chat_text_output(self, response: str) -> str:
        data = load_json_str(response)
        completions = data.get("choices", [{}])[0]

        obj_type = data.get("object")
        if not obj_type:
            # FIXME: TPA-47 workaround for vLLM not following OpenAI Completions
            # API specification when streaming, missing 'object' field:
            # https://platform.openai.com/docs/api-reference/completions
            return completions.get("text", "")
        elif obj_type == "chat.completion":  # non-streaming
            return completions["message"].get("content", "")
        elif obj_type == "chat.completion.chunk":  # streaming
            return completions["delta"].get("content", "")
        else:
            raise ValueError(f"Unknown OpenAI response object type '{obj_type}'.")

    def _extract_openai_completion_text_output(self, response: str) -> str:
        """Extract text from OpenAI completion response."""
        data = load_json_str(response)
        completions = data["choices"][0]  # type: ignore

        obj_type = data.get("object")
        if not obj_type:
            # FIXME: TPA-47 workaround for vLLM not following OpenAI Completions
            # API specification when streaming, missing 'object' field:
            # https://platform.openai.com/docs/api-reference/completions
            return completions.get("text", "")
        elif obj_type == "text_completion":  # legacy
            return completions.get("text", "")
        else:
            obj_type = data["object"]
            raise ValueError(f"Unknown OpenAI response object type '{obj_type}'.")

    def _throw_unknown_response_format_error(self, response: str) -> NoReturn:
        raise ValueError(f"Unknown response format: {self._response_format}")

    def _extract_generate_text_output(self, response: str) -> str:
        # (TODO) TPA-829: Add more proper SSE event stream support
        # Check for empty, comment, or event SSE response
        if not response or not_data_sse_field(response):
            return ""

        response = remove_sse_prefix(response)
        if response == "":
            return response
        data = load_json_str(response)
        return data["text_output"]

    def _is_empty_response(self, response: str) -> bool:
        """Returns true if the response is an LLM response with no content (or empty content)"""
        text = self._extract_text_output(response)
        return not bool(text)
