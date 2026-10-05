<!--
# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
-->

# Report a Security Vulnerability

To report a potential security vulnerability in any NVIDIA product, please use either:
* [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html), or
* Send email to psirt@nvidia.com

**OEM Partners should contact their NVIDIA Customer Program Manager**

If reporting a potential vulnerability via email, please encrypt it using NVIDIA’s public PGP key ([see PGP Key page](https://www.nvidia.com/en-us/security/pgp-key/)) and include the following information:
1. Product/Driver name and version/branch that contains the vulnerability
2. Type of vulnerability (code execution, denial of service, buffer overflow, etc.)
3. Instructions to reproduce the vulnerability
4. Proof-of-concept or exploit code
5. Potential impact of the vulnerability, including how an attacker could exploit the vulnerability

See https://www.nvidia.com/en-us/security/ for past NVIDIA Security Bulletins and Notices.

## Additional Reporting Channels

In addition to the channels above, you can use:

* The [NVIDIA Vulnerability Disclosure Program](https://www.nvidia.com/en-us/security/) (preferred).
* GitHub Private Vulnerability Reporting, via the **Security** tab of this repository, if it is enabled.

**Do not open a public GitHub issue or pull request for a suspected security vulnerability.**
NVIDIA PSIRT will acknowledge your report, assess it, and coordinate remediation and disclosure with you.

# Security Architecture & Context

Perf Analyzer is a command-line performance and load-generation tool, written primarily in C++ (`src/`) with a Python GenAI front end (`genai-perf/`). It drives inference requests against a separately deployed inference service (Triton Inference Server, an OpenAI-compatible API, TensorFlow Serving, TorchServe, or an in-process Triton C API) and reports latency and throughput. It is a client-side developer and benchmarking tool, not a network service: it opens no listening sockets and holds no user data of its own.

* **Software classification:** CLI application / SDK component.
* **Primary security responsibility:** safely parse operator-supplied arguments and files, and send requests to the configured target without weakening the transport security the operator selected.
* **Key interfaces and boundaries:**
  * Command-line arguments (`src/command_line_parser.cc`), including endpoint URLs, TLS options, and extra HTTP headers.
  * Input files and directories supplied with `--input-data`, parsed by `src/data_loader.cc`.
  * Client backends under `src/client_backend/` (HTTP/gRPC Triton, OpenAI, TensorFlow Serving, TorchServe, dynamic gRPC, Triton C API).
  * System and CUDA shared memory used for inputs and outputs (`src/infer_data_manager_shm.cc`, `src/cuda_runtime_library_manager.cc`).
  * Output files such as the profile export (`src/profile_data_exporter.cc`).
* **Repository Exposure Classification:** Public (the repository is publicly visible on GitHub).
* **Service Exposure Classification:** Internal-Isolated (medium confidence). The tool is run by developers and CI against test or staging services and does not itself serve external traffic.

# Threat Model

1. **Untrusted input data files:** `--input-data` JSON and binary files are parsed in `src/data_loader.cc`. Malformed or oversized files could cause excessive memory use or crashes. JSON tensor values are not resolved as file paths; in directory mode the loader reads files named after the model's input tensors from the data directory. A `message_generator` string in the JSON is run as a command (see threat 2), so an untrusted `--input-data` file must be treated as untrusted code, not only as untrusted data.
2. **Command execution through streamed input:** `DataLoader::ReadDataFromPipe` in `src/data_loader.cc` starts a shell process with `popen` to read streaming input data. The command string is taken directly from the `message_generator` field of the `--input-data` JSON (`DataLoader::ReadTensorData`), so loading an untrusted input-data file can execute arbitrary commands with the privileges of the Perf Analyzer user. Never use `--input-data` files from untrusted sources.
3. **Weakened or misconfigured TLS:** HTTPS peer and host verification and gRPC SSL options are set from command-line flags (`--ssl-https-*`, `--ssl-grpc-*`). Disabling verification, or pointing at the wrong certificate files, exposes request payloads and any authorization headers passed with the tool to interception on the network path.
4. **Sensitive headers and payloads in logs and exports:** The profile export file written by `src/profile_data_exporter.cc` records request inputs, response outputs and timestamps, but not HTTP headers. Exported files may therefore contain sensitive payload content and are written with the process's default file permissions. Verbose console output from the client libraries may also include request details such as headers, so treat console logs as sensitive.
5. **Dynamic library loading in the Triton C API mode:** `src/client_backend/triton_c_api/shared_library.cc` loads a server library with `dlopen` from a user-supplied `--triton-server-directory`, and `src/mpi_utils.cc` and `src/cuda_runtime_library_manager.cc` load `libmpi.so` and `libcudart.so` by name. A writable or attacker-influenced library path or search path can result in loading of unintended code.
6. **Shared memory exposure:** System and CUDA shared-memory regions and CUDA IPC handles (`src/infer_data_manager_shm.cc`) are shared with the target server. Other local processes with access to the same shared-memory namespace could read or tamper with test tensors, and leftover regions after an abnormal exit can persist.
7. **Build and dependency supply chain:** Builds and the Python `genai-perf` package pull third-party dependencies (see `CMakeLists.txt`, `pyproject.toml`). Unpinned or unverified dependencies could introduce vulnerable or malicious code into released artifacts.

# Critical Security Assumptions

* **Trusted operator and environment:** Perf Analyzer is run by a trusted user on a trusted workstation or CI host; command-line arguments, data files, and the server directory are assumed to be controlled by that user.
* **Trusted target service:** The configured endpoint is assumed to be the intended service. The tool does not authenticate the target beyond the TLS settings the operator provides.
* **Operator-controlled TLS:** Transport security is only as strong as the flags supplied. Perf Analyzer's defaults verify peers for HTTPS, but the operator is responsible for enabling TLS where required and not disabling verification outside test environments.
* **Trusted input data:** Input and data-directory files are assumed to be well formed and from a trusted source; the parser is not hardened as a boundary against hostile files.
* **Trusted local libraries:** Shared libraries loaded through `dlopen` and files read from the Triton server directory are assumed to be genuine and not writable by untrusted users.
* **Isolated local host:** Other local users are assumed not to access the process's shared memory, temporary files, or exported profile data.
* **Not for production traffic:** Perf Analyzer is intended for benchmarking and should not be exposed as a service or run with elevated privileges.
