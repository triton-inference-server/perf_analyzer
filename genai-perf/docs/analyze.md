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

# GenAI-Perf Analyze Subcommand

The `analyze` subcommand is used to sweep through PA or GenAI-Perf stimulus allowing the user to profile multiple scenarios with a single command. A sub-directory is created for each scenario profiled in the artifacts directory which contains all the same files (input stimulus, output JSON and CSV) as the equivalent profile command. In addition, a top-level CSV report is created which summerizes the results for all the scenarios run.

## Analyze CLI
The `analyze` subcommand uses the same CLI options as `profile` with the following additional options, which are used to specify the type and ranges of the stimulus you wish to sweep:

#### `--sweep-type` - The type of stimulus you wish the sweep over
The currently support stimulus values are `concurrency`, `request_rate`, `input_sequence_length`, and `num_dataset_entries`

#### `--sweep-range` - The range over which the stimulus will be swept
This can be represented as `min:max` or `min:max:step`. If a `step` is not specified then we assume the range to be min/max for power-of-2 values. For example, `8:256`, would sweep `8,16,32,64,128,256`

#### `--sweep-list` - A comma-separated list of values that stimulus will be swept over

### CLI Examples
```bash
genai-perf analyze -m <model> --sweep-type concurrency --sweep-range 1:256
```
This will sweep over concurrencies of 1,2,4,8,16,32,64,128, and 256

```bash
genai-perf analyze -m <model> --sweep-type request_rate --sweep-range 100:500:50
```
This will sweep over request rate at values of 100,150,...450,500
```bash
genai-perf analyze -m <model> --sweep-type input_sequence_length --sweep-list 100,150,200,400
```
This will sweep over ISL for values of 100,150,200 and 400

## Artifact Directories and Summary CSV Report

As when running `profile`, an artifact directory will be created for each scenario profiled. The name of the artifact directory is:
`<model_name>-<service-kind>-<backend>-<sweep_type><sweep_value>`

### Example
For this command:
```bash
genai-perf analyze -m gpt2 --service-kind triton --backend vllm --sweep-type num_dataset_entries --sweep-range 100:200:50
```

The following artifact directories would be created:
```bash
artifacts/gpt2-triton-vllm-num_dataset_entries100/
artifacts/gpt2-triton-vllm-num_dataset_entries150/
artifacts/gpt2-triton-vllm-num_dataset_entries200/
```

Each artifact directory contains the `inputs.json`, `profile_export.json`, and `profile_export_genai_perf.csv` just like it would if you ran `profile` individually for each scenario.

## Summary Report CSV
In the CWD a summary report CSV (`analyze_export_genai_perf.csv`) is created. In the first table, each row is a different scenario profiled while the columns show p99 perf metrics. In the second table, the rows are again scenario's profiled, while the columns are p99 GPU telemetry metrics.

### Example Summary Report CSV
```
Config Name,Concurrency,ISL,Num Dataset Entries,p99 Time To First Token (ms),p99 Inter Token Latency (ms),p99 Request Latency (ms),p99 Output Sequence Length (tokens),Avg. Output Token Throughput (tokens/sec),Request Throughput (requests/sec)
gpt2_run_config_2,1,201,200,33.54,7.16,779.75,132.10,149.63,1.32
gpt2_run_config_1,1,201,150,33.13,7.29,778.62,126.16,147.93,1.32
gpt2_run_config_0,1,201,100,82.02,7.53,879.20,124.55,145.93,1.30

Config Name,GPU,p99 GPU Power Usage (W),p99 GPU Energy Consumption (MJ),p99 GPU Utilization (%),p99 GPU Memory Used (GB),Avg. GPU Power Limit (W),Avg. GPU Total Memory (GB)
gpt2_run_config_2,gpu0,64.46,1.73,20.00,22.63,280.00,25.77
gpt2_run_config_1,gpu0,64.49,1.73,20.00,22.63,280.00,25.77
gpt2_run_config_0,gpu0,63.09,1.72,20.00,22.63,280.00,25.77
```

## Checkpointing
A new feature of `analyze` is the ability to save and restore results from scenarios that have previously run. This is done via reading a `checkpoint.json` (which is located at the CWD). This checkpoint file contains all the information necessary for GenAI-Perf to restore it's state from a previous run of `analyze` and skip over any scenarios that have previously been profiled.

### Example
```bash
genai-perf analyze -m gpt2 --service-kind triton --backend vllm --sweep-type num_dataset_entries --sweep-range 100:200:50
```
This command will profile three scenarios: num_dataset_entries with values 100, 150, & 200.

On a subsequent rerun with this command:
```bash
genai-perf analyze -m gpt2 --service-kind triton --backend vllm --sweep-type num_dataset_entries --sweep-range 100:300:50
```
The first 3 scenarios would not be re-profiled, and you would see the following message:
```
gpt2_run_config_0:num_dataset_entries100 found in checkpoint - skipping profiling...
gpt2_run_config_1:num_dataset_entries150 found in checkpoint - skipping profiling...
gpt2_run_config_2:num_dataset_entries200 found in checkpoint - skipping profiling...
```

*Note: If you want to re-profile all scenarios, first delete the checkpoint file (and artifacts) before running analyze.*

## Reading the Checkpoint and Using the Results API
For those who want to process the results from `analyze` you now have an additional option beyond consuming the CSV files: **GenAI-Perf now provides you the option of reading in the checkpoint file and using the APIs provided below to access the data.**

### Results, RunConfigs, and Records Classes
The Results class holds all the scenarios run during `analyze`, with each scenario assigned an instance of the RunConfig class. The RunConfig class holds information about the GenAI-Perf and PA configurations used for this scenario, as well as the performance and GPU Telemetry metrics. The metrics are held in instances of the Record class (one Record per metric captured).

To summerize:
  - **Results** - a list of RunConfigs (one per scenario)
  - **RunConfig** - contains GenAI-Perf & PA configuration information, plus performance and GPU Telemetry Records
  - **Records** - contains the performance or telemetry metric measured

### Record Class
Records are the class that GenAI-Perf uses to store metrics in the checkpoint. There is a unique Record for every type of metric GenAI-Perf can capture. A full list of these can be found at `genai-perf/genai_perf/record/types/`.

Each record contains a unique tag which is needed to access the value of the record when using the API. For example, if you wanted to find the p99 time to first token latency you would use: `TimeToFirstTokenP99.tag` (found in `time_to_first_token_p99.py`)

## Results API
The Results class stores the list of scenarios (RunConfigs) and is sorted based on the metric objective. To set the objective you use either the `set_gpu_metric_objectives` (for GPU telemetry metrics) or `set_perf_metric_objectives` (for performance metrics).

The format of an objective is as follows:
```python
{<ModelName>: {<RecordTag>: <Weighting>}}
```

If only one objective is set then the weighting does not matter (just set it to 1), but you can specify multiple objectives and bias one objective over another by giving it a higher weighting.

#### Examples
```
results.set_perf_metric_objectives({"gpt2": {TimetoFirstTokenP99.tag: 1}})
```
Sets the objective to be p99 Time-to-First Token latency.

```
results.set_gpu_metric_objectives({"gpt2": {GPUPowerUsageP99.tag: 1}})
```
Sets the objective to be p99 GPU Power Usage

```
results.set_perf_metric_objectives({"gpt2": {TimetoFirstTokenAvg.tag: 1, InterTokenLatencyP90.tag: 3}})
```
Sets the objective with a 3:1 bias for Inter-token latency (p90) vs. Time-to-First Token latency (avg)

### Constraints
In addition to setting objectives you can also set constraints which will filter the Results returned when using the `get_results_<passing/failing>_constriants` method.

Constraints are set using the Model/RunConstraints classes. Here is an example of how you would set the constraints to only return RunConfigs that have a Time-to-First Token latency below 10 ms:
```
model_constraints = ModelConstraints({TimetoFirstTokenP99.tag: 10})
run_constraints = RunConstraints("gpt2": model_constraints)
results.set_constraints(run_constraints)
passing_results = results.get_results_passing_constraints()
```

### RunConfig API
The RunConfig API contains methods that can return the:
  - GenAI-Perf/PA parameters set for this scenario
  - All or specific GPU Telemetry metrics
  - All or specific Performance metrics

Here are some examples of how the RunConfig API can be used:
```python
# Returns a dictionary of `{parameter_name: value}`
run_config.get_genai_perf_parameters()
run_config.get_perf_analyzer_parameters()
```

```python
# Returns a dictionary of `{GpuId: GpuRecords}`
run_config.get_all_gpu_metrics()
```

```python
# Returns a dictionary of `{ModelName: PerfRecords}`
run_config.get_all_perf_metrics()
```

```python
# Returns a list of PerfRecords for the gpt2 model
run_config.get_model_perf_metrics("gpt2")
```

```python
# Returns the PerfRecord for the Time-to-First Token (p99) latency for the gpt2 model
run_config.get_model_perf_metric("gpt2", TimeToFirstTokenP99.tag)
```

```python
# Returns the value of Time-to-First Token (p99) latency for the gpt2 model
run_config.get_model_perf_metric_value("gpt2", TimeToFirstTokenP99.tag)
```

```python
# Returns the value of Avg. GPU Power Usage for the GPU with ID gpu0
run_config.get_gpu_metric_value("gpu0", GPUPowerUsageAvg.tag)
```


### Example Python Code ###
Here is some example python code that demonstrates how the checkpoint is read and how the APIs can be used to access the data:
```python
from genai_perf.checkpoint.checkpoint import Checkpoint
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.run.results import Results
from genai_perf.record.types.input_sequence_length_p99 import InputSequenceLengthP99
from genai_perf.record.types.time_to_first_token_p99 import TimeToFirstTokenP99


# Read in the Checkpoint
model_name = "gpt2"
config = ConfigCommand(model_names=[model_name])
checkpoint = Checkpoint(config)
results = checkpoint.results

# Sort results based on Time-to-First Token latency
results.set_perf_metric_objectives({model_name: {TimeToFirstTokenP99.tag: 1}})

# Create lists of ISL along with their corresponding Time-to-First token latency (sorted by lowest latency)
isl = []
ttftl = []
for run_config in results.run_configs:
  isl.append = run_config.get_model_perf_metric_value(model_name, InputSequenceLengthP99.tag)
  ttftl.append = run_config.get_model_perf_metric_value(model_name, TimeToFirstTokenP99.tag)
```

