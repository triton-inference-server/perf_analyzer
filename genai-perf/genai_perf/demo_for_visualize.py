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

import random
import sys

from genai_perf.config.generate.search_parameters import SearchParameters
from genai_perf.config.generate.sweep_objective_generator import SweepObjectiveGenerator
from genai_perf.config.input.config_command import ConfigCommand
from genai_perf.config.run.results import Results
from genai_perf.config.run.run_config import RunConfig
from genai_perf.measurements.model_constraints import ModelConstraints
from genai_perf.measurements.run_constraints import RunConstraints
from genai_perf.record.types.input_sequence_length import InputSequenceLength
from genai_perf.record.types.perf_latency_p99 import PerfLatencyP99
from genai_perf.record.types.perf_throughput import PerfThroughput
from genai_perf.types import ModelSearchParameters
from tests.test_utils import create_run_config


def print_run_config(run_config: RunConfig) -> None:
    throughput = run_config.get_model_perf_metric_value(
        "test_model", PerfThroughput.tag
    )
    latency = run_config.get_model_perf_metric_value("test_model", PerfLatencyP99.tag)
    isl = run_config.get_model_perf_metric_value("test_model", InputSequenceLength.tag)
    pa_parameters = run_config.perf_analyzer_config.get_parameters()
    concurrency = pa_parameters["concurrency"]

    print(
        f"\t{run_config.name} \t concurrency: {concurrency} \t ISL: {isl} \t throughput: {throughput} \t latency: {latency}"
    )


def main():
    random.seed(10)

    # This is a demonstration of how sweep/analyze would run in
    # GenAI-Perf and how the output (Results class) can be used
    # by visualization

    # We don't have a new Config/CLI interface yet, so for now I've created
    # a dataclass that allows you to set values. For this first example,
    # we will use all default values (these will not be our actual defaults
    # in product)
    config = ConfigCommand(model_names=["test_model"])

    # In this next section we will determine what the search space is
    # by default right now this sweeps over concurrency from 1 to 1024 (by powers of 2)
    model_search_parameters = {"test_model": SearchParameters(config.analyze)}

    # Now we instance the Sweep Objective Generator which will create GenAI-Perf & PA
    # configs based on the user config and the model's search parameters
    sweep_objective_generator = SweepObjectiveGenerator(config, model_search_parameters)

    # Next we iterate through the generator - in the real world we would call PA
    # to find the metrics for each config profiled. For this example I will use a
    # test utility created to generate metrics

    # Each profile (or iteration) creates a RunConfig instance and the list of these
    # are stored in the Results class
    results = Results()
    for count, objective in enumerate(sweep_objective_generator.get_objectives()):

        # A RunConfig consists of a unique name, the GenAI-Perf config, the PA config
        # and GPU + Performance metrics. This test utility uses the information provided
        # here to create this
        run_config = create_run_config(
            # These values are used to set the GAP/PA config
            run_config_name="test_model_run_config_" + str(count),
            model_objective_parameters=objective,
            config=config,
            # Telemetry metrics
            gpu_power=random.randint(400, 500),
            gpu_utilization=random.randint(1, 100),
            # Performance metrics
            throughput=random.randint(100, 300),
            latency=random.randint(50, 100),
            input_seq_length=random.randint(20, 80),
            output_seq_length=random.randint(30, 60),
        )

        # Now we add the RunConfig to the Results class
        results.add_run_config(run_config)

    # At this point Analyze would be complete and the Results would be saved to a checkpoint file/
    # When visualize is called the checkpoint file would be read and the Results class would be
    # restored. I am omitting these steps as they are not relevant to the visualize work and
    # you can assume that when visualize is called the Results class will be passed in

    # Now I will demonstrate how Results and RunConfig can be utilized via the APIs

    # Results is a list of RunConfigs sorted by objective - for my "fake" config I've
    # set the default to be throughput. Results is always sorted based on objective with
    # the first entry being the best
    print("\nExample 1 - Objective is highest throughput:")
    for run_config in results.run_configs:
        print_run_config(run_config)

    # Now lets change the objective to latency
    results.set_perf_metric_objectives({"test_model": {PerfLatencyP99.tag: 1}})

    print("\nExample 2 - Objective is lowest latency:")
    for run_config in results.run_configs:
        print_run_config(run_config)

    # Now lets set the objective back to throughput, but place a constraint that latency has to
    # be below a certain value
    results.set_perf_metric_objectives({"test_model": {PerfThroughput.tag: 1}})

    model_constraints = ModelConstraints({PerfLatencyP99.tag: 70})
    run_constraints = RunConstraints({"test_model": model_constraints})
    results.set_constraints(run_constraints)

    print("\nExample 3 - Objective is throughput w/ a latency constraint of 70 ms:")
    for run_config in results.get_results_passing_constraints().run_configs:
        print_run_config(run_config)


if __name__ == "__main__":
    sys.exit(main())
