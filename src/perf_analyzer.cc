// Copyright 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "perf_analyzer.h"

#include "custom_request_schedule_manager.h"
#include "inference_load_mode.h"
#include "perf_analyzer_exception.h"
#include "periodic_concurrency_manager.h"
#include "report_writer.h"
#include "request_rate_manager.h"
#include "session_concurrency/session_concurrency_manager.h"

namespace pa = triton::perfanalyzer;

namespace triton::perfanalyzer {

volatile bool early_exit = false;

void
SignalHandler(int signum)
{
  std::cout << "Interrupt signal (" << signum << ") received." << std::endl;
  // Upon invoking the SignalHandler for the first time early_exit flag is
  // invoked and analyzer waits for in-flight inferences to complete before
  // exiting. On the second invocation, the program exits immediately.
  if (!early_exit) {
    std::cout << "Waiting for in-flight inferences to complete." << std::endl;
    early_exit = true;
  } else {
    std::cout << "Exiting immediately..." << std::endl;
    exit(0);
  }
}
}  // namespace triton::perfanalyzer

PerfAnalyzer::PerfAnalyzer(pa::PAParamsPtr params) : params_(params)
{
  CreateAnalyzerObjects();
}

void
PerfAnalyzer::Run()
{
  PrerunReport();
  Profile();
  WriteReport();
  GenerateProfileExport();
  Finalize();
}

void
PerfAnalyzer::CreateAnalyzerObjects()
{
  // trap SIGINT to allow threads to exit gracefully
  signal(SIGINT, pa::SignalHandler);
  std::shared_ptr<cb::ClientBackendFactory> factory;
  FAIL_IF_ERR(
      cb::ClientBackendFactory::Create(
          params_->kind, params_->url, params_->endpoint, params_->protocol,
          params_->ssl_options, params_->trace_options,
          params_->compression_algorithm, params_->http_headers,
          params_->triton_server_path, params_->model_repository_path,
          params_->extra_verbose, params_->metrics_url,
          params_->input_tensor_format, params_->output_tensor_format,
          params_->grpc_method, &factory),
      "failed to create client factory");

  FAIL_IF_ERR(
      factory->CreateClientBackend(&backend_),
      "failed to create triton client backend");

  parser_ = std::make_shared<pa::ModelParser>(params_->inference_load_mode);
  if (params_->kind == cb::BackendKind::TRITON ||
      params_->kind == cb::BackendKind::TRITON_C_API) {
    rapidjson::Document model_metadata;
    FAIL_IF_ERR(
        backend_->ModelMetadata(
            &model_metadata, params_->model_name, params_->model_version),
        "failed to get model metadata");
    rapidjson::Document model_config;
    FAIL_IF_ERR(
        backend_->ModelConfig(
            &model_config, params_->model_name, params_->model_version),
        "failed to get model config");

    FAIL_IF_ERR(
        parser_->InitTriton(
            model_metadata, model_config, params_->model_version,
            params_->bls_composing_models, params_->input_shapes, backend_),
        "failed to create model parser");
  } else if (params_->kind == cb::BackendKind::DYNAMIC_GRPC) {
    FAIL_IF_ERR(
        parser_->InitDynamicGrpc(
            params_->model_name, params_->model_version, params_->batch_size),
        "failed to create model parser");
  } else if (params_->kind == cb::BackendKind::OPENAI) {
    FAIL_IF_ERR(
        parser_->InitOpenAI(
            params_->model_name, params_->model_version, params_->batch_size),
        "failed to create model parser");
  } else if (params_->kind == cb::BackendKind::TENSORFLOW_SERVING) {
    rapidjson::Document model_metadata;
    FAIL_IF_ERR(
        backend_->ModelMetadata(
            &model_metadata, params_->model_name, params_->model_version),
        "failed to get model metadata");
    FAIL_IF_ERR(
        parser_->InitTFServe(
            model_metadata, params_->model_name, params_->model_version,
            params_->model_signature_name, params_->batch_size,
            params_->input_shapes, backend_),
        "failed to create model parser");
  } else if (params_->kind == cb::BackendKind::TORCHSERVE) {
    FAIL_IF_ERR(
        parser_->InitTorchServe(
            params_->model_name, params_->model_version, params_->batch_size),
        "failed to create model parser");
  } else {
    std::cerr << "unsupported client backend kind" << std::endl;
    throw pa::PerfAnalyzerException(pa::GENERIC_ERROR);
  }

  if ((parser_->MaxBatchSize() == 0) && params_->batch_size > 1) {
    std::cerr << "can not specify batch size > 1 as the model does not support "
                 "batching"
              << std::endl;
    throw pa::PerfAnalyzerException(pa::GENERIC_ERROR);
  }

  // Change the default value for the --async option for sequential models
  if ((parser_->SchedulerType() == pa::ModelParser::SEQUENCE) ||
      (parser_->SchedulerType() == pa::ModelParser::ENSEMBLE_SEQUENCE)) {
    if (!params_->async) {
      params_->async = params_->forced_sync ? false : true;
    }
    // Validate the batch_size specification
    if (params_->batch_size > 1) {
      std::cerr << "can not specify batch size > 1 when using a sequence model"
                << std::endl;
      throw pa::PerfAnalyzerException(pa::GENERIC_ERROR);
    }
  }

  if (params_->streaming && params_->kind != cb::BackendKind::DYNAMIC_GRPC) {
    if (params_->forced_sync) {
      std::cerr << "can not use streaming with synchronous API" << std::endl;
      throw pa::PerfAnalyzerException(pa::GENERIC_ERROR);
    }
    params_->async = true;
  }

  if (parser_->IsDecoupled() &&
      (params_->async == false || params_->streaming == false)) {
    throw pa::PerfAnalyzerException(
        "Decoupled models must be run with `--async` and `--streaming` and "
        "either `-i grpc` or `--service-kind=triton_c_api`");
  }

  const uint64_t max_concurrency{std::max(
      params_->concurrency_range.start, params_->concurrency_range.end)};

  std::unique_ptr<pa::LoadManager> manager;
  if (params_->inference_load_mode == pa::InferenceLoadMode::Concurrency) {
    if ((parser_->SchedulerType() == pa::ModelParser::SEQUENCE) ||
        (parser_->SchedulerType() == pa::ModelParser::ENSEMBLE_SEQUENCE)) {
      if (params_->concurrency_range.end == pa::NO_LIMIT && params_->async) {
        std::cerr << "The 'end' concurrency can not be 0 for sequence "
                     "models when using asynchronous API."
                  << std::endl;
        throw pa::PerfAnalyzerException(pa::GENERIC_ERROR);
      }
    }

    if (!params_->async) {
      if (params_->concurrency_range.end == pa::NO_LIMIT) {
        std::cerr
            << "WARNING: The maximum attainable concurrency will be limited by "
               "max_threads specification."
            << std::endl;
        params_->concurrency_range.end = params_->max_threads;
      } else {
        // As only one synchronous request can be generated from a thread at a
        // time, to maintain the requested concurrency, that many threads need
        // to be generated.
        if (params_->max_threads_specified) {
          std::cerr
              << "WARNING: Overriding max_threads specification to ensure "
                 "requested concurrency range."
              << std::endl;
        }
        params_->max_threads = std::max(
            params_->concurrency_range.start, params_->concurrency_range.end);
      }
    }
    if ((params_->sequence_id_range != 0) &&
        (params_->sequence_id_range < max_concurrency)) {
      std::cerr << "sequence id range specified is smaller than the "
                << "maximum possible concurrency, sequence id collision may "
                << "occur." << std::endl;
      throw pa::PerfAnalyzerException(pa::GENERIC_ERROR);
    }
    FAIL_IF_ERR(
        pa::ConcurrencyManager::Create(
            params_->async, params_->streaming, params_->batch_size,
            params_->max_threads, max_concurrency, params_->shared_memory_type,
            params_->output_shm_size, parser_, factory, &manager,
            params_->request_parameters),
        "failed to create concurrency manager");
  } else if (
      params_->inference_load_mode == pa::InferenceLoadMode::RequestRate ||
      params_->inference_load_mode == pa::InferenceLoadMode::FixedSchedule) {
    if ((params_->sequence_id_range != 0) &&
        (params_->sequence_id_range < params_->num_of_sequences)) {
      std::cerr
          << "sequence id range specified is smaller than the "
          << "maximum possible number of sequences, sequence id collision "
          << "may occur." << std::endl;
      throw pa::PerfAnalyzerException(pa::GENERIC_ERROR);
    }
    if (params_->inference_load_mode == pa::InferenceLoadMode::FixedSchedule) {
      FAIL_IF_ERR(
          pa::CustomRequestScheduleManager::Create(
              *params_, parser_, factory, &manager),
          "failed to create custom request schedule manager");
    } else {
      FAIL_IF_ERR(
          pa::RequestRateManager::Create(
              params_->async, params_->streaming,
              params_->measurement_window_ms, params_->max_trials,
              params_->request_distribution, params_->batch_size,
              params_->max_threads, params_->num_of_sequences,
              params_->shared_memory_type, params_->output_shm_size,
              params_->serial_sequences, parser_, factory, &manager,
              params_->request_parameters),
          "failed to create request rate manager");
    }
  } else if (
      params_->inference_load_mode == pa::InferenceLoadMode::CustomIntervals) {
    if ((params_->sequence_id_range != 0) &&
        (params_->sequence_id_range < params_->num_of_sequences)) {
      std::cerr
          << "sequence id range specified is smaller than the "
          << "maximum possible number of sequences, sequence id collision "
          << "may occur." << std::endl;
      throw pa::PerfAnalyzerException(pa::GENERIC_ERROR);
    }
    FAIL_IF_ERR(
        pa::CustomLoadManager::Create(
            params_->async, params_->streaming, params_->measurement_window_ms,
            params_->max_trials, params_->request_intervals_file,
            params_->batch_size, params_->max_threads,
            params_->num_of_sequences, params_->shared_memory_type,
            params_->output_shm_size, params_->serial_sequences, parser_,
            factory, &manager, params_->request_parameters),
        "failed to create custom load manager");
  } else if (
      params_->inference_load_mode ==
      pa::InferenceLoadMode::PeriodicConcurrency) {
    manager = std::make_unique<pa::PeriodicConcurrencyManager>(
        params_->async, params_->streaming, params_->batch_size,
        params_->max_threads, max_concurrency, params_->shared_memory_type,
        params_->output_shm_size, parser_, factory,
        params_->periodic_concurrency_range, params_->request_period,
        params_->request_parameters);
  } else if (
      params_->inference_load_mode ==
      pa::InferenceLoadMode::SessionConcurrency) {
    manager = std::make_unique<pa::SessionConcurrencyManager>(
        params_->async, params_->streaming, params_->batch_size,
        params_->max_threads, params_->shared_memory_type,
        params_->output_shm_size, parser_, factory, params_->request_parameters,
        params_->session_concurrency);
  }

  manager->InitManager(
      params_->string_length, params_->string_data, params_->zero_input,
      params_->user_data, params_->start_sequence_id,
      params_->sequence_id_range, params_->sequence_length,
      params_->sequence_length_specified, params_->sequence_length_variation);

  FAIL_IF_ERR(
      pa::ProfileDataCollector::Create(&collector_),
      "failed to create profile data collector");

  FAIL_IF_ERR(
      pa::ProfileDataExporter::Create(&exporter_),
      "failed to create profile data exporter");

  FAIL_IF_ERR(
      pa::InferenceProfiler::Create(
          params_->verbose, params_->stability_threshold,
          params_->measurement_window_ms, params_->max_trials,
          params_->percentile, params_->latency_threshold_ms, params_->protocol,
          parser_, std::move(backend_), std::move(manager), &profiler_,
          params_->measurement_request_count, params_->measurement_mode,
          params_->mpi_driver, params_->metrics_interval_ms,
          params_->should_collect_metrics, params_->overhead_pct_threshold,
          params_->async, collector_, !params_->profile_export_file.empty()),
      "failed to create profiler");
}

void
PerfAnalyzer::PrerunReport()
{
  std::cout << "*** Measurement Settings ***" << std::endl;
  if (params_->kind == cb::BackendKind::TRITON || params_->using_batch_size) {
    std::cout << "  Batch size: " << params_->batch_size << std::endl;
  }

  std::cout << "  Service Kind: " << BackendKindToString(params_->kind)
            << std::endl;

  if (params_->warmup_request_count > 0) {
    std::cout << "  Sending " << params_->warmup_request_count
              << " warmup request"
              << (params_->warmup_request_count > 1 ? "s" : "") << std::endl;
  }

  if (params_->request_count != 0) {
    std::cout << "  Sending " << params_->request_count << " benchmark request"
              << (params_->request_count > 1 ? "s" : "") << std::endl;
  } else {
    if (params_->measurement_mode == pa::MeasurementMode::COUNT_WINDOWS) {
      std::cout << "  Using \"count_windows\" mode for stabilization"
                << std::endl;
    } else {
      std::cout << "  Using \"time_windows\" mode for stabilization"
                << std::endl;
    }

    std::string stabilization_metric = "latency and throughput";
    if (params_->async) {
      stabilization_metric = "throughput";
    }
    if (params_->percentile == -1) {
      std::cout << "  Stabilizing using average " << stabilization_metric
                << std::endl;
    } else {
      std::cout << "  Stabilizing using p" << params_->percentile
                << stabilization_metric << std::endl;
    }

    if (params_->measurement_mode == pa::MeasurementMode::TIME_WINDOWS) {
      std::cout << "  Measurement window: " << params_->measurement_window_ms
                << " msec" << std::endl;
    } else if (
        params_->measurement_mode == pa::MeasurementMode::COUNT_WINDOWS) {
      std::cout << "  Minimum number of samples in each window: "
                << params_->measurement_request_count << std::endl;
    }
  }

  if (params_->concurrency_range.end != 1) {
    std::cout << "  Latency limit: " << params_->latency_threshold_ms << " msec"
              << std::endl;
    if (params_->concurrency_range.end != pa::NO_LIMIT) {
      std::cout << "  Concurrency limit: "
                << std::max(
                       params_->concurrency_range.start,
                       params_->concurrency_range.end)
                << " concurrent requests" << std::endl;
    }
  }
  if (params_->request_rate_range[pa::SEARCH_RANGE::kEND] != 1.0) {
    std::cout << "  Latency limit: " << params_->latency_threshold_ms << " msec"
              << std::endl;
    if (params_->request_rate_range[pa::SEARCH_RANGE::kEND] !=
        static_cast<double>(pa::NO_LIMIT)) {
      std::cout << "  Request Rate limit: "
                << std::max(
                       params_->request_rate_range[pa::SEARCH_RANGE::kSTART],
                       params_->request_rate_range[pa::SEARCH_RANGE::kEND])
                << " requests per seconds" << std::endl;
    }
  }
  if (params_->inference_load_mode == pa::InferenceLoadMode::RequestRate) {
    if (params_->request_distribution == pa::Distribution::POISSON) {
      std::cout << "  Using poisson distribution on request generation"
                << std::endl;
    } else {
      std::cout << "  Using uniform distribution on request generation"
                << std::endl;
    }
  }
  if (params_->search_mode == pa::SearchMode::BINARY) {
    std::cout << "  Using Binary Search algorithm" << std::endl;
  }
  if (params_->async) {
    std::cout << "  Using asynchronous calls for inference" << std::endl;
  } else {
    std::cout << "  Using synchronous calls for inference" << std::endl;
  }
  if (parser_->IsDecoupled()) {
    std::cout << "  Detected decoupled model, using the first response for "
                 "measuring latency"
              << std::endl;
  }

  std::cout << std::endl;
}

void
PerfAnalyzer::Profile()
{
  params_->mpi_driver->MPIBarrierWorld();

  cb::Error err;
  if (params_->inference_load_mode == pa::InferenceLoadMode::Concurrency) {
    err = profiler_->Profile<size_t>(
        params_->concurrency_range.start, params_->concurrency_range.end,
        params_->concurrency_range.step, params_->search_mode,
        params_->warmup_request_count, params_->request_count, perf_statuses_);
  } else if (
      params_->inference_load_mode == pa::InferenceLoadMode::RequestRate ||
      params_->inference_load_mode == pa::InferenceLoadMode::CustomIntervals ||
      params_->inference_load_mode == pa::InferenceLoadMode::FixedSchedule) {
    err = profiler_->Profile<double>(
        params_->request_rate_range[pa::SEARCH_RANGE::kSTART],
        params_->request_rate_range[pa::SEARCH_RANGE::kEND],
        params_->request_rate_range[pa::SEARCH_RANGE::kSTEP],
        params_->search_mode, params_->warmup_request_count,
        params_->request_count, perf_statuses_);
  } else if (
      params_->inference_load_mode ==
      pa::InferenceLoadMode::PeriodicConcurrency) {
    err = profiler_->BenchmarkPeriodicConcurrencyMode();
  } else if (
      params_->inference_load_mode ==
      pa::InferenceLoadMode::SessionConcurrency) {
    err = profiler_->BenchmarkSessionConcurrencyMode();
  }

  params_->mpi_driver->MPIBarrierWorld();

  if (!err.IsOk()) {
    std::cerr << err;
    // In the case of early_exit, the thread does not return and continues to
    // report the summary
    if (!pa::early_exit) {
      throw pa::PerfAnalyzerException(err.Err());
    }
  }
}

void
PerfAnalyzer::WriteReport()
{
  if (!perf_statuses_.size() ||
      params_->inference_load_mode ==
          pa::InferenceLoadMode::PeriodicConcurrency) {
    return;
  }

  // Can print more depending on verbose, but it seems too much information
  std::cout << "Inferences/Second vs. Client ";
  if (params_->percentile == -1) {
    std::cout << "Average Batch Latency" << std::endl;
  } else {
    std::cout << "p" << params_->percentile << " Batch Latency" << std::endl;
  }

  for (pa::PerfStatus& status : perf_statuses_) {
    if (params_->inference_load_mode == pa::InferenceLoadMode::Concurrency) {
      std::cout << "Concurrency: " << status.concurrency << ", ";
    } else {
      std::cout << "Request Rate: " << status.request_rate << ", ";
    }
    std::cout << "throughput: " << status.client_stats.infer_per_sec
              << " infer/sec, latency "
              << (status.stabilizing_latency_ns / 1000) << " usec" << std::endl;
  }

  bool should_output_metrics{
      params_->should_collect_metrics && params_->verbose_csv};

  std::unique_ptr<pa::ReportWriter> writer;

  FAIL_IF_ERR(
      pa::ReportWriter::Create(
          params_->filename,
          params_->inference_load_mode == pa::InferenceLoadMode::Concurrency,
          perf_statuses_, params_->verbose_csv, profiler_->IncludeServerStats(),
          params_->percentile, parser_, &writer, should_output_metrics),
      "failed to create report writer");

  writer->GenerateReport();
}

void
PerfAnalyzer::GenerateProfileExport()
{
  if (!params_->profile_export_file.empty()) {
    exporter_->Export(
        collector_->GetData(), collector_->GetVersion(),
        params_->profile_export_file, params_->kind, params_->endpoint);
  }
}

void
PerfAnalyzer::Finalize()
{
  params_->mpi_driver->MPIFinalize();
}
