// Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "concurrency_manager.h"
#include "constants.h"
#include "custom_load_manager.h"
#include "custom_request_schedule_manager.h"
#include "metrics.h"
#include "metrics_manager.h"
#include "model_parser.h"
#include "mpi_utils.h"
#include "periodic_concurrency_manager.h"
#include "profile_data_collector.h"
#include "request_rate_manager.h"
#include "session_concurrency/session_concurrency_manager.h"

namespace triton::perfanalyzer {

#ifndef DOCTEST_CONFIG_DISABLE
class NaggyMockInferenceProfiler;
class TestInferenceProfiler;
class ModelParser;
#endif

/// Constant parameters that determine the whether stopping criteria has met
/// for the current phase of testing
struct LoadParams {
  // The number of measurements to account for during calculation of load
  // status
  uint32_t stability_window;
  // The +/- range to account for while assessing load status
  double stability_threshold;
};

/// Data structure to keep track of real-time load status and determine whether
/// stopping criteria has met for the current phase of testing.
struct LoadStatus {
  // Stores the observations of infer_per_sec and latencies in a vector
  std::vector<double> infer_per_sec;
  std::vector<uint64_t> latencies;
  // Records the average inference per second within the stability window
  double avg_ips = 0;
  // Stores the average latency within the stability window
  uint64_t avg_latency = 0;
};

/// Configuration for the Measure function
struct MeasureConfig {
  uint64_t measurement_window{0};
  bool is_count_based{false};
  bool clamp_window{false};
};

// Holds the total of the timiming components of composing models of an
// ensemble.
struct EnsembleDurations {
  EnsembleDurations()
      : total_queue_time_avg_us(0), total_compute_time_avg_us(0),
        total_cache_hit_time_avg_us(0), total_cache_miss_time_avg_us(0),
        total_combined_cache_compute_time_avg_us(0)
  {
  }
  uint64_t total_queue_time_avg_us;
  uint64_t total_compute_time_avg_us;
  // Time spent on cache lookups/copies for cache hits
  uint64_t total_cache_hit_time_avg_us;
  // Time spent on cache lookups/copies/insertions for cache misses
  uint64_t total_cache_miss_time_avg_us;

  // Combined average of cache and compute times
  uint64_t total_combined_cache_compute_time_avg_us;
};

/// Holds the server-side inference statisitcs of the target model and its
/// composing models
struct ServerSideStats {
  uint64_t inference_count;
  uint64_t execution_count;
  uint64_t cache_hit_count;
  uint64_t cache_miss_count;
  uint64_t success_count;
  uint64_t queue_count;
  uint64_t compute_input_count;
  uint64_t compute_infer_count;
  uint64_t compute_output_count;
  uint64_t cumm_time_ns;
  uint64_t queue_time_ns;
  uint64_t compute_input_time_ns;
  uint64_t compute_infer_time_ns;
  uint64_t compute_output_time_ns;
  // Time spent on cache lookups/copies for cache hits
  uint64_t cache_hit_time_ns;
  // Time spent on cache lookups/copies/insertions for cache misses
  uint64_t cache_miss_time_ns;

  std::map<cb::ModelIdentifier, ServerSideStats> composing_models_stat;
  // This function sets composing model server stats to 0 in case of a cache hit
  // when top level response cache is enabled, since composing models are not
  // executed and do not have any stats
  void Reset()
  {
    inference_count = 0;
    execution_count = 0;
    success_count = 0;
    queue_count = 0;
    compute_input_count = 0;
    compute_infer_count = 0;
    compute_output_count = 0;
    cumm_time_ns = 0;
    queue_time_ns = 0;
    compute_input_time_ns = 0;
    compute_infer_time_ns = 0;
    compute_output_time_ns = 0;
    cache_hit_count = 0;
    cache_hit_time_ns = 0;
    cache_miss_count = 0;
    cache_miss_time_ns = 0;
  }
};

/// Holds the statistics recorded at the client side.
struct ClientSideStats {
  // Request count and elapsed time measured by client
  uint64_t request_count;
  // Only record sequences that finish within the measurement window
  uint64_t sequence_count;
  // The number of requests that missed their schedule
  uint64_t delayed_request_count;
  // The number of responses
  uint64_t response_count;
  uint64_t duration_ns;
  uint64_t avg_latency_ns;
  // a ordered map of percentiles to be reported (<percentile, value> pair)
  std::map<size_t, uint64_t> percentile_latency_ns;
  // List of all the valid latencies.
  std::vector<uint64_t> latencies;
  // Using usec to avoid square of large number (large in nsec)
  uint64_t std_us;
  uint64_t avg_request_time_ns;
  uint64_t avg_send_time_ns;
  uint64_t avg_receive_time_ns;
  // Per sec stat
  double infer_per_sec;
  double responses_per_sec;
  double sequence_per_sec;

  // Completed request count reported by the client library
  uint64_t completed_count;
};

/// The entire statistics record.
struct PerfStatus {
  uint32_t concurrency;
  double request_rate;
  size_t batch_size;
  ServerSideStats server_stats;
  ClientSideStats client_stats;
  std::vector<Metrics> metrics{};
  double overhead_pct;
  bool on_sequence_model;

  // placeholder for the latency value that is used for conditional checking
  uint64_t stabilizing_latency_ns;
  // Metric for requests sent per second
  double send_request_rate{0.0};
};

cb::Error ReportPrometheusMetrics(const Metrics& metrics);

//==============================================================================
/// A InferenceProfiler is a helper class that measures and summarizes the
/// inference statistic under different concurrency level.
///
/// The profiler can adjust the number of concurrent requests by informing the
/// concurrency manager. And after the adjustment, the profiler will actively
/// collecting the statistic from both the concurrency manager and the inference
/// server directly until it is stable. Once stable, the profiler updates the
/// 'status_summary' based on the most recent measurement.
///
/// The measurement procedure:
/// 1. The profiler gets start status from the server and records the start
/// time.
/// 2. After given time interval, the profiler gets end status from the server
///    and records the end time.
/// 3. The profiler obtains the request records recorded by concurrency manager,
///    and uses the request records that are recorded between start time and end
///    time to measure client side status and update status_summary.
///
class InferenceProfiler {
 public:
  /// Create a profiler that collects and summarizes inference statistic.
  /// \param verbose Whether to print verbose logging.
  /// \param stability_threshold The range that the measurement is considered as
  /// stable. i.e. within (1 +/- stability_threshold) * average value of the
  /// last 3 measurements. The criteria are "infer per second" and "average
  /// latency", or "infer per second" and "percentile latency" if valid
  /// percentile is set (see 'percentile' below).
  /// \param measurement_window_ms The duration of each measurement in msec.
  /// \param max_trials The maximum number of attempts to obtain
  /// stable measurement.
  /// \param percentile The percentile in terms of latency to be reported.
  /// if it is a valid percentile value, the percentile latency will reported
  /// and used as stable criteria instead of average latency. If it is -1,
  /// average latency will be reported and used as stable criteria.
  /// \param latency_threshold_ms The threshold on the latency measurements in
  /// microseconds.
  /// \param parser The ModelParse object which holds all the details about the
  /// model.
  /// \param profile_backend The ClientBackend object used to communicate
  /// with the server by profiler.
  /// \param manager The LoadManager object that will produce load on the
  /// server.
  /// \param profiler Returns a new InferenceProfiler object.
  /// \param measurement_request_count The number of requests to capture when
  /// using "count_windows" mode.
  /// \param measurement_mode The measurement mode to use for windows.
  /// \param mpi_driver The driver class for MPI operations.
  /// \param metrics_interval_ms The interval at which the server-side metrics
  /// \param should_collect_metrics Whether server-side inference server metrics
  /// should be collected.
  /// \param overhead_pct_threshold User set threshold above which the PA
  /// overhead is too significant to provide usable results.
  /// \param collector Collector for the profile data from experiments
  /// \param should_collect_profile_data Whether to collect profile data.
  /// \return cb::Error object indicating success or failure.
  static cb::Error Create(
      const bool verbose, const double stability_threshold,
      const uint64_t measurement_window_ms, const size_t max_trials,
      const int64_t percentile, const uint64_t latency_threshold_ms,
      const cb::ProtocolType protocol, std::shared_ptr<ModelParser>& parser,
      std::shared_ptr<cb::ClientBackend> profile_backend,
      std::unique_ptr<LoadManager> manager,
      std::unique_ptr<InferenceProfiler>* profiler,
      uint64_t measurement_request_count, MeasurementMode measurement_mode,
      std::shared_ptr<MPIDriver> mpi_driver, const uint64_t metrics_interval_ms,
      const bool should_collect_metrics, const double overhead_pct_threshold,
      const bool async_mode,
      const std::shared_ptr<ProfileDataCollector> collector,
      const bool should_collect_profile_data);

  /// Performs the profiling on the given range with the given search algorithm.
  /// For profiling using request rate invoke template with double, otherwise
  /// invoke with size_t for concurrency search.
  /// \param start The starting point of the search range.
  /// \param end The ending point of the search range.
  /// \param step The step size to move along the search range in linear search
  /// or the precision in binary search.
  /// \param search_mode The search algorithm to be applied.
  /// \param warmup_request_count The number of warmup requests to send before
  /// benchmarking.
  /// \param request_count The number of requests to generate in each
  /// experiment. If 0, then there is no limit, and it will generate until
  /// stable.
  /// \param summary Returns the trace of the measurement along the search path.
  /// \return cb::Error object indicating success or failure.
  template <typename T>
  cb::Error Profile(
      const T start, const T end, const T step, const SearchMode search_mode,
      size_t warmup_request_count, const size_t request_count,
      std::vector<PerfStatus>& perf_statuses)
  {
    cb::Error err;
    bool meets_threshold, is_stable;
    if (search_mode == SearchMode::NONE) {
      err = Profile(
          warmup_request_count, request_count, perf_statuses, meets_threshold,
          is_stable);
      if (!err.IsOk()) {
        return err;
      }
    } else if (search_mode == SearchMode::LINEAR) {
      T current_value = start;
      do {
        err = Profile(
            current_value, warmup_request_count, request_count, perf_statuses,
            meets_threshold, is_stable);
        if (!err.IsOk()) {
          return err;
        }
        current_value += step;
      } while (((current_value <= end) || (end == static_cast<T>(NO_LIMIT))) &&
               (meets_threshold));
      // If there was only one concurrency we swept over and it did not meet the
      // stability threshold, we should return an error.
      if (current_value == (start + step) && is_stable == false) {
        return cb::Error(
            "Failed to obtain stable measurement.", pa::STABILITY_ERROR);
      }
    } else {
      err = Profile(
          start, warmup_request_count, request_count, perf_statuses,
          meets_threshold, is_stable);
      if (!err.IsOk() || (!meets_threshold)) {
        return err;
      }
      err = Profile(
          end, warmup_request_count, request_count, perf_statuses,
          meets_threshold, is_stable);
      if (!err.IsOk() || (meets_threshold)) {
        return err;
      }

      T this_start = start;
      T this_end = end;
      while ((this_end - this_start) > step) {
        T current_value = (this_end + this_start) / 2;
        err = Profile(
            current_value, warmup_request_count, request_count, perf_statuses,
            meets_threshold, is_stable);
        if (!err.IsOk()) {
          return err;
        }
        if (meets_threshold) {
          this_start = current_value;
        } else {
          this_end = current_value;
        }
      }
    }
    return cb::Error::Success;
  }

  cb::Error BenchmarkPeriodicConcurrencyMode()
  {
    auto& manager{dynamic_cast<PeriodicConcurrencyManager&>(*manager_)};
    std::vector<RequestRecord> request_records{manager.RunExperiment()};
    // FIXME - Refactor collector class to not need ID or window in the case of
    // periodic concurrency mode
    ProfileDataCollector::InferenceLoadMode id{1, 0.0};
    collector_->AddWindow(id, 0, std::numeric_limits<uint64_t>::max());
    collector_->AddData(id, std::move(request_records));
    return cb::Error::Success;
  }

  cb::Error BenchmarkSessionConcurrencyMode()
  {
    auto& manager{dynamic_cast<SessionConcurrencyManager&>(*manager_)};
    std::vector<RequestRecord> request_records{manager.Start()};
    ProfileDataCollector::InferenceLoadMode id{0, 0.0};
    collector_->AddWindow(id, 0, std::numeric_limits<uint64_t>::max());
    collector_->AddData(id, std::move(request_records));
    return cb::Error::Success;
  }

  bool IncludeServerStats() { return include_server_stats_; }

 private:
  InferenceProfiler(
      const bool verbose, const double stability_threshold,
      const int32_t measurement_window_ms, const size_t max_trials,
      const bool extra_percentile, const size_t percentile,
      const uint64_t latency_threshold_ms, const cb::ProtocolType protocol,
      std::shared_ptr<ModelParser>& parser,
      std::shared_ptr<cb::ClientBackend> profile_backend,
      std::unique_ptr<LoadManager> manager, uint64_t measurement_request_count,
      MeasurementMode measurement_mode, std::shared_ptr<MPIDriver> mpi_driver,
      const uint64_t metrics_interval_ms, const bool should_collect_metrics,
      const double overhead_pct_threshold, const bool async_mode,
      const std::shared_ptr<ProfileDataCollector> collector,
      const bool should_collect_profile_data);

  /// Actively measure throughput in every 'measurement_window' msec until the
  /// throughput is stable. Once the throughput is stable, it adds the
  /// observations on summary trace and returns whether the setting met the
  /// threshold. NOTE: the requests are being sent regardless of the
  /// measurement, so the data returned by the server (see struct
  /// PerforamnceStatusStruct) will include more requests than what the client
  /// measures (we can't get the exact server status right before the first
  /// request and right after the last request in the measurement window).
  /// \param concurrent_request_count The concurrency level for the measurement.
  /// \param perf_statuses Appends the measurements summary at the end of this
  /// list.
  /// \param warmup_request_count The number of warmup requests to send before
  /// benchmarking.
  /// \param request_count The number of requests to generate when profiling. If
  /// 0, then there is no limit, and it will generate until stable.
  /// \param meets_threshold Returns whether the setting meets the
  /// threshold.
  /// \param is_stable Returns whether the measurement is stable.
  /// \return cb::Error object indicating success or failure.
  cb::Error Profile(
      const size_t concurrent_request_count, size_t warmup_request_count,
      const size_t request_count, std::vector<PerfStatus>& perf_statuses,
      bool& meets_threshold, bool& is_stable);

  /// Similar to above function, but instead of setting the concurrency, it
  /// sets the specified request rate for measurements.
  /// \param request_rate The request rate for inferences.
  /// \param warmup_request_count The number of warmup requests to send before
  /// benchmarking.
  /// \param request_count The number of requests to generate when profiling. If
  /// 0, then there is no limit, and it will generate until stable.
  /// \param perf_statuses Appends the measurements summary at the end of this
  /// list.
  /// \param meets_threshold Returns whether the setting meets the
  /// threshold.
  /// \param is_stable Returns whether the measurement is stable.
  /// \return cb::Error object indicating success or failure.
  cb::Error Profile(
      const double request_rate, size_t warmup_request_count,
      const size_t request_count, std::vector<PerfStatus>& perf_statuses,
      bool& meets_threshold, bool& is_stable);

  /// Measures throughput and latencies for custom load without controlling
  /// request rate nor concurrency. Requires load manager to be loaded with
  /// a file specifying the time intervals.
  /// \param warmup_request_count The number of warmup requests to send before
  /// benchmarking.
  /// \param request_count The number of requests to generate when profiling. If
  /// 0, then there is no limit, and it will generate until stable.
  /// \param perf_statuses Appends the measurements summary at the end of this
  /// list.
  /// \param meets_threshold Returns whether the measurement met the
  /// threshold.
  /// \param is_stable Returns whether the measurement is stable.
  /// \return cb::Error object indicating success
  /// or failure.
  cb::Error Profile(
      size_t warmup_request_count, const size_t request_count,
      std::vector<PerfStatus>& perf_statuses, bool& meets_threshold,
      bool& is_stable);


  /// A helper function for profiling functions.
  /// \param status_summary Returns the summary of the measurement.
  /// \param request_count The number of requests to generate when profiling. If
  /// 0, then there is no limit, and it will generate until stable.
  /// \param is_stable Returns whether the measurement stabilized or not.
  /// \return cb::Error object indicating success or failure.
  cb::Error ProfileHelper(
      PerfStatus& status_summary, size_t request_count, bool* is_stable);

  /// A helper function to determine if profiling is stable
  /// \param load_status Stores the observations of infer_per_sec and latencies
  /// \param check_latency Whether to check latency for stability
  /// \return Returns if the threshold and latencies are stable.
  bool DetermineStability(LoadStatus& load_status, bool check_latency = true);

  /// Check if latency at index idx is within the latency threshold
  /// \param idx index in latency vector
  /// \param load_status Stores the observations of infer_per_sec and latencies
  /// \return Returns whether the latencies are below the max threshold
  bool CheckWithinThreshold(size_t idx, LoadStatus& load_status);

  /// A helper function to determine if profiling is done
  /// \param load_status Stores the observations of infer_per_sec and latencies
  /// \param is_stable Returns whether the measurement stabilized or not.
  /// \return Returns if we should break out of the infinite stability check
  /// loop.
  bool IsDoneProfiling(LoadStatus& load_status, bool* is_stable);

  /// Check if observed inferences and latencies are within threshold
  /// for a single window starting at idx
  /// \param idx index in latency vector
  /// \param load_status Stores the observations of infer_per_sec and latencies
  /// \param check_latency Whether to check latency for stability
  /// \return Returns whether inference and latency are stable
  bool CheckWindowForStability(
      size_t idx, LoadStatus& load_status, bool check_latency);

  /// Check if observed inferences are within threshold
  /// for a single window starting at idx
  /// \param idx index in latency vector
  /// \param load_status Stores the observations of infer_per_sec and latencies
  /// \return Returns whether inference is stable
  bool IsInferWindowStable(size_t idx, LoadStatus& load_status);

  /// Check if observed latencies are within threshold
  /// for a single window starting at idx
  /// \param idx index in latency vector
  /// \param load_status Stores the observations of infer_per_sec and latencies
  /// \return Returns whether latency is stable
  bool IsLatencyWindowStable(size_t idx, LoadStatus& load_status);

  /// Helper function to perform measurement.
  /// \param status_summary The summary of this measurement.
  /// \param config The configuration for measurement.
  /// \return cb::Error object indicating success or failure.
  cb::Error Measure(PerfStatus& status_summary, MeasureConfig config);

  /// Gets the server side statistics
  /// \param model_status Returns the status of the models provided by
  /// the server. If the model being profiled is non-ensemble model,
  /// only its status will be returned. Otherwise, the status of the composing
  /// models will also be returned.
  /// \return cb::Error object indicating success or failure.
  cb::Error GetServerSideStatus(
      std::map<cb::ModelIdentifier, cb::ModelStatistics>* model_status);

  /// Summarize the measurement with the provided statistics.
  /// \param start_status The model status at the start of the measurement.
  /// \param end_status The model status at the end of the measurement.
  /// \param start_stat The accumulated context status at the start.
  /// \param end_stat The accumulated context status at the end.
  /// \param summary Returns the summary of the measurement.
  /// \param window_start_ns The window start timestamp in nanoseconds.
  /// \param window_end_ns The window end timestamp in nanoseconds.
  /// \param clamp_window If true, the actual window range is reduced to the
  /// start of the first request to the final response.
  /// \return cb::Error object indicating success or failure.
  cb::Error Summarize(
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
      const cb::InferStat& start_stat, const cb::InferStat& end_stat,
      PerfStatus& summary, uint64_t window_start_ns, uint64_t window_end_ns,
      bool clamp_window);

  /// \param valid_range The start and end timestamp of the measurement window.
  /// \param valid_sequence_count Returns the number of completed sequences
  /// during the measurement. A sequence is a set of correlated requests sent to
  /// sequence model.
  /// \param latencies Returns the vector of request latencies where the
  /// requests are completed within the measurement window.
  /// \param response_count Returns the number of responses
  /// \param valid_requests Returns a vector of valid request records
  virtual void ValidLatencyMeasurement(
      const std::pair<uint64_t, uint64_t>& valid_range,
      size_t& valid_sequence_count, size_t& delayed_request_count,
      std::vector<uint64_t>* latencies, size_t& response_count,
      std::vector<RequestRecord>& valid_requests);

  /// Clamp a window around a set of requests, from the earliest start time to
  /// the latest response
  /// \param requests A vector of requests to clamp the window around.
  /// \return std::pair object containing <start, end> of the window.
  std::pair<uint64_t, uint64_t> ClampWindow(
      std::vector<RequestRecord>& requests);

  /// Add the data from the request records to the Raw Data Collector
  /// \param perf_status PerfStatus of the current measurement
  /// \param window_start_ns The window start timestamp in nanoseconds.
  /// \param window_end_ns The window end timestamp in nanoseconds.
  /// \param request_records The request records to collect.
  void CollectData(
      PerfStatus& perf_status, uint64_t window_start_ns, uint64_t window_end_ns,
      std::vector<RequestRecord>&& request_records);

  /// \param latencies The vector of request latencies collected.
  /// \param summary Returns the summary that the latency related fields are
  /// set.
  /// \return cb::Error object indicating success or failure.
  virtual cb::Error SummarizeLatency(
      const std::vector<uint64_t>& latencies, PerfStatus& summary);

  /// \param latencies The vector of request latencies collected.
  /// \return std::tuple object containing:
  ///   * mean of latencies in nanoseconds
  ///   * sample standard deviation of latencies in microseconds
  std::tuple<uint64_t, uint64_t> GetMeanAndStdDev(
      const std::vector<uint64_t>& latencies);

  /// \param start_stat The accumulated client statistics at the start.
  /// \param end_stat The accumulated client statistics at the end.
  /// \param duration_ns The duration of the measurement in nsec.
  /// \param valid_request_count The number of completed requests recorded.
  /// \param valid_sequence_count The number of completed sequences recorded.
  /// \param delayed_request_count The number of requests that missed their
  /// schedule.
  /// \param response_count The number of responses.
  /// \param summary Returns the summary that the fields recorded by
  /// client are set.
  /// \return cb::Error object indicating success or failure.
  virtual cb::Error SummarizeClientStat(
      const cb::InferStat& start_stat, const cb::InferStat& end_stat,
      const uint64_t duration_ns, const size_t valid_request_count,
      const size_t delayed_request_count, const size_t valid_sequence_count,
      const size_t response_count, PerfStatus& summary);

  /// Adds the send request rate metric to the summary object.
  /// \param window_duration_s The duration of the window in seconds.
  /// \param num_sent_requests The number of requests sent during the last
  /// window.
  /// \param summary The summary object to be updated with the send request rate
  /// metric.
  void SummarizeSendRequestRate(
      const double window_duration_s, const size_t num_sent_requests,
      PerfStatus& summary);

  /// Given a model_identifier to gather stats for, and a map of ALL stats,
  /// determine which version of the model should be gathered
  /// \param model_identifier A pair of model_name and model_version to identify
  /// a specific model
  /// \param start_stats The stats for all models at the start of the
  ///  measurement
  /// \param end_stats The stats for all models at the end of the measurement
  /// \param model_version The determined model version

  cb::Error DetermineStatsModelVersion(
      const cb::ModelIdentifier& model_identifier,
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_stats,
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_stats,
      int64_t* model_version);

#ifndef DOCTEST_CONFIG_DISABLE
  cb::Error SetTopLevelResponseCaching(bool enable_top_level_request_caching);
#endif

  /// \param start_status The model status at the start of the measurement.
  /// \param end_status The model status at the end of the measurement.
  /// \param server_stats Returns the summary that the fields recorded by server
  /// are set.
  /// \return cb::Error object indicating success or failure.
  cb::Error SummarizeServerStats(
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
      ServerSideStats* server_stats);

  /// \param model_identifier A pair of model_name and model_version to identify
  /// a specific model.
  /// \param start_status The model status at the start of the measurement.
  /// \param end_status The model status at the end of the measurement.
  /// \param server_stats Returns the summary that the fields recorded by server
  /// are set.
  /// \return cb::Error object indicating success or failure.
  cb::Error SummarizeServerStats(
      const cb::ModelIdentifier& model_identifier,
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
      ServerSideStats* server_stats);

  /// \param model_identifier A pair of model_name and model_version to identify
  /// a specific model.
  /// \param start_status The model status at the start of the measurement.
  /// \param end_status The model status at the end of the measurement.
  /// \param server_stats Returns the summary that the fields recorded by server
  /// are set.
  /// \return cb::Error object indicating success or failure.
  cb::Error SummarizeServerStatsHelper(
      const cb::ModelIdentifier& model_identifier,
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& start_status,
      const std::map<cb::ModelIdentifier, cb::ModelStatistics>& end_status,
      ServerSideStats* server_stats);

  /// Calculate the overhead and put the results into the summary
  ///
  /// \param window_duration_ns The duration of the window
  /// \param idle_ns The average worker idle time during the window
  /// \param summary The summary object to be updated with overhead stats
  ///
  void SummarizeOverhead(
      const uint64_t window_duration_ns, const uint64_t idle_ns,
      PerfStatus& summary);

  /// Returns true if all MPI ranks (models) are stable. Should only be run if
  /// and only if IsMPIRun() returns true.
  /// \param current_rank_stability The stability of the current rank.
  /// \return True if all MPI ranks are stable.
  bool AllMPIRanksAreStable(bool current_rank_stability);

  /// Merge individual perf status reports into a single perf status.  This
  /// function is used to merge the results from multiple Measure runs into a
  /// single report.
  /// \param perf_status List of perf status reports to be merged.
  /// \param summary_status Final merged summary status.
  /// \return cb::Error object indicating success or failure.
  virtual cb::Error MergePerfStatusReports(
      std::deque<PerfStatus>& perf_status, PerfStatus& summary_status);

  /// Merge individual server side statistics into a single server side report.
  /// \param server_side_stats List of server side statistics reports to be
  /// merged.
  /// \param server_side_summary Final merged summary status.
  /// \return cb::Error object indicating success or failure.
  virtual cb::Error MergeServerSideStats(
      std::vector<ServerSideStats>& server_side_stats,
      ServerSideStats& server_side_summary);

  /// \param all_metrics Individual metrics from all intervals from stable
  /// passes.
  /// \param merged_metrics Output merged metrics from all intervals from stable
  /// passes.
  /// \return cb::Error object indicating success or failure.
  cb::Error MergeMetrics(
      const std::vector<std::reference_wrapper<const Metrics>>& all_metrics,
      Metrics& merged_metrics);

  template <typename T>
  void GetMetricAveragePerGPU(
      const std::vector<std::reference_wrapper<const std::map<std::string, T>>>&
          input_metric_maps,
      std::map<std::string, T>& output_metric_map)
  {
    std::map<std::string, size_t> metric_count_per_gpu{};

    for (const auto& input_metric_map : input_metric_maps) {
      for (const auto& input_metric : input_metric_map.get()) {
        const auto& gpu_uuid{input_metric.first};
        const auto& metric{input_metric.second};

        if (output_metric_map.find(gpu_uuid) == output_metric_map.end()) {
          output_metric_map[gpu_uuid] = 0;
          metric_count_per_gpu[gpu_uuid] = 0;
        }

        output_metric_map[gpu_uuid] += metric;
        metric_count_per_gpu[gpu_uuid]++;
      }
    }

    for (auto& output_metric : output_metric_map) {
      const auto& gpu_uuid{output_metric.first};
      auto& metric{output_metric.second};
      const auto& metric_count{metric_count_per_gpu[gpu_uuid]};
      if (metric_count > 0) {
        metric /= metric_count;
      }
    }
  }

  template <typename T>
  void GetMetricMaxPerGPU(
      const std::vector<std::reference_wrapper<const std::map<std::string, T>>>&
          input_metric_maps,
      std::map<std::string, T>& output_metric_map)
  {
    for (const auto& input_metric_map : input_metric_maps) {
      for (const auto& input_metric : input_metric_map.get()) {
        const auto& gpu_uuid{input_metric.first};
        const auto& metric{input_metric.second};

        if (output_metric_map.find(gpu_uuid) == output_metric_map.end()) {
          output_metric_map[gpu_uuid] = 0;
        }

        output_metric_map[gpu_uuid] =
            std::max(output_metric_map[gpu_uuid], metric);
      }
    }
  }

  template <typename T>
  void GetMetricFirstPerGPU(
      const std::vector<std::reference_wrapper<const std::map<std::string, T>>>&
          input_metric_maps,
      std::map<std::string, T>& output_metric_map)
  {
    for (const auto& input_metric_map : input_metric_maps) {
      for (const auto& input_metric : input_metric_map.get()) {
        const auto& gpu_uuid{input_metric.first};
        const auto& metric{input_metric.second};

        if (output_metric_map.find(gpu_uuid) == output_metric_map.end()) {
          output_metric_map[gpu_uuid] = metric;
        }
      }
    }
  }

  bool verbose_;
  uint64_t measurement_window_ms_;
  uint64_t measurement_request_count_;
  MeasurementMode measurement_mode_;
  size_t max_trials_;
  bool extra_percentile_;
  size_t percentile_;
  uint64_t latency_threshold_ms_;

  cb::ProtocolType protocol_;
  std::string model_name_;
  int64_t model_version_;

  std::shared_ptr<ModelParser> parser_;
  std::shared_ptr<cb::ClientBackend> profile_backend_;
  std::unique_ptr<LoadManager> manager_;
  std::shared_ptr<ProfileDataCollector> collector_;
  LoadParams load_parameters_;

  bool include_lib_stats_;
  bool include_server_stats_;
  std::shared_ptr<MPIDriver> mpi_driver_;

  /// The request records of the requests completed during all measurements
  std::vector<RequestRecord> all_request_records_;

  /// The end time of the previous measurement window
  uint64_t previous_window_end_ns_;

  /// Server side statistics from the previous measurement window
  std::map<cb::ModelIdentifier, cb::ModelStatistics> prev_server_side_stats_;

  /// Client side statistics from the previous measurement window
  cb::InferStat prev_client_side_stats_;

  /// Metrics manager that collects server-side metrics periodically
  std::shared_ptr<MetricsManager> metrics_manager_{nullptr};

  /// Whether server-side inference server metrics should be collected.
  bool should_collect_metrics_{false};

  /// User set threshold above which the PA overhead is too significant to
  /// provide usable results.
  const double overhead_pct_threshold_{0.0};

  // Whether to collect profile data.
  bool should_collect_profile_data_{false};

  // Whether the client is operating in async mode.
  const bool async_mode_{false};

#ifndef DOCTEST_CONFIG_DISABLE
  friend NaggyMockInferenceProfiler;
  friend TestInferenceProfiler;
  friend ModelParser;

 public:
  InferenceProfiler() = default;
#endif
};

}  // namespace triton::perfanalyzer
