// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "load_manager.h"
#include "request_rate_manager.h"

namespace triton { namespace perfanalyzer {
    
//==============================================================================
/// CustomRequestScheduleManager is a helper class to send inference requests to
/// inference server in accordance with the schedule set by the user.
///
/// Detail:
/// An instance of this load manager will be created at the beginning of the
/// perf analyzer and it will be used to schedule to send requests at that
/// particular second defined by the user. The particular seconds at which a 
/// request should be sent can be set by the user using the `schedule` option.
/// For example, if the `schedule` is set to 1,2,4,5,6.5, CustomRequestScheduleManager
/// sends request at 1st second, 2nd second, 4th second and so on.
/// 

class CustomRequestScheduleManager : public RequestRateManager {
  public:
    ~CustomRequestScheduleManager() = default;

    /// Creates an object of CustomRequestScheduleManager
    /// \param async Whether to use asynchronous or synchronous API for infer request
    /// \param streaming Whether to use gRPC streaming API for infer request
    /// \param measurement_window_ms The time window for measurements
    /// \param max_trials The maximum number of windows that will be measured
    /// \param schedule The vector containing the schedule for requests
    /// \param batch_size The batch size used for each request
    /// \param max_threads The maximum number of working threads to be spawned
    /// \param num_of_sequences The number of concurrent sequences to maintain on the server
    /// \param shared_memory_type The type of shared memory to use for inputs
    /// \param output_shm_size The size of the shared memory to allocate for the output
    /// \param serial_sequences Enable serial sequence mode
    /// \param parser The ModelParser object to get the model details
    /// \param factory The ClientBackendFactory object used to create client to the server
    /// \param manager Returns a new CustomRequestScheduleManager object
    /// \param request_parameters Custom request parameters to send to the server
    /// \return cb::Error object indicating success or failure
    static cb::Error Create(
      const bool async, const bool streaming,
      const uint64_t measurement_window_ms, const size_t max_trials,
      const std::vector<float>& schedule, const int32_t batch_size,
      const size_t max_threads, const uint32_t num_of_sequences,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const bool serial_sequences, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      std::unique_ptr<LoadManager>* manager,
      const std::unordered_map<std::string, cb::RequestParameter>&
        request_parameters);
  
    /// Performs warmup for benchmarking by sending a fixed number of requests
    /// according to the specified request rate
    /// \param request_rate The rate at which requests must be issued to the server
    /// \param warmup_request_count The number of warmup requests to send
    /// \return cb::Error object indicating success or failure
    cb::Error PerformWarmup(double request_rate, size_t warmup_request_count) override;

    /// Adjusts the rate of issuing requests to be the same as 'request_rate'
    /// \param request_rate The rate at which requests must be issued to the server
    /// \param request_count The number of requests to generate when profiling
    /// \return cb::Error object indicating success or failure
    cb::Error ChangeRequestRate(const double request_rate, const size_t request_count) override;

  protected:
    /// Constructor for CustomRequestScheduleManager
    /// \param async Whether to use asynchronous or synchronous API for infer request
    /// \param streaming Whether to use gRPC streaming API for infer request
    /// \param measurement_window_ms The time window for measurements
    /// \param max_trials The maximum number of windows that will be measured
    /// \param schedule The vector containing the schedule for requests
    /// \param batch_size The batch size used for each request
    /// \param max_threads The maximum number of working threads to be spawned
    /// \param num_of_sequences The number of concurrent sequences to maintain on the server
    /// \param shared_memory_type The type of shared memory to use for inputs
    /// \param output_shm_size The size of the shared memory to allocate for the output
    /// \param serial_sequences Enable serial sequence mode
    /// \param parser The ModelParser object to get the model details
    /// \param factory The ClientBackendFactory object used to create client to the server
    /// \param manager Returns a new CustomRequestScheduleManager object
    /// \param request_parameters Custom request parameters to send to the server
    CustomRequestScheduleManager(
      const bool async, const bool streaming,
      const uint64_t measurement_window_ms, const size_t max_trials,
      const std::vector<float>& schedule, const int32_t batch_size,
      const size_t max_threads, const uint32_t num_of_sequences,
      const SharedMemoryType shared_memory_type, const size_t output_shm_size,
      const bool serial_sequences, const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      const std::unordered_map<std::string, cb::RequestParameter>&
        request_parameters);

    /// Generates and updates the request schedule as per the given request rate and schedule
    /// \param request_rate The request rate to use for new schedule
    /// \param schedule The vector containing the schedule for requests
    void GenerateSchedule(const double request_rate, const std::vector<float>& schedule);

    /// Creates worker schedules based on the provided schedule
    /// \param duration The maximum duration for the schedule
    /// \param schedule The vector containing the schedule for requests
    /// \return A vector of RateSchedulePtr_t representing the worker schedules
    std::vector<RateSchedulePtr_t> CreateWorkerSchedules(
      const std::vector<float>& schedule);

    /// The vector containing the schedule for requests
    std::vector<float> schedule_;
};

}}