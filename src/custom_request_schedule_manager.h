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

#include "command_line_parser.h"
#include "load_manager.h"
#include "request_rate_manager.h"

namespace triton::perfanalyzer {

//==============================================================================
/// CustomRequestScheduleManager is a helper class to send inference requests to
/// inference server in accordance with the schedule set by the user.
///
/// Detail:
/// An instance of this load manager will be created at the beginning of the
/// perf analyzer and it will be used to schedule to send requests at that
/// particular second defined by the user. The particular seconds at which a
/// request should be sent can be set by the user using the `schedule` option.
/// For example, if the `schedule` is set to 1,2,4,5,6.5,
/// CustomRequestScheduleManager sends request at 1st second, 2nd second, 4th
/// second and so on.
///

class CustomRequestScheduleManager : public RequestRateManager {
 public:
  ~CustomRequestScheduleManager() = default;

  /// Creates an object of CustomRequestScheduleManager
  /// \param params A PAParamsPtr (std::shared_ptr<PerfAnalyzerParameters>) that
  /// holds configuration parameters to create CustomRequestScheduleManager
  /// object
  ///
  static cb::Error Create(
      const PerfAnalyzerParameters& params,
      const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory,
      std::unique_ptr<LoadManager>* manager);

  /// Performs warmup for benchmarking by sending a fixed number of requests
  /// according to the specified request rate
  /// \param request_rate The rate at which requests must be issued to the
  /// server \param warmup_request_count The number of warmup requests to send
  /// \return cb::Error object indicating success or failure
  cb::Error PerformWarmup(
      double request_rate, size_t warmup_request_count) override;

  /// Adjusts the rate of issuing requests to be the same as 'request_rate'
  /// \param request_rate The rate at which requests must be issued to the
  /// server \param request_count The number of requests to generate when
  /// profiling \return cb::Error object indicating success or failure
  cb::Error ChangeRequestRate(
      const double request_rate, const size_t request_count) override;


 protected:
  /// Constructor for CustomRequestScheduleManager
  ///
  /// Initializes a CustomRequestScheduleManager instance using a PAParamsPtr
  /// object that contains all necessary parameters for request scheduling.
  ///
  /// \param params A PAParamsPtr (std::shared_ptr<PerfAnalyzerParameters>) that
  /// holds configuration parameters to create CustomRequestScheduleManager
  /// object
  ///
  CustomRequestScheduleManager(
      const PerfAnalyzerParameters& params,
      const std::shared_ptr<ModelParser>& parser,
      const std::shared_ptr<cb::ClientBackendFactory>& factory);

  /// Generates and updates the request schedule as per the given request rate
  /// and schedule \param request_rate The request rate to use for new schedule
  /// \param schedule The vector containing the schedule for requests
  void GenerateSchedule(const double request_rate);

  /// Creates worker schedules based on the provided schedule
  /// \param duration The maximum duration for the schedule
  /// \param schedule The vector containing the schedule for requests
  /// \return A vector of RateSchedulePtr_t representing the worker schedules
  std::vector<RateSchedulePtr_t> CreateWorkerSchedules(
      const std::vector<float>& schedule);

  /// The vector containing the schedule for requests
  std::vector<float> schedule_;
};

}  // namespace triton::perfanalyzer
