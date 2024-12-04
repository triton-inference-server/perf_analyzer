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

#include <vector>

#include "command_line_parser.h"
#include "custom_request_schedule_manager.h"
#include "doctest.h"
#include "request_rate_worker.h"
#include "test_load_manager_base.h"

namespace triton::perfanalyzer {

/// Class to test CustomRequestScheduleManager
///
class TestCustomRequestScheduleManager : public TestLoadManagerBase,
                                         public CustomRequestScheduleManager {
 public:
  TestCustomRequestScheduleManager(
      PerfAnalyzerParameters params, bool is_sequence_model = false,
      bool is_decoupled_model = false, bool use_mock_infer = false)
      : use_mock_infer_(use_mock_infer),
        TestLoadManagerBase(params, is_sequence_model, is_decoupled_model),
        CustomRequestScheduleManager(params, GetParser(), GetFactory()),
        schedule_(params.schedule)
  {
  }

  void TestSchedule(double request_rate, PerfAnalyzerParameters params)
  {
    int request_count = schedule_.size();
    PauseWorkers();
    ConfigureThreads(request_count);
    GenerateSchedule(request_rate);

    std::vector<std::chrono::nanoseconds> expected_timestamps;
    std::chrono::nanoseconds timestamp(0);

    for (float schedule_value : schedule_) {
      float scaled_value = schedule_value / static_cast<float>(request_rate);

      timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<float>(scaled_value));

      expected_timestamps.push_back(timestamp);
    }

    size_t request_index = 0;
    for (auto worker : workers_) {
      auto timestamp = std::dynamic_pointer_cast<RequestRateWorker>(worker)
                           ->GetNextTimestamp();
      REQUIRE(timestamp.count() == expected_timestamps[request_index].count());
      request_index++;
    }
  }

 private:
  bool use_mock_infer_;
  std::vector<float> schedule_;
};

TEST_CASE("custom_request_schedule")
{
  PerfAnalyzerParameters params;
  params.max_trials = 10;
  bool is_sequence = false;
  bool is_decoupled = false;
  bool use_mock_infer = true;
  double request_rate;

  const auto& ParameterizeRequestRate{[&]() {
    SUBCASE("rate 1")
    {
      request_rate = 1;
    }
    SUBCASE("rate 10")
    {
      request_rate = 10;
    }
    SUBCASE("rate 100")
    {
      request_rate = 100;
    }
  }};

  const auto& ParameterizeSchedule{[&]() {
    SUBCASE("schedule A")
    {
      ParameterizeRequestRate();
      params.schedule = {1.0, 2.0, 3.0, 4.0, 5.0};
    }
    SUBCASE("schedule B")
    {
      ParameterizeRequestRate();
      params.schedule = {0.5, 2.0, 3.5};
    }
    SUBCASE("schedule C")
    {
      ParameterizeRequestRate();
      params.schedule = {0.1, 0.3, 0.8, 1.5};
    }
    SUBCASE("schedule D")
    {
      ParameterizeRequestRate();
      params.schedule = {1.0, 5.0, 10.0};
    }
    SUBCASE("schedule E")
    {
      ParameterizeRequestRate();
      params.schedule = {1.0};
    }
  }};

  ParameterizeSchedule();
  TestCustomRequestScheduleManager tcrsm(
      params, is_sequence, is_decoupled, use_mock_infer);

  tcrsm.TestSchedule(request_rate, params);
}
}  // namespace triton::perfanalyzer
