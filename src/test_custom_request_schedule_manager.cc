// Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <chrono>
#include <vector>

#include "command_line_parser.h"
#include "custom_request_schedule_manager.h"
#include "doctest.h"
#include "request_rate_worker.h"
#include "test_load_manager_base.h"

namespace triton::perfanalyzer {

class TestCustomRequestScheduleManager : public TestLoadManagerBase,
                                         public CustomRequestScheduleManager {
 public:
  TestCustomRequestScheduleManager(PerfAnalyzerParameters params)
      : TestLoadManagerBase(params, false, false),
        CustomRequestScheduleManager(params, GetParser(), GetFactory())
  {
  }

  void TestSchedule(const std::vector<std::chrono::milliseconds> schedule)
  {
    const size_t request_count = schedule.size();
    max_threads_ = std::min(max_threads_, request_count);
    PauseWorkers();
    ConfigureThreads(request_count);
    DistributeScheduleToWorkers(schedule);

    std::vector<std::chrono::nanoseconds> expected_timestamps{};

    for (const auto& timestamp_ms : schedule) {
      const auto timestamp{
          std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp_ms)};
      expected_timestamps.push_back(timestamp);
    }

    std::vector<std::chrono::nanoseconds> actual_timestamps{};

    for (const auto& iworker : workers_) {
      const auto worker{std::dynamic_pointer_cast<RequestRateWorker>(iworker)};
      for (size_t i{0}; i < worker->schedule_->intervals.size(); ++i) {
        const auto timestamp{worker->GetNextTimestamp()};
        actual_timestamps.push_back(timestamp);
      }
    }

    std::sort(actual_timestamps.begin(), actual_timestamps.end());

    CHECK(expected_timestamps == actual_timestamps);
  }
};

TEST_CASE("custom_request_schedule")
{
  PerfAnalyzerParameters params{};

  std::vector<std::chrono::milliseconds> schedule{};

  const auto& ParameterizeSchedule{[&]() {
    using std::literals::chrono_literals::operator""ms;

    SUBCASE("schedule A")
    {
      schedule = {1000ms, 2000ms, 3000ms, 4000ms, 5000ms};
    }
    SUBCASE("schedule B")
    {
      schedule = {500ms, 2000ms, 3500ms};
    }
    SUBCASE("schedule C")
    {
      schedule = {100ms, 300ms, 800ms, 1500ms};
    }
    SUBCASE("schedule D")
    {
      schedule = {0ms, 5000ms, 10000ms};
    }
    SUBCASE("schedule E")
    {
      schedule = {1000ms};
    }
  }};

  ParameterizeSchedule();

  params.request_count = schedule.size();

  TestCustomRequestScheduleManager tcrsm(params);

  tcrsm.TestSchedule(schedule);
}
}  // namespace triton::perfanalyzer
