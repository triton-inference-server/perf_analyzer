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

#include "custom_request_schedule_manager.h"

namespace triton { namespace perfanalyzer {
  
cb::Error CustomRequestScheduleManager::Create(
  const bool async, const bool streaming,
  const uint64_t measurement_window_ms, const size_t max_trials,
  const std::vector<float>& schedule, const int32_t batch_size,
  const size_t max_threads, const uint32_t num_of_sequences,
  const SharedMemoryType shared_memory_type, const size_t output_shm_size,
  const bool serial_sequences, const std::shared_ptr<ModelParser>& parser,
  const std::shared_ptr<cb::ClientBackendFactory>& factory,
  std::unique_ptr<LoadManager>* manager,
  const std::unordered_map<std::string, cb::RequestParameter>& request_parameters) 
{
  std::unique_ptr<CustomRequestScheduleManager> local_manager(new CustomRequestScheduleManager(
    async, streaming, measurement_window_ms, max_trials, schedule, batch_size,
    max_threads, num_of_sequences, shared_memory_type,
    output_shm_size, serial_sequences, parser, factory, request_parameters));

  *manager = std::move(local_manager);

  return cb::Error::Success;
}

CustomRequestScheduleManager::CustomRequestScheduleManager(
  const bool async, const bool streaming,
  const uint64_t measurement_window_ms, const size_t max_trials,
  const std::vector<float>& schedule, const int32_t batch_size,
  const size_t max_threads, const uint32_t num_of_sequences,
  const SharedMemoryType shared_memory_type, const size_t output_shm_size,
  const bool serial_sequences, const std::shared_ptr<ModelParser>& parser,
  const std::shared_ptr<cb::ClientBackendFactory>& factory,
  const std::unordered_map<std::string, cb::RequestParameter>& request_parameters)
  : RequestRateManager(
      async, streaming, Distribution::CUSTOM, batch_size,
      measurement_window_ms, max_trials, max_threads, num_of_sequences,
      shared_memory_type, output_shm_size, serial_sequences, parser,
      factory, request_parameters), schedule_(schedule)
{ 
}

cb::Error
CustomRequestScheduleManager::PerformWarmup(
    double request_rate, size_t warmup_request_count)
{
  if (warmup_request_count == 0) {
    return cb::Error::Success;
  }
  RETURN_IF_ERROR(ChangeRequestRate(request_rate, warmup_request_count));
  WaitForWarmupAndCleanup();
  return cb::Error::Success;
}

cb::Error
CustomRequestScheduleManager::ChangeRequestRate(
    const double request_rate, const size_t request_count)
{
  PauseWorkers();
  ConfigureThreads(request_count);
  GenerateSchedule(request_rate, schedule_);
  ResumeWorkers();

  return cb::Error::Success;
}

void CustomRequestScheduleManager::GenerateSchedule(const double request_rate, const std::vector<float>& schedule)
{
  std::vector<float> scaled_schedule;
  scaled_schedule.reserve(schedule.size());
  if (schedule.size() > 0) {
    for (const auto& value : schedule) {
      scaled_schedule.push_back(value / static_cast<float>(request_rate));
    }
  }
  auto worker_schedules =
      CreateWorkerSchedules(schedule);
  GiveSchedulesToWorkers(worker_schedules);
}

std::vector<RateSchedulePtr_t>
CustomRequestScheduleManager::CreateWorkerSchedules(
    const std::vector<float>& schedule)
{
  std::vector<RateSchedulePtr_t> worker_schedules =
    CreateEmptyWorkerSchedules();
  std::vector<size_t> thread_ids{CalculateThreadIds()};
  std::chrono::nanoseconds next_timestamp(0);
  size_t thread_id_index = 0;
  size_t worker_index = 0;

  for (const float& val : schedule) {
    next_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<float>(val));
    worker_index = thread_ids[thread_id_index];
    thread_id_index = ++thread_id_index % thread_ids.size();
    worker_schedules[worker_index]->intervals.emplace_back(next_timestamp);
  }
  SetScheduleDurations(worker_schedules);

  return worker_schedules;
}

}}
