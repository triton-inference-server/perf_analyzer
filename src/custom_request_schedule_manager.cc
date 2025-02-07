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

#include "custom_request_schedule_manager.h"

namespace triton::perfanalyzer {

cb::Error
CustomRequestScheduleManager::Create(
    const PerfAnalyzerParameters& params,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory,
    std::unique_ptr<LoadManager>* manager)
{
  std::unique_ptr<CustomRequestScheduleManager> local_manager(
      new CustomRequestScheduleManager(params, parser, factory));

  *manager = std::move(local_manager);

  return cb::Error::Success;
}

CustomRequestScheduleManager::CustomRequestScheduleManager(
    const PerfAnalyzerParameters& params,
    const std::shared_ptr<ModelParser>& parser,
    const std::shared_ptr<cb::ClientBackendFactory>& factory)
    : RequestRateManager(
          params.async, params.streaming, Distribution::CUSTOM,
          params.batch_size, params.measurement_window_ms, params.max_trials,
          params.max_threads, params.num_of_sequences,
          params.shared_memory_type, params.output_shm_size,
          params.serial_sequences, parser, factory, params.request_parameters)
{
  max_threads_ = std::min(max_threads_, params.request_count);
}

cb::Error
CustomRequestScheduleManager::InitCustomSchedule(const size_t request_count)
{
  PauseWorkers();
  ConfigureThreads(request_count);
  GenerateSchedule();
  ResumeWorkers();

  return cb::Error::Success;
}

void
CustomRequestScheduleManager::GenerateSchedule()
{
  auto worker_schedules = CreateWorkerSchedules(schedule_);
  GiveSchedulesToWorkers(worker_schedules);
}

std::vector<RateSchedulePtr_t>
CustomRequestScheduleManager::CreateWorkerSchedules(
    const std::vector<std::chrono::milliseconds>& schedule)
{
  std::vector<RateSchedulePtr_t> worker_schedules =
      CreateEmptyWorkerSchedules();
  std::vector<size_t> thread_ids{CalculateThreadIds()};
  size_t thread_id_index = 0;
  size_t worker_index = 0;

  for (const auto& timestamp : schedule) {
    const auto timestamp_ns{
        std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp)};
    worker_index = thread_ids[thread_id_index];
    thread_id_index = ++thread_id_index % thread_ids.size();
    worker_schedules[worker_index]->intervals.push_back(timestamp_ns);
  }
  SetScheduleDurations(worker_schedules);

  return worker_schedules;
}

void
CustomRequestScheduleManager::InitManagerFinalize()
{
  schedule_ = GetScheduleFromDataset();
  parser_->Inputs()->erase("timestamp");
}

std::vector<std::chrono::milliseconds>
CustomRequestScheduleManager::GetScheduleFromDataset() const
{
  std::vector<std::chrono::milliseconds> schedule{};

  if (data_loader_->GetDataStreamsCount() != 1) {
    throw std::runtime_error(
        "Expected input data JSON to have one stream. Fixed schedule mode "
        "must have an input data JSON with a single flat array for the "
        "\"data\" field with one element per request payload.");
  }

  const size_t dataset_size{data_loader_->GetTotalSteps(0)};

  for (size_t dataset_index{0}; dataset_index < dataset_size; ++dataset_index) {
    const auto timestamp{GetTimestamp(dataset_index)};
    schedule.push_back(timestamp);
  }

  std::sort(schedule.begin(), schedule.end());

  return schedule;
}

std::chrono::milliseconds
CustomRequestScheduleManager::GetTimestamp(size_t dataset_index) const
{
  TensorData timestamp_tensor_data{};

  const auto error{data_loader_->GetInputData(
      (*parser_->Inputs())["timestamp"], 0, dataset_index,
      timestamp_tensor_data)};

  if (!error.IsOk()) {
    throw std::runtime_error(error.Message());
  }

  const uint64_t timestamp_ms{
      *reinterpret_cast<const uint64_t*>(timestamp_tensor_data.data_ptr)};

  return std::chrono::milliseconds(timestamp_ms);
}

}  // namespace triton::perfanalyzer
