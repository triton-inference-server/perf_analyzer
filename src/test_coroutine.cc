// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <coroutine>
#include <queue>

#include "coroutine.h"
#include "doctest.h"

namespace triton::perfanalyzer {

// A simple coroutine that returns an integer.
// It awaits for an Awaiter object and then returns 42,
// meaning that it will resume twice before completing
// as the coroutine is created in a suspended state.
Coroutine<int>
CoroutineTest()
{
  co_await Coroutine<int>::Awaiter{};
  co_return 42;
}

TEST_CASE("coroutine:testing the Coroutine class")
{
  auto coroutine = CoroutineTest();

  REQUIRE(!coroutine.Done());  

  coroutine.Resume(); // resume from initial suspension  
  coroutine.Resume(); // resume from suspension at end, coroutine completes  

  CHECK(coroutine.Done());  
  CHECK(coroutine.Value() == 42);
}

// A simple coroutine that returns void.
// This is the same as the previous test, but with a void return type,
// in order to test the void specialization of the Coroutine class.
Coroutine<>
CoroutineVoidTest()
{
  co_await Coroutine<>::Awaiter{};
}

TEST_CASE("coroutine:testing the Coroutine class with void")
{
  auto coroutine = CoroutineVoidTest();

  unsigned rounds = 0;
  while (!coroutine.Done()) {
    coroutine.Resume();
    rounds++;
  }

  CHECK(rounds == 2);
  CHECK(coroutine.Done());
  static_assert(std::is_same_v<std::decay_t<decltype(coroutine.Value())>, std::monostate>);
}

// This tests cascading coroutines, where one coroutine awaits another.
// To do this without a global scheduler, we use a queue to store the
// pending coroutines.
std::queue<std::coroutine_handle<>> pendingCoroutines;

// The specialized awaiter will simply queue the coroutine handle to
// be resumed immediately in the pseudo-scheduler loop of the test.
struct QueuedAwaiter {
  bool await_ready() { return false; }
  void await_suspend(std::coroutine_handle<> h) { pendingCoroutines.push(h); }
  void await_resume() {}
};

// A coroutine that schedules itself to be resumed later, and returns 42.
Coroutine<int>
CascadeCoroutine()
{
  co_await QueuedAwaiter{};
  co_return 42;
}

// The main coroutine that awaits the previous one and returns its value.
Coroutine<int>
CascadeCoroutinesTest()
{
  co_return co_await CascadeCoroutine() * 2;
}

TEST_CASE("coroutine:testing the Coroutine class with cascading coroutines")
{
  auto coroutine = CascadeCoroutinesTest();
  coroutine.Resume();

  unsigned rounds = 0;
  while (!coroutine.Done()) {
    CHECK(!pendingCoroutines.empty());
    auto pending = pendingCoroutines.front();
    pendingCoroutines.pop();
    pending.resume();
    rounds++;
  }

  auto result = coroutine.Value();

  CHECK(rounds == 1);
  CHECK(result == 84);
  CHECK(coroutine.Done());
  CHECK(pendingCoroutines.empty());
}

}  // namespace triton::perfanalyzer
