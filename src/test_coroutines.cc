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

#include "coroutine.h"
#include "doctest.h"

namespace triton::perfanalyzer {

Coroutine<int>
CoroutineTest()
{
  co_await std::suspend_always{};
  co_return 42;
}

TEST_CASE("testing the Coroutine class")
{
  auto coroutine = CoroutineTest();

  unsigned rounds = 0;
  while (!coroutine.Done()) {
    coroutine.Resume();
    rounds++;
  }

  auto result = coroutine.Value();

  CHECK(rounds == 2);
  CHECK(result == 42);
  CHECK(coroutine.Done());
}

Coroutine<>
CoroutineVoidTest()
{
  co_await std::suspend_always{};
}

TEST_CASE("testing the Coroutine class with void")
{
  auto coroutine = CoroutineVoidTest();

  unsigned rounds = 0;
  while (!coroutine.Done()) {
    coroutine.Resume();
    rounds++;
  }

  CHECK(rounds == 2);
  CHECK(coroutine.Done());
}

Coroutine<int>
CascadeCoroutines()
{
  co_await CoroutineVoidTest();
  auto result = co_await CoroutineTest();
  co_return result;
}

TEST_CASE("testing the Coroutine class with cascading coroutines")
{
  auto coroutine = CascadeCoroutines();

  unsigned rounds = 0;
  while (!coroutine.Done()) {
    coroutine.Resume();
    rounds++;
  }

  auto result = coroutine.Value();

  CHECK(rounds == 4);
  CHECK(result == 42);
  CHECK(coroutine.Done());
}

}  // namespace triton::perfanalyzer
