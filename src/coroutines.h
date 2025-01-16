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
#pragma once

#include <coroutine>
#include <type_traits>
#include <utility>

namespace triton::perfanalyzer {

template <typename T = void>
class Coroutine {
 public:
  struct Empty {};
  typedef
      typename std::conditional<std::is_void<T>::value, Empty, T>::type SafeT;

  Coroutine() = default;
  Coroutine(Coroutine&& other) = default;
  Coroutine& operator=(Coroutine&& other) = default;
  Coroutine(Coroutine const&) = delete;
  Coroutine& operator=(Coroutine const&) = delete;

  class Awaiter {
   public:
    Awaiter(Awaiter&& other) = default;
    Awaiter& operator=(Awaiter&& other) = default;
    Awaiter(Awaiter const&) = default;
    Awaiter& operator=(Awaiter const&) = default;
    constexpr bool await_ready() const noexcept
    {
      bool ret = coroutine_->earlyResume_;
      coroutine_->earlyResume_ = false;
      return ret;
    }
    constexpr void await_suspend(std::coroutine_handle<> h)
    {
      coroutine_->suspended_ = true;
    }
    constexpr void await_resume() const noexcept {}

   private:
    Awaiter(Coroutine* coroutine) : coroutine_(coroutine) {}
    Coroutine* coroutine_;
    friend struct Coroutine;
  };

  Awaiter awaiter() { return Awaiter(this); }

  void resume()
  {
    if (!handle_)
      return;
    if (!suspended_) {
      earlyResume_ = true;
      return;
    }
    suspended_ = false;
    handle_.resume();
  }

  bool done()
  {
    if (!handle_)
      return true;
    bool isDone = handle_.done();
    if (isDone) {
      if constexpr (!std::is_void<T>::value) {
        value_ = std::move(handle_.promise().value_);
      }
      handle_.destroy();
      handle_ = nullptr;
    }
    return isDone;
  }

  const SafeT& value() const { return value_; }

 private:
  struct PromiseVoid {
    Coroutine<> get_return_object()
    {
      return Coroutine<>{
          std::move(std::coroutine_handle<Promise>::from_promise(*this))};
    }
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception() {}
    void return_void()
    {
      auto awaitingCoroutine = awaitingCoroutine_;
      if (awaitingCoroutine) {
        __builtin_coro_resume(awaitingCoroutine);
      }
    }
    [[no_unique_address]] Empty value_;
    void* awaitingCoroutine_ = nullptr;
  };
  struct PromiseValue {
    Coroutine<T> get_return_object()
    {
      return Coroutine{
          std::move(std::coroutine_handle<Promise>::from_promise(*this))};
    }
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception() {}
    void return_value(T&& value)
    {
      value_ = std::move(value);
      auto awaitingCoroutine = awaitingCoroutine_;
      if (awaitingCoroutine) {
        __builtin_coro_resume(awaitingCoroutine);
      }
    }
    T value_;
    void* awaitingCoroutine_ = nullptr;
  };
  typedef typename std::conditional<
      std::is_void<T>::value, PromiseVoid, PromiseValue>::type Promise;
  Coroutine(std::coroutine_handle<Promise>&& handle)
      : handle_(std::move(handle))
  {
  }
  std::coroutine_handle<Promise> handle_;
  [[no_unique_address]] SafeT value_;
  void* awaitingCoroutine_ = nullptr;
  bool suspended_ = true;
  bool earlyResume_ = false;

 public:
  using promise_type = Promise;

  constexpr bool await_ready() { return handle_.done(); }
  template <typename U>
  constexpr void await_suspend(std::coroutine_handle<U> h)
  {
    auto& promise = handle_.promise();
    promise.awaitingCoroutine_ = h.address();
    resume();
  }
  constexpr SafeT await_resume()
  {
    SafeT value = std::move(handle_.promise().value_);
    handle_.destroy();
    return value;
  }
};

}  // namespace triton::perfanalyzer
