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
#include <variant>

namespace triton::perfanalyzer {

/**
 * @brief A C++20 coroutine implementation that supports both void and
 * value-returning coroutines
 *
 * @details This class implements a coroutine that can be used to create
 * cooperative multitasking functionality. It supports both void coroutines and
 * coroutines that return a value of type T.
 *
 * The coroutine starts in a suspended state and must be explicitly resumed
 * using Resume(). The coroutine's completion status can be checked using
 * Done(), and for value-returning coroutines, the result can be retrieved using
 * Value().
 *
 * Key features:
 * - Support for void and value-returning coroutines.
 * - Manual resume control via Resume().
 * - Status checking via Done().
 * - Value retrieval via Value() for non-void coroutines.
 * - Awaitable interface for use in other coroutines, enabling cascading
 * coroutines.
 *
 * @tparam T The type of value returned by the coroutine. Use void for
 * coroutines that don't return a value.
 */
template <typename T = void>
class Coroutine {
  // The Promise class is used to manage the coroutine's state and control flow.
  // We need one to handle void coroutines and another for value-returning
  // coroutines. Their implementations are very similar, but the value-returning
  // version stores the return value. The names of the methods are based on the
  // C++20 coroutine specification, and cannot be changed. The two classes
  // eventually coalesce into the Promise type alias below. The Promise class is
  // completely internal to the Coroutine class and is not meant to be used
  // directly.
  struct PromiseVoid {
    Coroutine<> get_return_object()
    {
      return Coroutine<>{
          std::move(std::coroutine_handle<Promise>::from_promise(*this))};
    }
    std::suspend_always initial_suspend()
    {
      suspended_ = true;
      return {};
    }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception() {}
    void return_void()
    {
      auto awaitingCoroutine = awaitingCoroutine_;
      if (awaitingCoroutine) {
        awaitingCoroutine_ = nullptr;
        std::coroutine_handle<>::from_address(awaitingCoroutine).resume();
      }
    }
    [[no_unique_address]] std::monostate value_{};
    void* awaitingCoroutine_ = nullptr;
    bool earlyResume_ = false;
    bool suspended_ = false;
  };
  struct PromiseValue {
    Coroutine<T> get_return_object()
    {
      return Coroutine{
          std::move(std::coroutine_handle<Promise>::from_promise(*this))};
    }
    std::suspend_always initial_suspend()
    {
      suspended_ = true;
      return {};
    }
    std::suspend_always final_suspend() noexcept { return {}; }
    void unhandled_exception() {}
    void return_value(T&& value)
    {
      value_ = std::move(value);
      auto awaitingCoroutine = awaitingCoroutine_;
      if (awaitingCoroutine) {
        awaitingCoroutine_ = nullptr;
        std::coroutine_handle<>::from_address(awaitingCoroutine).resume();
      }
    }
    T value_{};
    void* awaitingCoroutine_ = nullptr;
    bool earlyResume_ = false;
    bool suspended_ = false;
  };
  typedef
      typename std::conditional_t<std::is_void_v<T>, PromiseVoid, PromiseValue>
          Promise;

 public:
  // The SafeT alias is used to handle the case where T is void. In this case,
  // we use std::monostate as the type of the value_ member of the Promise
  // class. This allows us to use a single Coroutine class for both void and
  // value-returning coroutines.
  typedef typename std::conditional_t<std::is_void_v<T>, std::monostate, T>
      SafeT;

  Coroutine() = default;
  Coroutine(Coroutine&& other) = default;
  Coroutine& operator=(Coroutine&& other) = default;
  // The copy constructor and copy assignment operator are deleted copying
  // coroutines doesn't make sense.
  Coroutine(Coroutine const&) = delete;
  Coroutine& operator=(Coroutine const&) = delete;

  /**
   * @brief A helper class used to implement the coroutine awaitable interface.
   *
   * @details The Awaiter class provides the necessary methods for the co_await
   * operator to work with coroutines. Its interface is based on the C++20
   * coroutine specification, and the names of the methods cannot be changed.
   *
   * This class enables proper synchronization between coroutines and allows for
   * optimization through early resume functionality to avoid unnecessary
   * suspensions.
   *
   * Its main purpose is to be a generic awaiter mechanism, in case creating a
   * specific awaiter for an asynchronous operation would be too cumbersome, but
   * in general, it is recommended to create a specific awaiter for each
   * asynchronous operation instead.
   */
  class Awaiter {
   public:
    Awaiter(Awaiter&& other) = default;
    Awaiter& operator=(Awaiter&& other) = default;
    Awaiter(Awaiter const&) = default;
    Awaiter& operator=(Awaiter const&) = default;
    constexpr bool await_ready() const noexcept { return false; }
    template <typename U>
    constexpr bool await_suspend(std::coroutine_handle<U> h)
    {
      auto& promise = h.promise();
      bool ret = promise.earlyResume_;
      promise.earlyResume_ = false;
      if (!ret) {
        promise.suspended_ = true;
      }
      return !ret;
    }
    constexpr void await_resume() const noexcept {}
  };

  /**
   * @brief Resumes the coroutine.
   *
   * @details This method resumes the coroutine if it is suspended using the
   * Awaiter mechanism above, or for its initial execution. If the coroutine is
   * not suspended, it sets a flag to resume it early when it is next suspended.
   */
  void Resume()
  {
    if (!handle_) {
      return;
    }
    auto& promise = handle_.promise();
    if (!promise.suspended_) {
      promise.earlyResume_ = true;
      return;
    }
    promise.suspended_ = false;
    handle_.resume();
  }

  /**
   * @brief Checks if the coroutine has completed.
   *
   * @details This method checks if the coroutine has completed its execution.
   * If the coroutine has completed, it will free the internal coroutine
   * resources, and the coroutine object will be in a state where it can be
   * safely destroyed. The internal Promise will be resolved, and its value will
   * become available through the Value() method.
   *
   * @return true if the coroutine has completed, false otherwise.
   */
  bool Done()
  {
    if (!handle_) {
      return true;
    }
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

  /**
   * @brief Retrieves the value returned by the coroutine.
   *
   * @return const SafeT& The value returned by the coroutine.
   */
  const SafeT& Value() const { return value_; }

 private:
  Coroutine(std::coroutine_handle<Promise>&& handle)
      : handle_(std::move(handle))
  {
  }
  std::coroutine_handle<Promise> handle_;
  [[no_unique_address]] SafeT value_;
  void* awaitingCoroutine_ = nullptr;

 public:
  // While the remainder of the class is public, the following methods are
  // the necessary boilerplate to implement the cascade coroutine mechanism.
  using promise_type = Promise;

  constexpr bool await_ready() { return handle_.done(); }
  template <typename U>
  constexpr void await_suspend(std::coroutine_handle<U> h)
  {
    auto& promise = handle_.promise();
    promise.awaitingCoroutine_ = h.address();
    Resume();
  }
  constexpr SafeT await_resume()
  {
    SafeT value = std::move(handle_.promise().value_);
    handle_.destroy();
    handle_ = nullptr;
    return value;
  }
};

}  // namespace triton::perfanalyzer
