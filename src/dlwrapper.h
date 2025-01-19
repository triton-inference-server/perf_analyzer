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

#include <stdexcept>
#include <string>
#include <utility>

/**
 * @brief A wrapper class for dynamic library loading and symbol resolution.
 *
 * @details This class provides a C++ interface for dynamically loading shared
 * libraries and resolving symbols at runtime using dlopen/dlsym/dlclose
 * functionality. It manages the lifecycle of the loaded library and provides
 * type-safe access to library symbols.
 *
 *  The class supports:
 * - Loading shared libraries with configurable flags.
 * - Type-safe symbol resolution.
 * - Automatic library cleanup on destruction.
 * - Function pointer wrapping with type safety.
 *
 * The class is designed to be used as a base class for a library import class,
 * which declares the library symbols as public members. The symbols are
 * resolved when the library is loaded.
 *
 * @note Thread safety is not guaranteed. The user must ensure thread-safe
 * access.
 *
 * Usage example:
 * @code
 *   class SomeImport : public DLWrapper {
 *     public:
 *       SomeImport() : DLWrapper("some_library.so") { load(); }
 *
 *       Function<void(int)> MyFunction1 = {this, "MyFunction1"};
 *       Function<int(float, double)> MyFunction2 = {this, "MyFunction2"};
 *       Import<int> MyVariable = {this, "MyVariable"};
 *   };
 *
 *   SomeImport someImport;
 *   void callSomeImport1() { someImport.MyFunction1(42); }
 *   int callSomeImport2() { return someImport.MyFunction2(8.0f, 12.0); }
 * @endcode
 *
 * @see RTLD For loading flags and options
 * @see Function For type-safe function wrapper
 * @see NullAllowed For null symbol handling policy
 *
 * @throws std::runtime_error If library loading fails
 * @throws std::runtime_error If symbol resolution fails (when NullAllowed::No)
 */
class DLWrapper {
 public:
  /**
   * @brief Policy for handling null symbol resolution.
   * @details This enum is used to specify the behavior when a symbol is not
   * found in the library. If NullAllowed::No is specified, the class will throw
   * an exception when the symbol is not found. If NullAllowed::Yes is
   * specified, the class will return nullptr when the symbol is not found.
   */
  enum class NullAllowed : bool { No = false, Yes = true };

  /**
   * @brief RTLD flags and options for library loading.
   */
  struct RTLD {
    /**
     * @brief Library loading type.
     * @details This enum is used to specify the library loading type.
     * LAZY: Resolve symbols only when needed. This is useful only for
     *       libraries with circular dependencies, and can lead to difficult
     *       debugging scenarios.
     * NOW: Resolve all symbols immediately. This is the default behavior.
     */
    enum class TYPE {
      LAZY,
      NOW,
    };
    /**
     * @brief Library loading flags.
     * @details This enum is used to specify the library loading flags.
     * GLOBAL: Symbols in the library are available to other libraries.
     * LOCAL: Symbols in the library are not available to other libraries.
     * NODELETE: Do not unload the library when no longer used.
     * NOLOAD: Do not load the library. This is useful for checking the library
     *         without actually loading it, but should only be used with a
     *         temporary object, as no symbols will be resolvable.
     * DEEPBIND: Use the library's symbols instead of the global symbols.
     */
    enum class FLAG : int {
      GLOBAL = 1,
      LOCAL = 2,
      NODELETE = 4,
      NOLOAD = 8,
      DEEPBIND = 16,
    };
  };

 private:
  class ImportBase {
    ImportBase* next_ = nullptr;
    NullAllowed nullAllowed_ = NullAllowed::No;

   protected:
    ImportBase(DLWrapper* wrapper, NullAllowed nullAllowed = NullAllowed::No)
        : next_(wrapper->head_), nullAllowed_(nullAllowed)
    {
      wrapper->head_ = this;
    }
    virtual void resolve(DLWrapper* wrapper) = 0;

    friend DLWrapper;

   private:
    ImportBase(const ImportBase&) = delete;
    ImportBase& operator=(const ImportBase&) = delete;
  };

 public:
  /**
   * @brief Construct a new DLWrapper object.
   * @details This constructor opens the shared library with the specified name
   * or path. The library is loaded with the specified type and flags. See
   * `RTLD::TYPE` and `RTLD::FLAG` for more information.
   */
  template <typename... Flags>
  DLWrapper(const char* path, RTLD::TYPE type = RTLD::TYPE::NOW, Flags... flags)
  {
    static_assert(
        (std::is_same_v<Flags, RTLD::FLAG> && ...),
        "All flags must be of type RTLD::FLAG");
    int combinedFlags = 0;
    (void(combinedFlags |= static_cast<int>(flags)), ...);
    open(path, type, combinedFlags);
  }
  template <typename... Flags>
  DLWrapper(
      const std::string& path, RTLD::TYPE type = RTLD::TYPE::NOW,
      Flags... flags)
      : DLWrapper(path.c_str(), type, flags...)
  {
  }

  DLWrapper(DLWrapper&& other) noexcept
      : handle_(other.handle_), head_(other.head_)
  {
    other.handle_ = nullptr;
    other.head_ = nullptr;
  }
  DLWrapper& operator=(DLWrapper&& other) noexcept
  {
    if (this != &other) {
      handle_ = other.handle_;
      head_ = other.head_;
      other.handle_ = nullptr;
      other.head_ = nullptr;
    }
    return *this;
  }
  ~DLWrapper();

  /**
   * @brief Resolve a symbol in the library.
   * @details This function resolves a symbol in the library and returns a
   * pointer to the symbol. If the symbol is not found, the behavior is
   * determined by the nullAllowed parameter. This is only useful for a
   * temporary import. The preferred way to access symbols is through the
   * `Import` and `Function` classes below.
   */
  template <typename T>
  T* getSymbol(const char* name, NullAllowed nullAllowed = NullAllowed::No)
  {
    return reinterpret_cast<T*>(resolve(name, nullAllowed));
  }

  /**
   * @brief Declare an import of a symbol in the library.
   * @details This function declares an import of a symbol in the library. The
   * symbol is resolved when the library is loaded. The import is type-safe and
   * provides access to the symbol through the `get`, `operator*`, and
   * `operator->` methods. A cast operator is also provided for direct access to
   * the symbol.
   * @note See the example in the `DLWrapper` class description for usage.
   * @tparam T The type of the symbol.
   */
  template <typename T>
  class Import : public ImportBase {
   public:
    template <size_t N>
    Import(DLWrapper* wrapper, const char (&name)[N])
        : ImportBase(wrapper), name_(name)
    {
    }
    const T* get() const { return ptr_; }
    operator const T&() const { return *ptr_; }
    const T* operator->() const { return ptr_; }
    T* get() { return ptr_; }
    operator T&() { return *ptr_; }
    T* operator->() { return ptr_; }

   private:
    void resolve(DLWrapper* wrapper) override
    {
      ptr_ = reinterpret_cast<T*>(wrapper->resolve(name_));
    }
    T* ptr_;
    const char* const name_;
  };

  /**
   * @brief Declare a function import in the library.
   * @details This function declares an import of a function in the library. The
   * function is resolved when the library is loaded. The import is type-safe
   * and becomes a callable object.
   * @note See the example in the `DLWrapper` class description for usage.
   * @tparam Ret The return type of the function.
   * @tparam Args The argument types of the function.
   */
  template <typename>
  class Function;
  template <typename Ret, typename... Args>
  class Function<Ret(Args...)> : public ImportBase {
   public:
    template <size_t N>
    Function(DLWrapper* wrapper, const char (&name)[N])
        : ImportBase(wrapper), name_(name)
    {
    }

    Ret operator()(Args... args) const
    {
      if constexpr (!std::is_same_v<void, Ret>) {
        return ptr_(std::forward<Args>(args)...);
      } else {
        ptr_(std::forward<Args>(args)...);
      }
    }

   private:
    void resolve(DLWrapper* wrapper) override
    {
      ptr_ = reinterpret_cast<FuncPtr>(wrapper->resolve(name_));
    }
    using FuncPtr = Ret (*)(Args...);
    FuncPtr ptr_ = {nullptr};
    const char* const name_;
  };

 protected:
  /**
   * @brief Load the library.
   * @details This function is called to load the library. It resolves all
   * declared imports in the library.
   * @note This function is meant to be called by the derived class'
   * constructor.
   */
  void load();

 private:
  DLWrapper(const DLWrapper&) = delete;
  DLWrapper& operator=(const DLWrapper&) = delete;

  void open(const char* path, RTLD::TYPE type, int flags);
  void* resolve(const char* name, NullAllowed nullAllowed = NullAllowed::No);
  void* handle_;
  ImportBase* head_ = nullptr;
  friend ImportBase;
};  // class DLWrapper
