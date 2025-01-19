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

#include "dlwrapper.h"

#include <dlfcn.h>

void
DLWrapper::open(const char* path, RTLD::TYPE type, int flags)
{
  int dlflags = type == RTLD::TYPE::LAZY ? RTLD_LAZY : RTLD_NOW;
  if (flags & static_cast<int>(RTLD::FLAG::GLOBAL)) {
    dlflags |= RTLD_GLOBAL;
  }
  if (flags & static_cast<int>(RTLD::FLAG::LOCAL)) {
    dlflags |= RTLD_LOCAL;
  }
  if (flags & static_cast<int>(RTLD::FLAG::NODELETE)) {
    dlflags |= RTLD_NODELETE;
  }
  if (flags & static_cast<int>(RTLD::FLAG::NOLOAD)) {
    dlflags |= RTLD_NOLOAD;
  }
  if (flags & static_cast<int>(RTLD::FLAG::DEEPBIND)) {
    dlflags |= RTLD_DEEPBIND;
  }
  handle_ = dlopen(path, dlflags);
  if (!handle_) {
    throw std::runtime_error(dlerror());
  }
}

void
DLWrapper::load()
{
  for (ImportBase* func = head_; func; func = func->next_) {
    func->resolve(this);
  }
}

DLWrapper::~DLWrapper()
{
  dlclose(handle_);
}

void*
DLWrapper::resolve(const char* name, NullAllowed nullAllowed)
{
  void* symbol = dlsym(handle_, name);
  if (!symbol && (nullAllowed == NullAllowed::No)) {
    throw std::runtime_error(dlerror());
  }
  return symbol;
}
