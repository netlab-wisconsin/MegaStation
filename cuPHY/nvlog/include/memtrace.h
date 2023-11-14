/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#pragma once
#ifndef MEMTRACE_H
#define MEMTRACE_H
#include <cstdlib>

#define MI_MEMTRACE_CONFIG_ENABLE (1)
#define MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE (2)

#if defined(__cplusplus)
#define aerial_decl_externc extern "C"
#else
#define aerial_decl_externc
#endif

aerial_decl_externc int mi_memtrace_get_config(void) __attribute__((weak));;
aerial_decl_externc void mi_memtrace_set_config(int config) __attribute__((weak));;

inline int memtrace_get_config(void)
{
   if (mi_memtrace_get_config)
   {
      return mi_memtrace_get_config();
   }
   return 0;
}

inline void memtrace_set_config(int value)
{
   if (mi_memtrace_set_config)
   {
      static const char* env = std::getenv("AERIAL_MEMTRACE");
      if (env != nullptr)
      {
         mi_memtrace_set_config(value);
      }
   }
}

class MemtraceDisableScope
{
public:
   MemtraceDisableScope()
   {
      prev_config = memtrace_get_config();
      memtrace_set_config(0);
   };

   ~MemtraceDisableScope()
   {
      memtrace_set_config(prev_config);
   };

private:
   int prev_config;
};

#endif // MEMTRACE_H
