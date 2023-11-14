/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef _NVLOG_HPP_
#define _NVLOG_HPP_

#include <stdint.h>

#include <sstream>
#include <iostream>

#include "nvlog.h"
#include "nv_utils.h"

#include "yaml.hpp"

#include "nvlog_fmt.hpp"


// Initialize the fmtlog for nvlog
pthread_t nvlog_fmtlog_init(const char* yaml_file, const char* name,void (*exit_hdlr_cb)());
void nvlog_fmtlog_thread_init();

// close the fmtlog for nvlog
void nvlog_fmtlog_close(pthread_t bg_thread_id);

int get_root_path(char* path, int cubb_root_path_relative_num);
int get_full_path_file(char* dest_buf, const char* relative_path, const char* file_name, int cubb_root_dir_relative_num);

#endif /* _NVLOG_HPP_ */
