/**
 * @file colors.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Color Terminal Output
 * @version 0.1
 * @date 2024-04-14
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

// Color definitions
#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"

// Reset color
#define RST "\x1B[0m"

// Bold color
#define BLD "\x1B[1m"
// Underline color
#define UND "\x1B[4m"
// Italic color
#define ITA "\x1B[3m"

#if defined(__DEBUG__)
#define DEBUG_INFO(...) spdlog::info(__VA_ARGS__)
#define DEBUG_WARN(...) spdlog::warn(__VA_ARGS__)
#else
#define DEBUG_INFO(...) ((void)0)
#define DEBUG_WARN(...) ((void)0)
#endif
