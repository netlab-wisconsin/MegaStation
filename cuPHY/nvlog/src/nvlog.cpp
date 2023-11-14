/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <signal.h>
#include <libgen.h>
#include "nvlog.h"
#include "nvlog.hpp"
#include "memtrace.h"

#define TAG "NVLOG.CPP"

static char logfile_base[1024];
static int64_t logfile_count = -1;
static size_t max_log_file_size_bytes = 20000000000;

exit_handler& pExitHandler=exit_handler::getInstance();


void update_log_filename()
{
    char logfile[1026];
    logfile_count++;
    if (logfile_count == 0)
    {
        sprintf(logfile, "%s", logfile_base);
    }
    else
    {
        sprintf(logfile, "%s.%d", logfile_base, (int)(logfile_count & 0x7));
        fmtlog::closeLogFile();
    }
    fmtlog::setLogFile(logfile, true);
}

static std::atomic_bool anyLogWasFull{false};
void logfullcb_aerial(void* unused)
{
   anyLogWasFull.store(true);
}

void logcb_aerial(int64_t ns, fmtlog::LogLevel level, fmt::string_view location, size_t basePos, fmt::string_view threadName,
           fmt::string_view msg, size_t bodyPos, size_t logFilePos) {
    if (level >= fmtlog::WRN)
    {
        fmt::print("{}\n", msg);
        fflush(stdout);
    }

    if (logFilePos > max_log_file_size_bytes)
    {
        update_log_filename();
    }
}

static std::atomic_bool threadRunning{false};
static unsigned int usec_poll_period = 100000;

void *bg_fmtlog_collector(void *)
{
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    sigaddset(&mask, SIGUSR1);
    pthread_sigmask(SIG_BLOCK, &mask, NULL);
    if(pthread_setname_np(pthread_self(), "bg_fmtlog") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }
    while (threadRunning.load() == true)
    {
        fmtlog::poll(false);
        usleep(usec_poll_period);
    }
    fmtlog::poll(true);
    printf("Exiting bg_fmtlog_collector - log queue ever was full: %d\n", anyLogWasFull.load());
    return NULL;
}

pthread_t startNVPollingThread()
{
    pthread_t thread_id;
    threadRunning.store(true);
    int ret = pthread_create(&thread_id, NULL, bg_fmtlog_collector, NULL);
    if(ret != 0)
    {
        threadRunning.store(false);
        printf("fmtlog background thread creation failed\n");
        return -1;
    }
    return thread_id;
}

void nvlog_fmtlog_close(pthread_t bg_thread_id)
{
    if (threadRunning.load() == false) return;
    threadRunning.store(false);
    pthread_join(bg_thread_id, NULL);
    fmtlog::closeLogFile();
}

void nvlog_fmtlog_thread_init()
{
    fmtlog::preallocate();
}

pthread_t nvlog_fmtlog_init(const char* yaml_file, const char* name,void (*exit_hdlr_cb)())
{
    char* env;
    pthread_t thread_id = -1;
    if(env = std::getenv("AERIAL_LOG_PATH"))
    {
        printf("AERIAL_LOG_PATH set to %s\n", env);
        strcpy(logfile_base, env);
    }
    else
    {
        printf("AERIAL_LOG_PATH unset\n"
               "Using default log path\n");
        strcpy(logfile_base, "/tmp");
    }

    if(exit_hdlr_cb)
    {
        pExitHandler.set_exit_handler_cb(exit_hdlr_cb);
    }

    strcat(logfile_base, "/");
    strncat(logfile_base, name, 64);
    printf("Log file set to %s\n", logfile_base);
    update_log_filename();
    fmtlog::setHeaderPattern("{HMSf} {l} {t} {O} ");
    fmtlog::setLogCB(logcb_aerial, fmtlog::DBG);
    fmtlog::setLogLevel(fmtlog::DBG);
    fmtlog::setLogQFullCB(logfullcb_aerial,NULL);
    thread_id = startNVPollingThread();
    if(thread_id == -1)
    {
        return -1;
    }

    for(size_t n = 0; n < NVLOG_FMTLOG_NUM_TAGS; n++)
    {
        g_nvlog_component_levels[n] = fmtlog::WRN;
    }

    if (yaml_file == NULL)
    {
        NVLOGC_FMT(TAG, "No nvlog config yaml found, using default nvlog configuration");
        // FIXME temporary set FHGEN to log info
        int i = 0;
        for(auto id : g_nvlog_component_ids)
        {
            if(strcmp (id.name, "FHGEN") == 0)
            {
                g_nvlog_component_levels[i] = fmtlog::INF;
            }
            ++i;
        }
    }
    else
    {
        NVLOGC_FMT(TAG, "Using {} for nvlog configuration", yaml_file);
        yaml::file_parser fp(yaml_file);
        yaml::document    doc       = fp.next_document();
        yaml::node        root_node = doc.root();
        // size_t num_tags = sizeof(g_nvlog_component_ids) / sizeof(nvlog_component_ids);

        yaml::node nvlog_node = root_node["nvlog"];

        int shm_log_level     = nvlog_node["shm_log_level"].as<int>();            // Log level of printing to SHM cache and disk file
        max_log_file_size_bytes = nvlog_node["max_file_size_bytes"].as<size_t>();      // maximum size of each file
        switch (shm_log_level)
        {
            case NVLOG_NONE:
            {
                for(size_t n = 0; n < NVLOG_FMTLOG_NUM_TAGS; n++)
                {
                    g_nvlog_component_levels[n] = fmtlog::OFF;
                }
                break;
            }
            case NVLOG_FATAL:
            case NVLOG_ERROR:
            {
                for(size_t n = 0; n < NVLOG_FMTLOG_NUM_TAGS; n++)
                {
                    g_nvlog_component_levels[n] = fmtlog::ERR;
                }
                break;
            }
            case NVLOG_CONSOLE:
            case NVLOG_WARN:
            {
                for(size_t n = 0; n < NVLOG_FMTLOG_NUM_TAGS; n++)
                {
                    g_nvlog_component_levels[n] = fmtlog::WRN;
                }
                break;
            }
            case NVLOG_INFO:
            {
                for(size_t n = 0; n < NVLOG_FMTLOG_NUM_TAGS; n++)
                {
                    g_nvlog_component_levels[n] = fmtlog::INF;
                }
                break;
            }
            case NVLOG_DEBUG:
            case NVLOG_VERBOSE:
            {
                for(size_t n = 0; n < NVLOG_FMTLOG_NUM_TAGS; n++)
                {
                    g_nvlog_component_levels[n] = fmtlog::DBG;
                }
                break;
            }
            default:
            {
                printf("NVLOG has invalid global log level %d, setting to WRN level\n", shm_log_level);
                for(size_t n = 0; n < NVLOG_FMTLOG_NUM_TAGS; n++)
                {
                    g_nvlog_component_levels[n] = fmtlog::WRN;
                }
                break;
            }
        }

        if(nvlog_node.has_key("nvlog_tags"))
        {
            yaml::node all_tags = nvlog_node["nvlog_tags"];

            for(size_t n = 0; n < all_tags.length(); n++)
            {
                yaml::node  tag_node = all_tags[n];
                std::string key      = tag_node.key(0);
                int         itag     = atoi(key.c_str());
                if(itag >= 0 && itag < NVLOG_DEFAULT_TAG_NUM)
                {
                    std::string tag_name = tag_node[key.c_str()].as<std::string>();
                    // nvlog_safe_strncpy(tag.tag_name, tag_name.c_str(), cfg->max_tag_len);
                    bool found = false;
                    for(int i = 0; i < NVLOG_FMTLOG_NUM_TAGS; ++i)

                    // (auto &c : g_nvlog_component_ids)
                    {
                        auto &c = g_nvlog_component_ids[i];
                        if (itag == c.id)
                        {
                            found = true;
                            if(tag_node.has_key("shm_level"))
                            {
                                switch (tag_node["shm_level"].as<int>())
                                {
                                    case NVLOG_NONE:
                                    {
                                        g_nvlog_component_levels[i] = fmtlog::OFF;
                                        break;
                                    }
                                    case NVLOG_FATAL:
                                    {
                                        g_nvlog_component_levels[i] = fmtlog::ERR;
                                        break;
                                    }
                                    case NVLOG_ERROR:
                                    {
                                        g_nvlog_component_levels[i] = fmtlog::ERR;
                                        break;
                                    }
                                    case NVLOG_CONSOLE:
                                    {
                                        g_nvlog_component_levels[i] = fmtlog::WRN;
                                        break;
                                    }
                                    case NVLOG_WARN:
                                    {
                                        g_nvlog_component_levels[i] = fmtlog::WRN;
                                        break;
                                    }
                                    case NVLOG_INFO:
                                    {
                                        g_nvlog_component_levels[i] = fmtlog::INF;
                                        break;
                                    }
                                    case NVLOG_DEBUG:
                                    {
                                        g_nvlog_component_levels[i] = fmtlog::DBG;
                                        break;
                                    }
                                    case NVLOG_VERBOSE:
                                    {
                                        g_nvlog_component_levels[i] = fmtlog::DBG;
                                        break;
                                    }
                                    default:
                                    {
                                        printf("NVLOG tag %s has invalid log level %d, setting to WRN level\n", tag_name.c_str(), tag_node["shm_level"].as<int>());
                                        g_nvlog_component_levels[i] = fmtlog::WRN;
                                        break;
                                    }
                                }
                            }
                            // printf("NVLOG tag %s level set to %d\n", tag_name.c_str(), g_nvlog_component_levels[i]);
                        }
                    }

                    if(!found)
                    {
                        printf("NVLOG tag %s do not match the ones specified in nvlog_fmt.hpp, we currently do not support dynamic tag names, skipping\n", tag_name.c_str());
                        continue;
                    }

                }
            }
        }
    }
    return thread_id;
}

static inline fmtlog::LogLevel getfmtLogLevel(int level)
{
    fmtlog::LogLevel fmt_log_level = fmtlog::OFF;
    switch (level)
    {
        case NVLOG_NONE:
        {
            fmt_log_level = fmtlog::OFF;
            break;
        }
        case NVLOG_FATAL:
        case NVLOG_ERROR:
        {
            fmt_log_level = fmtlog::ERR;
            break;
        }
        case NVLOG_CONSOLE:
        case NVLOG_WARN:
        {
            fmt_log_level = fmtlog::WRN;
            break;
        }
        case NVLOG_INFO:
        {
            fmt_log_level = fmtlog::INF;
            break;
        }
        case NVLOG_DEBUG:
        case NVLOG_VERBOSE:
        {
            fmt_log_level = fmtlog::DBG;
            break;
        }
        default:
        {
            printf("invalid log level %d, setting to WRN level\n", level);
            fmt_log_level = fmtlog::WRN;
            break;
        }
    }
    return fmt_log_level;
}

extern "C" int fmt_log_level_validate(int level, int itag, const char** stag)
{
    int retVal = 0;
    fmtlog::LogLevel reqested_fmtLogLevel = getfmtLogLevel(level);
    for(int i = 0; i < NVLOG_FMTLOG_NUM_TAGS; ++i)
    {
        auto &c = g_nvlog_component_ids[i];
        if (itag == c.id)
        {
            *stag = c.name;
            if(reqested_fmtLogLevel >= g_nvlog_component_levels[i])
            {
                retVal = 1;
            }
            break;
        }
    }
    return retVal;
}

#define MAX_C_FORMATTED_STR_SIZE 1024

extern "C" void nvlog_vprint_fmt(int level, const char* stag, const char* format, va_list va)
{
    char buffer[MAX_C_FORMATTED_STR_SIZE];
    vsnprintf(buffer, MAX_C_FORMATTED_STR_SIZE, format, va);
    fmtlog::LogLevel fmt_log_level = getfmtLogLevel(level);
    MemtraceDisableScope md; // disable memtrace while this variable is in scope
    FMTLOG_ONCE_PRINTF(fmt_log_level, "[%s] %s", stag, buffer);
}

extern "C" void nvlog_e_vprint_fmt(int level, const char* event, const char* stag, const char* format, va_list va)
{
    char buffer[MAX_C_FORMATTED_STR_SIZE];
    vsnprintf(buffer, MAX_C_FORMATTED_STR_SIZE, format, va);
    fmtlog::LogLevel fmt_log_level = getfmtLogLevel(level);
    MemtraceDisableScope md; // disable memtrace while this variable is in scope
    FMTLOG_ONCE_PRINTF(fmt_log_level, "[%s] [%s] %s", stag, event, buffer);
}

void logcb_cunit(int64_t ns, fmtlog::LogLevel level, fmt::string_view location, size_t basePos, fmt::string_view threadName,
           fmt::string_view msg, size_t bodyPos, size_t logFilePos) {
    if (level >= fmtlog::WRN)
    {
        fmt::print("{}\n", msg);
        fflush(stdout);
    }
}

extern "C" void nvlog_c_init(const char *file)
{
    fmtlog::setLogFile(file, true);
    fmtlog::setHeaderPattern("{HMSf} {l} {t} {O} ");
    fmtlog::setLogCB(logcb_cunit, fmtlog::INF);
    fmtlog::setLogLevel(fmtlog::INF);
    fmtlog::startPollingThread(1000L * 1000 * 100);
}

extern "C" void nvlog_c_close()
{
    fmtlog::stopPollingThread();
    fmtlog::closeLogFile();
}

#define MAX_PATH_LEN 1024
#define CONFIG_CUBB_ROOT_ENV "CUBB_HOME"

int get_root_path(char* path, int cubb_root_path_relative_num) {
    int length = -1;

    // If CUBB_HOME was set in system environment variables, return it
    char* env = getenv(CONFIG_CUBB_ROOT_ENV);
    if (env != NULL) {
        length = snprintf(path, MAX_PATH_LEN - 1, "%s", env);
        if (path[length - 1] != '/') {
            path[length] = '/';
            path[++length] = '\0';
        }
        return length;
    }

    // Get current process directory, and go up to
    char buf[MAX_PATH_LEN];
    size_t size = readlink("/proc/self/exe", buf, MAX_PATH_LEN - 1);
    if (size > 0 && size < MAX_PATH_LEN) {
        buf[size] = '\0';
        char* tmp = dirname(buf);
        for (int i = 0; i < cubb_root_path_relative_num; i++) {
            tmp = dirname(tmp);
        }
        length = snprintf(path, MAX_PATH_LEN - 1, "%s/", tmp);
    }
    return length;
}

int get_full_path_file(char* dest_buf, const char* relative_path, const char* file_name, int cubb_root_dir_relative_num)
{
    int length = get_root_path(dest_buf, cubb_root_dir_relative_num);

    if(relative_path != NULL)
    {
        length += snprintf(dest_buf + length, MAX_PATH_LEN - length, "%s", relative_path);
        if(dest_buf[length - 1] != '/')
        {
            dest_buf[length]   = '/';
            dest_buf[++length] = '\0';
        }
    }

    if(file_name != NULL)
    {
        length += snprintf(dest_buf + length, MAX_PATH_LEN - length, "%s", file_name);
    }
    NVLOGV_FMT(TAG, "{}: length={} full_path={}", __func__, length, dest_buf);
    return length;
}

#if 0
} // namespace nv
#endif
