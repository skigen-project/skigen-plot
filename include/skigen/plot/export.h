#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(SKIGENPLOT_BUILDING_LIBRARY)
        #define SKIGENPLOT_EXPORT __declspec(dllexport)
    #else
        #define SKIGENPLOT_EXPORT __declspec(dllimport)
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    #define SKIGENPLOT_EXPORT __attribute__((visibility("default")))
#else
    #define SKIGENPLOT_EXPORT
#endif
