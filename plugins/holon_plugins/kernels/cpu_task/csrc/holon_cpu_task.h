// The task ABI of `cpu_task@1` (docs/dev/v1-offload.md D6): what a CPU kernel plugin hands the worker so that the
// worker's threads run it without Python. A task runs on every thread of the worker's pool at once; the threads
// meet at `barrier`. It returns 0, or an error code: the worker then answers the request with an error (fail-open,
// it never stops answering). A plugin that provides tasks declares this struct itself, with the same version.
#pragma once
#include <stdint.h>

#define HOLON_CPU_TASK_ABI 1

typedef struct holon_cpu_env {
    int thread;                                          // this thread, 0 … threads − 1
    int threads;
    void (*barrier)(const struct holon_cpu_env * env);   // every thread of the run meets here
    void * pool;                                         // the worker's, for `barrier`
} holon_cpu_env;

typedef int (*holon_cpu_task_fn)(void * arg, const holon_cpu_env * env);
