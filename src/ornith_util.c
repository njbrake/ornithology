/* ornith_util.c — error strings and the thread-unsafe last-error buffer.
 * Kept deliberately tiny; this engine is single-process by design. */
#include "ornith.h"
#include <stdarg.h>
#include <string.h>

static char g_err[512];

const char *ornith_strerror(ornith_status s) {
    switch (s) {
    case ORNITH_OK:            return "ok";
    case ORNITH_ERR_IO:        return "I/O error";
    case ORNITH_ERR_FORMAT:    return "format error";
    case ORNITH_ERR_UNSUPPORTED:return "unsupported";
    case ORNITH_ERR_OOM:       return "out of memory";
    case ORNITH_ERR_NOTFOUND:  return "not found";
    }
    return "unknown error";
}

const char *ornith_last_error(void) {
    return g_err[0] ? g_err : "(no detail)";
}

void ornith_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
}
