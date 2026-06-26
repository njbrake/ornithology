/* ornith_config.h — load an ornith_arch from a Hugging Face config.json. */
#ifndef ORNITH_CONFIG_H
#define ORNITH_CONFIG_H

#include "ornith.h"
#include "ornith_model.h"

/* Parse a config.json file at `path` into `out`. Unknown fields are ignored;
 * missing fields keep the Ornith-397B defaults so a partial config still
 * yields a usable architecture. Returns ORNITH_OK on success. */
ornith_status ornith_config_load(const char *path, ornith_arch *out);

#endif /* ORNITH_CONFIG_H */
