// undistort_params.h -- reads the `key value` parameter file written by
// utils/undistort_params.py into the kernels' Params. Shared by every host
// wrapper, like undistort_kernels.cuh.

#pragma once

#include <clocale>
#include <cstdio>
#include <cstring>

#include "undistort_kernels.cuh"

namespace nvundistort {

// Returns false, with the reason on stderr, if the file cannot be used.
inline bool load_params(const char* path, Params* p) {
  static const char* const keys[] = {"width", "height", "fx", "fy", "cx", "cy",
                                     "new_fx", "new_fy", "new_cx", "new_cy",
                                     "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"};
  const int n = sizeof(keys) / sizeof(keys[0]);
  double val[n];
  bool seen[n] = {};

  FILE* f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "[nvundistort] cannot open params file %s\n", path);
    return false;
  }
  // gst-launch applies the system locale, and under a comma-decimal one
  // (e.g. el_GR) %lf stops at the '.': p1 = 6.98e-05 reads as 6. Parse in the
  // "C" locale; uselocale() switches only this thread.
  locale_t c_locale = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
  locale_t prev_locale = uselocale(c_locale);
  char line[256], key[64];
  double v;
  while (fgets(line, sizeof line, f)) {
    if (line[0] == '#' || sscanf(line, "%63s %lf", key, &v) != 2) continue;
    for (int i = 0; i < n; ++i)
      if (!strcmp(key, keys[i])) { val[i] = v; seen[i] = true; }
  }
  uselocale(prev_locale);
  freelocale(c_locale);
  fclose(f);
  for (int i = 0; i < n; ++i) {
    if (!seen[i]) {
      fprintf(stderr, "[nvundistort] %s: missing '%s'\n", path, keys[i]);
      return false;
    }
  }

  p->w = (int)val[0];
  p->h = (int)val[1];
  if (p->w <= 0 || p->h <= 0 || (p->w & 1) || (p->h & 1)) {
    fprintf(stderr, "[nvundistort] %s: width/height must be positive and even\n", path);
    return false;
  }
  p->fx = (float)val[2];  p->fy = (float)val[3];
  p->cx = (float)val[4];  p->cy = (float)val[5];
  p->ifx = (float)(1.0 / val[6]);  p->ify = (float)(1.0 / val[7]);
  p->ncx = (float)val[8];  p->ncy = (float)val[9];
  p->k1 = (float)val[10]; p->k2 = (float)val[11];
  p->p1 = (float)val[12]; p->p2 = (float)val[13];
  p->k3 = (float)val[14]; p->k4 = (float)val[15];
  p->k5 = (float)val[16]; p->k6 = (float)val[17];
  return true;
}

}  // namespace nvundistort
