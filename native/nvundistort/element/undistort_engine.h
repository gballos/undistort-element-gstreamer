// undistort_engine.h -- the CUDA side of the nvundistort element, behind a plain
// C++ interface. nvcc and the glib headers do not mix (nvcc defines
// __noinline__ as a macro), so the element (gstnvundistort.cpp, g++) and the
// engine (undistort_engine.cu, nvcc) are separate translation units.

#pragma once

struct NvBufSurface;

namespace nvundistort {

struct Engine;

// Loads the parameter file. Returns null, with the reason on stderr, if it
// cannot be used.
Engine* engine_new(const char* params_file);
void engine_free(Engine* e);

// Frame size the parameters are valid for.
int engine_width(const Engine* e);
int engine_height(const Engine* e);

// Undistorts `in` (I420) into `out` (NV12), both pitch-linear NVMM surfaces of
// the calibrated size. The CUDA mapping of every surface is kept and reused
// when the same buffer comes round again. Returns false on any error; CUDA
// errors are sticky, so the engine is then of no further use.
bool engine_process(Engine* e, NvBufSurface* in, NvBufSurface* out);
// Why the last engine_process() failed.
const char* engine_error(const Engine* e);
// One line describing the first input and output surface, for the log.
const char* engine_info(const Engine* e);

// Drops every kept mapping. Call it when the buffers may go away: on a caps
// change and before the pipeline stops.
void engine_forget_surfaces(Engine* e);

}  // namespace nvundistort
