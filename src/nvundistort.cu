// nvundistort.cu -- nvivafilter CUDA library: exact lens undistortion of NV12 frames.
//
// Loaded by  nvivafilter cuda-process=true customer-lib-name=<abs path>/libnvundistort.so
// For each frame, nvivafilter hands gpu_process() an EGLImage of its NV12 output
// surface. We map it into CUDA, copy the Y and UV planes to scratch buffers (a
// remap reads neighbouring pixels, so it cannot run in place), and remap them
// back into the surface, reading the scratch copies through textures.
//
// The source coordinate of every output pixel is computed in-kernel with the
// same model as OpenCV's initUndistortRectifyMap (rational radial + tangential),
// so no lookup table is stored and the result matches cv2 up to float rounding.
// The maths and the kernels live in undistort_kernels.cuh; this file is the
// nvivafilter wrapper around them.
//
// Environment (read once, in init()):
//   NVUNDISTORT_PARAMS  parameter file written by tools/undistort_params.py (required)
//   NVUNDISTORT_MODE    off | map | y | full (default full). Bring-up aids:
//                         off  -- return immediately (measures nvivafilter alone)
//                         map  -- map/unmap the surface only (measures EGL interop)
//                         y    -- remap luma only (geometry check; colour stays distorted)
//   NVUNDISTORT_STATS   N > 0: print the mean gpu_process time every N frames
//
// Any CUDA error disables undistortion for the rest of the run (CUDA errors are
// sticky) and is reported once on stderr; frames then pass through unmodified.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda_egl_interop.h>
#include <cuda_runtime.h>

// init/deinit are resolved by nvivafilter via dlsym, so they need C linkage
// whether or not NVIDIA's header declares them that way.
extern "C" {
#include "customer_functions.h"
}

#include "undistort_kernels.cuh"
#include "undistort_params.h"

namespace {

using nvundistort::load_params;
using nvundistort::make_plane_texture;
using nvundistort::Params;
using nvundistort::remap_uv;
using nvundistort::remap_y;

enum Mode { MODE_OFF, MODE_MAP, MODE_Y, MODE_FULL };
const char* const kModeNames[] = {"off", "map", "y", "full"};

Params g_p;
Mode g_mode = MODE_FULL;
bool g_ok = false;             // false => frames pass through untouched
// Scratch copies of the input planes, allocated once, and the textures the
// kernels read them through. Separate pitched allocations keep each plane
// texture-aligned at any resolution.
uint8_t* g_scratch_y = nullptr;   // w x h
uint8_t* g_scratch_uv = nullptr;  // w bytes (w/2 U,V pairs) x h/2
size_t g_scratch_y_pitch = 0, g_scratch_uv_pitch = 0;
cudaTextureObject_t g_tex_y = 0, g_tex_uv = 0;
cudaStream_t g_stream = nullptr;
int g_stats_every = 0;
long g_stats_frames = 0;
double g_stats_ms = 0.0;

bool check(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return true;
  fprintf(stderr, "[nvundistort] %s failed: %s -- UNDISTORTION DISABLED, frames pass through\n",
          what, cudaGetErrorString(err));
  g_ok = false;
  return false;
}

// ── Host code ────────────────────────────────────────────────────────────────

// Row pitch of the interleaved UV plane in bytes. The Jetson driver describes
// NV12's UV plane as w/2 one-channel elements (each element a U,V byte pair)
// and gives its pitch in those elements: 4608-wide NV12 reports pitch 2304 for
// rows that are 4608 bytes. A UV row must hold w bytes, so a smaller pitch can
// only be in pairs.
size_t uv_pitch_bytes(const cudaEglFrame& fr) {
  const size_t p = fr.frame.pPitch[1].pitch;
  return p < (size_t)g_p.w ? 2 * p : p;
}

// Checks the layout the kernels rely on: pitch-linear, a w x h luma plane, and
// one interleaved chroma plane of h/2 rows at least w bytes wide. Deliberately
// not planeDesc[1].numChannels (see uv_pitch_bytes) nor UV vs VU byte order,
// since both chroma bytes get the same remap.
bool surface_ok(const cudaEglFrame& fr) {
  static bool described = false;
  const cudaEglPlaneDesc& uv = fr.planeDesc[1];
  if (!described) {
    described = true;
    fprintf(stderr, "[nvundistort] surface: frameType=%d colorFormat=%d planes=%u "
                    "Y %ux%u pitch %zu, UV %ux%u ch=%u bits=%d,%d pitch %zu -> %zu bytes\n",
            (int)fr.frameType, (int)fr.eglColorFormat, fr.planeCount,
            fr.planeDesc[0].width, fr.planeDesc[0].height, fr.frame.pPitch[0].pitch,
            uv.width, uv.height, uv.numChannels, uv.channelDesc.x, uv.channelDesc.y,
            fr.frame.pPitch[1].pitch, uv_pitch_bytes(fr));
  }
  if (fr.frameType == cudaEglFrameTypePitch && fr.planeCount == 2 &&
      (int)fr.planeDesc[0].width == g_p.w && (int)fr.planeDesc[0].height == g_p.h &&
      (int)uv.height == g_p.h / 2 && fr.frame.pPitch[0].pitch >= (size_t)g_p.w &&
      uv_pitch_bytes(fr) >= (size_t)g_p.w)
    return true;
  fprintf(stderr, "[nvundistort] unexpected surface (see line above); expected pitch-linear NV12 "
                  "%dx%d -- UNDISTORTION DISABLED, frames pass through\n", g_p.w, g_p.h);
  g_ok = false;
  return false;
}

void undistort(const cudaEglFrame& fr) {
  const int w = g_p.w, h = g_p.h;
  uint8_t* y = (uint8_t*)fr.frame.pPitch[0].ptr;
  uint8_t* uv = (uint8_t*)fr.frame.pPitch[1].ptr;
  const int y_pitch = (int)fr.frame.pPitch[0].pitch;
  const int uv_pitch = (int)uv_pitch_bytes(fr);
  const dim3 block(32, 8);

  if (!check(cudaMemcpy2DAsync(g_scratch_y, g_scratch_y_pitch, y, y_pitch, w, h,
                               cudaMemcpyDeviceToDevice, g_stream),
             "copy Y to scratch"))
    return;
  remap_y<<<dim3((w + 31) / 32, (h + 7) / 8), block, 0, g_stream>>>(g_tex_y, y, y_pitch, g_p);

  if (g_mode == MODE_FULL) {
    // The UV plane is w bytes wide: w/2 interleaved U,V pairs.
    if (!check(cudaMemcpy2DAsync(g_scratch_uv, g_scratch_uv_pitch, uv, uv_pitch, w, h / 2,
                                 cudaMemcpyDeviceToDevice, g_stream),
               "copy UV to scratch"))
      return;
    remap_uv<<<dim3((w / 2 + 31) / 32, (h / 2 + 7) / 8), block, 0, g_stream>>>(g_tex_uv, uv,
                                                                                uv_pitch, g_p);
  }
  if (check(cudaGetLastError(), "kernel launch"))
    check(cudaStreamSynchronize(g_stream), "cudaStreamSynchronize");
}

void record_time(std::chrono::steady_clock::time_point t0) {
  if (g_stats_every <= 0) return;
  g_stats_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (++g_stats_frames % g_stats_every == 0) {
    fprintf(stderr, "[nvundistort] mode=%s %.2f ms/frame (mean of %d)\n", kModeNames[g_mode],
            g_stats_ms / g_stats_every, g_stats_every);
    g_stats_ms = 0.0;
  }
}

void gpu_process(EGLImageKHR image, void** /*usrptr*/) {
  if (g_mode == MODE_OFF || !g_ok) return;
  const auto t0 = std::chrono::steady_clock::now();

  cudaGraphicsResource_t res = nullptr;
  if (!check(cudaGraphicsEGLRegisterImage(&res, image, cudaGraphicsRegisterFlagsNone),
             "cudaGraphicsEGLRegisterImage"))
    return;
  cudaEglFrame fr;
  if (check(cudaGraphicsResourceGetMappedEglFrame(&fr, res, 0, 0),
            "cudaGraphicsResourceGetMappedEglFrame") &&
      surface_ok(fr) && g_mode != MODE_MAP)
    undistort(fr);
  check(cudaGraphicsUnregisterResource(res), "cudaGraphicsUnregisterResource");
  record_time(t0);
}

// CPU hooks, unused (only cuda-process is enabled). Set anyway so nvivafilter
// never calls through a null pointer if pre/post-process get switched on.
void noop_cpu_process(void**, unsigned int*, unsigned int*, unsigned int*, unsigned int*,
                      ColorFormat*, unsigned int, void**) {}

}  // namespace

extern "C" void init(CustomerFunction* funcs) {
  funcs->fPreProcess = noop_cpu_process;
  funcs->fGPUProcess = gpu_process;
  funcs->fPostProcess = noop_cpu_process;

  const char* mode = getenv("NVUNDISTORT_MODE");
  if (mode && *mode) {
    int m = 0;
    while (m < 4 && strcmp(mode, kModeNames[m])) ++m;
    if (m == 4) {
      fprintf(stderr, "[nvundistort] unknown NVUNDISTORT_MODE '%s' (off|map|y|full) -- "
                      "UNDISTORTION DISABLED\n", mode);
      return;
    }
    g_mode = (Mode)m;
  }
  const char* stats = getenv("NVUNDISTORT_STATS");
  g_stats_every = stats ? atoi(stats) : 0;
  if (g_mode == MODE_OFF) {
    fprintf(stderr, "[nvundistort] mode=off: frames pass through\n");
    return;
  }

  const char* path = getenv("NVUNDISTORT_PARAMS");
  if (!path || !*path) {
    fprintf(stderr, "[nvundistort] NVUNDISTORT_PARAMS is not set -- UNDISTORTION DISABLED\n");
    return;
  }
  if (!load_params(path, &g_p)) {
    fprintf(stderr, "[nvundistort] UNDISTORTION DISABLED\n");
    return;
  }
  g_ok = true;  // check() clears it on failure
  const int w = g_p.w, h = g_p.h;
  if (!check(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking), "cudaStreamCreate") ||
      !check(cudaMallocPitch(&g_scratch_y, &g_scratch_y_pitch, w, h), "cudaMallocPitch Y") ||
      !check(cudaMallocPitch(&g_scratch_uv, &g_scratch_uv_pitch, w, h / 2), "cudaMallocPitch UV") ||
      !check(make_plane_texture(g_scratch_y, w, h, g_scratch_y_pitch, 1, &g_tex_y), "Y texture") ||
      !check(make_plane_texture(g_scratch_uv, w / 2, h / 2, g_scratch_uv_pitch, 2, &g_tex_uv),
             "UV texture"))
    return;
  fprintf(stderr, "[nvundistort] mode=%s %dx%d params=%s\n", kModeNames[g_mode], g_p.w, g_p.h,
          path);
}

extern "C" void deinit(void) {
  if (g_tex_y) cudaDestroyTextureObject(g_tex_y);
  if (g_tex_uv) cudaDestroyTextureObject(g_tex_uv);
  if (g_scratch_y) cudaFree(g_scratch_y);
  if (g_scratch_uv) cudaFree(g_scratch_uv);
  if (g_stream) cudaStreamDestroy(g_stream);
  g_tex_y = g_tex_uv = 0;
  g_scratch_y = g_scratch_uv = nullptr;
  g_stream = nullptr;
  g_ok = false;
}
