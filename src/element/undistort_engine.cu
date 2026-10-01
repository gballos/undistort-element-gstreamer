// undistort_engine.cu -- the CUDA side of the nvundistort element.
//
// Reads nvv4l2decoder's I420 surface through textures and writes the
// undistorted frame, as NV12, straight into the element's output surface. No
// copy of the input is needed: source and destination are different buffers.
//
// A surface reaches CUDA through an EGLImage (NvBufSurfaceMapEglImage +
// cudaGraphicsEGLRegisterImage). That costs about 3 ms per frame, so every
// mapping is kept, keyed by the buffer's dmabuf fd: the decoder and the output
// pool each cycle through a handful of buffers.

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda_egl_interop.h>
#include <cuda_runtime.h>

#include "nvbufsurface.h"

#include "../undistort_kernels.cuh"
#include "../undistort_params.h"
#include "undistort_engine.h"

namespace nvundistort {

namespace {

// More mappings than this means buffers are not being recycled; start over
// rather than grow without bound.
const size_t kMaxMappings = 16;

struct Mapping {
  const NvBufSurface* origin = nullptr;  // the buffer's own NvBufSurface
  // Our copy of the surface description. The EGLImage is created from, and
  // stored in, this copy, so unmapping never touches a buffer that may have
  // been freed meanwhile.
  NvBufSurface surf;
  NvBufSurfaceParams params;
  bool egl = false;
  cudaGraphicsResource_t res = nullptr;
  cudaEglFrame fr;
  cudaTextureObject_t tex[3] = {0, 0, 0};
};

typedef std::map<uint64_t, Mapping> Cache;  // by dmabuf fd

}  // namespace

struct Engine {
  Params p;
  cudaStream_t stream = nullptr;
  Cache in, out;
  std::string error, info;
};

namespace {

bool fail(Engine* e, const char* what, cudaError_t err = cudaSuccess) {
  e->error = what;
  if (err != cudaSuccess) e->error += std::string(": ") + cudaGetErrorString(err);
  return false;
}

// Row pitch of plane `i` in bytes. The Jetson driver gives the pitch of NV12's
// UV plane in U,V pairs (2304 for 4608-byte rows); a row holds `row_bytes`, so
// a smaller pitch can only be in pairs.
size_t pitch_bytes(const cudaEglFrame& fr, int i, int row_bytes) {
  const size_t p = fr.frame.pPitch[i].pitch;
  return p < (size_t)row_bytes ? 2 * p : p;
}

void describe(Engine* e, const char* tag, const Mapping& m) {
  char buf[256];
  snprintf(buf, sizeof buf, "%s%s: colorFormat=%d layout=%d frameType=%d planes=%u %ux%u pitch %zu",
           e->info.empty() ? "" : "; ", tag, (int)m.params.colorFormat, (int)m.params.layout,
           (int)m.fr.frameType, m.fr.planeCount, m.fr.planeDesc[0].width, m.fr.planeDesc[0].height,
           m.fr.frame.pPitch[0].pitch);
  e->info += buf;
}

void unmap(Mapping* m) {
  for (int i = 0; i < 3; ++i)
    if (m->tex[i]) cudaDestroyTextureObject(m->tex[i]);
  if (m->res) cudaGraphicsUnregisterResource(m->res);
  if (m->egl) NvBufSurfaceUnMapEglImage(&m->surf, 0);
  *m = Mapping();
}

void forget(Cache* cache) {
  for (auto& kv : *cache) unmap(&kv.second);
  cache->clear();
}

// Maps `s` into CUDA and checks the layout the kernels rely on: pitch-linear,
// a w x h luma plane, and half-size chroma -- two planes (U, V) for the I420
// input, one interleaved plane for the NV12 output.
bool map_surface(Engine* e, const NvBufSurface* s, bool input, Mapping* m) {
  const int w = e->p.w, h = e->p.h;
  m->origin = s;
  m->params = s->surfaceList[0];
  m->params.mappedAddr = NvBufSurfaceMappedAddr();
  m->surf = *s;
  m->surf.surfaceList = &m->params;
  m->surf.batchSize = m->surf.numFilled = 1;

  if (NvBufSurfaceMapEglImage(&m->surf, 0) != 0) return fail(e, "NvBufSurfaceMapEglImage");
  m->egl = true;
  cudaError_t err = cudaGraphicsEGLRegisterImage(&m->res, m->params.mappedAddr.eglImage,
                                                 cudaGraphicsRegisterFlagsNone);
  if (err != cudaSuccess) return fail(e, "cudaGraphicsEGLRegisterImage", err);
  err = cudaGraphicsResourceGetMappedEglFrame(&m->fr, m->res, 0, 0);
  if (err != cudaSuccess) return fail(e, "cudaGraphicsResourceGetMappedEglFrame", err);

  const cudaEglFrame& fr = m->fr;
  const bool first = input ? e->in.size() == 1 : e->out.size() == 1;
  if (first) describe(e, input ? "in" : "out", *m);
  bool ok = fr.frameType == cudaEglFrameTypePitch && fr.planeCount == (input ? 3u : 2u) &&
            (int)fr.planeDesc[0].width == w && (int)fr.planeDesc[0].height == h &&
            fr.frame.pPitch[0].pitch >= (size_t)w && (int)fr.planeDesc[1].height == h / 2;
  if (ok && input) {
    // Planes must be U then V.
    ok = (m->params.colorFormat == NVBUF_COLOR_FORMAT_YUV420 ||
          m->params.colorFormat == NVBUF_COLOR_FORMAT_YUV420_ER) &&
         (int)fr.planeDesc[1].width == w / 2 && (int)fr.planeDesc[2].width == w / 2 &&
         (int)fr.planeDesc[2].height == h / 2;
  } else if (ok) {
    ok = pitch_bytes(fr, 1, w) >= (size_t)w;
  }
  if (!ok) {
    char buf[160];
    snprintf(buf, sizeof buf, "unexpected %s surface, expected pitch-linear %s %dx%d",
             input ? "input" : "output", input ? "I420" : "NV12", w, h);
    return fail(e, buf);
  }

  if (input) {
    for (int i = 0; i < 3; ++i) {
      err = make_plane_texture(fr.frame.pPitch[i].ptr, fr.planeDesc[i].width, fr.planeDesc[i].height,
                               fr.frame.pPitch[i].pitch, 1, &m->tex[i]);
      if (err != cudaSuccess) return fail(e, "texture on the input surface", err);
    }
  }
  return true;
}

Mapping* lookup(Engine* e, Cache* cache, const NvBufSurface* s, bool input) {
  const uint64_t fd = s->surfaceList[0].bufferDesc;
  Cache::iterator it = cache->find(fd);
  if (it != cache->end()) {
    if (it->second.origin == s) return &it->second;
    // The fd number now belongs to another buffer.
    unmap(&it->second);
    cache->erase(it);
  }
  if (cache->size() >= kMaxMappings) forget(cache);
  Mapping* m = &(*cache)[fd];
  return map_surface(e, s, input, m) ? m : nullptr;
}

}  // namespace

Engine* engine_new(const char* params_file) {
  Engine* e = new Engine();
  if (!load_params(params_file, &e->p)) {
    delete e;
    return nullptr;
  }
  const cudaError_t err = cudaStreamCreateWithFlags(&e->stream, cudaStreamNonBlocking);
  if (err != cudaSuccess) {
    fprintf(stderr, "[nvundistort] cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
    delete e;
    return nullptr;
  }
  return e;
}

void engine_free(Engine* e) {
  if (!e) return;
  engine_forget_surfaces(e);
  if (e->stream) cudaStreamDestroy(e->stream);
  delete e;
}

int engine_width(const Engine* e) { return e->p.w; }
int engine_height(const Engine* e) { return e->p.h; }
const char* engine_error(const Engine* e) { return e->error.c_str(); }
const char* engine_info(const Engine* e) { return e->info.c_str(); }

void engine_forget_surfaces(Engine* e) {
  forget(&e->in);
  forget(&e->out);
}

bool engine_process(Engine* e, NvBufSurface* in, NvBufSurface* out) {
  const Mapping* src = lookup(e, &e->in, in, true);
  if (!src) return false;
  const Mapping* dst = lookup(e, &e->out, out, false);
  if (!dst) return false;

  const Params& p = e->p;
  const int w = p.w, h = p.h;
  const cudaEglFrame& o = dst->fr;
  const dim3 block(32, 8);
  remap_y<<<dim3((w + 31) / 32, (h + 7) / 8), block, 0, e->stream>>>(
      src->tex[0], (uint8_t*)o.frame.pPitch[0].ptr, (int)o.frame.pPitch[0].pitch, p);
  remap_uv_planar<<<dim3((w / 2 + 31) / 32, (h / 2 + 7) / 8), block, 0, e->stream>>>(
      src->tex[1], src->tex[2], (uint8_t*)o.frame.pPitch[1].ptr, (int)pitch_bytes(o, 1, w), p);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return fail(e, "kernel launch", err);
  err = cudaStreamSynchronize(e->stream);
  if (err != cudaSuccess) return fail(e, "cudaStreamSynchronize", err);
  return true;
}

}  // namespace nvundistort
