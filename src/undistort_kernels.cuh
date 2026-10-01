// undistort_kernels.cuh -- the undistortion maths and the NV12 remap kernels.
//
// Shared by every host wrapper (the nvivafilter library, nvundistort.cu, and
// the GStreamer element, element/undistort_engine.cu), so the model exists in
// one place. Kernels read the source frame through a
// texture (make_plane_texture) and write to `dst`, which must not be the memory
// the texture reads: each output pixel reads neighbouring input pixels.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace nvundistort {

struct Params {
  int w, h;
  float fx, fy, cx, cy;      // camera matrix K of the distorted input
  float ifx, ify, ncx, ncy;  // camera matrix new_K of the output: 1/fx, 1/fy, cx, cy
  float k1, k2, p1, p2, k3, k4, k5, k6;
};

// Output pixel (u, v) -> source pixel (xs, ys). Same steps as
// cv::initUndistortRectifyMap with R = I: unproject with new_K, apply the lens
// model forward, project with K.
__device__ __forceinline__ void src_coords(const Params& p, float u, float v,
                                           float& xs, float& ys) {
  const float x = (u - p.ncx) * p.ifx;
  const float y = (v - p.ncy) * p.ify;
  const float r2 = x * x + y * y;
  const float kr = (1.f + ((p.k3 * r2 + p.k2) * r2 + p.k1) * r2) /
                   (1.f + ((p.k6 * r2 + p.k5) * r2 + p.k4) * r2);
  const float xd = x * kr + 2.f * p.p1 * x * y + p.p2 * (r2 + 2.f * x * x);
  const float yd = y * kr + p.p1 * (r2 + 2.f * y * y) + 2.f * p.p2 * x * y;
  xs = p.fx * xd + p.cx;
  ys = p.fy * yd + p.cy;
}

// Texture over one pitch-linear 8-bit plane: `channels` 1 for Y, 2 for NV12's
// interleaved U,V pairs (one fetch returns both). The texture unit does the
// bilinear blend, with 8-bit fractional weights, so a result can differ from a
// float blend by one level. Edges clamp; reads return [0, 1]. `ptr` and `pitch`
// must meet the device's texture alignment (cudaMallocPitch guarantees it).
inline cudaError_t make_plane_texture(const void* ptr, int width, int height, size_t pitch,
                                      int channels, cudaTextureObject_t* tex) {
  cudaResourceDesc res = {};
  res.resType = cudaResourceTypePitch2D;
  res.res.pitch2D.devPtr = const_cast<void*>(ptr);
  res.res.pitch2D.desc = channels == 2 ? cudaCreateChannelDesc<uchar2>()
                                       : cudaCreateChannelDesc<unsigned char>();
  res.res.pitch2D.width = width;
  res.res.pitch2D.height = height;
  res.res.pitch2D.pitchInBytes = pitch;
  cudaTextureDesc desc = {};
  desc.addressMode[0] = desc.addressMode[1] = cudaAddressModeClamp;
  desc.filterMode = cudaFilterModeLinear;
  desc.readMode = cudaReadModeNormalizedFloat;
  desc.normalizedCoords = 0;
  return cudaCreateTextureObject(tex, &res, &desc, nullptr);
}

// [0, 1] texture read -> byte, rounded.
__device__ __forceinline__ uint8_t to_byte(float v) { return (uint8_t)(v * 255.f + 0.5f); }

// The comparisons are also false for NaN, which then lands on the border value.
__device__ __forceinline__ bool inside(float x, float y, int w, int h) {
  return x >= 0.f && y >= 0.f && x <= (float)(w - 1) && y <= (float)(h - 1);
}

// Texture coordinates put the centre of pixel i at i + 0.5, while the model
// (like OpenCV) puts it at i, hence the + 0.5 on every fetch.
static __global__ void remap_y(cudaTextureObject_t src, uint8_t* __restrict__ dst,
                               int dst_pitch, Params p) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= p.w || v >= p.h) return;

  float xs, ys;
  src_coords(p, (float)u, (float)v, xs, ys);
  uint8_t out = 0;  // black outside the source frame
  if (inside(xs, ys, p.w, p.h)) out = to_byte(tex2D<float>(src, xs + 0.5f, ys + 0.5f));
  dst[v * dst_pitch + u] = out;
}

// NV12 chroma: half resolution, U and V interleaved. Chroma sample j is centred
// on luma coordinate 2j + 0.5, and a luma coordinate L is chroma (L - 0.5) / 2.
static __global__ void remap_uv(cudaTextureObject_t src, uint8_t* __restrict__ dst,
                                int dst_pitch, Params p) {
  const int cw = p.w >> 1, ch = p.h >> 1;
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= cw || v >= ch) return;

  float xs, ys;
  src_coords(p, 2.f * u + 0.5f, 2.f * v + 0.5f, xs, ys);
  xs = (xs - 0.5f) * 0.5f;
  ys = (ys - 0.5f) * 0.5f;
  uint8_t cu = 128, cv = 128;  // neutral chroma outside the source frame
  if (inside(xs, ys, cw, ch)) {
    const float2 c = tex2D<float2>(src, xs + 0.5f, ys + 0.5f);
    cu = to_byte(c.x);
    cv = to_byte(c.y);
  }
  uint8_t* o = dst + v * dst_pitch + 2 * u;
  o[0] = cu;
  o[1] = cv;
}

// As remap_uv, for an I420 source: U and V come from two one-channel textures
// and are written as NV12's interleaved pairs.
static __global__ void remap_uv_planar(cudaTextureObject_t src_u, cudaTextureObject_t src_v,
                                       uint8_t* __restrict__ dst, int dst_pitch, Params p) {
  const int cw = p.w >> 1, ch = p.h >> 1;
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  const int v = blockIdx.y * blockDim.y + threadIdx.y;
  if (u >= cw || v >= ch) return;

  float xs, ys;
  src_coords(p, 2.f * u + 0.5f, 2.f * v + 0.5f, xs, ys);
  xs = (xs - 0.5f) * 0.5f;
  ys = (ys - 0.5f) * 0.5f;
  uint8_t cu = 128, cv = 128;  // neutral chroma outside the source frame
  if (inside(xs, ys, cw, ch)) {
    cu = to_byte(tex2D<float>(src_u, xs + 0.5f, ys + 0.5f));
    cv = to_byte(tex2D<float>(src_v, xs + 0.5f, ys + 0.5f));
  }
  uint8_t* o = dst + v * dst_pitch + 2 * u;
  o[0] = cu;
  o[1] = cv;
}

}  // namespace nvundistort
