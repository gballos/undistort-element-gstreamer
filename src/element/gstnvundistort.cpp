// gstnvundistort.cpp -- GStreamer element `nvundistort`: exact lens undistortion
// of nvv4l2decoder's frames on the GPU.
//
//   nvv4l2decoder mjpeg=1 ! nvundistort params-file=<abs path>
//     ! 'video/x-raw(memory:NVMM),format=NV12' ! ...
//
// Sink: NVMM I420, what nvv4l2decoder outputs for MJPEG. Src: NVMM NV12 of the
// same size, in buffers from the element's own pool. The I420 -> NV12
// conversion happens inside the remap, so no nvvidconv is needed in front.
// The pixel values keep the decoder's range (full range for JPEG).
//
// If undistortion cannot run (no or unusable params-file, frame size other
// than the calibrated one, a CUDA error) the element says so once and from
// then on only converts I420 -> NV12: frames keep flowing, distorted.
//
// The maths and kernels are in ../undistort_kernels.cuh, the CUDA host code in
// undistort_engine.cu.

#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include "gstnvdsbufferpool.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"

#include "undistort_engine.h"

GST_DEBUG_CATEGORY_STATIC(gst_nvundistort_debug);
#define GST_CAT_DEFAULT gst_nvundistort_debug

#define GST_TYPE_NVUNDISTORT (gst_nvundistort_get_type())
G_DECLARE_FINAL_TYPE(GstNvUndistort, gst_nvundistort, GST, NVUNDISTORT, GstBaseTransform)

struct _GstNvUndistort {
  GstBaseTransform parent;

  gchar* params_file;
  guint stats_interval;

  nvundistort::Engine* engine;  // null => frames are only converted
  gboolean info_logged;
  guint stats_frames;
  gdouble stats_ms;
};

G_DEFINE_TYPE(GstNvUndistort, gst_nvundistort, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_PARAMS_FILE, PROP_STATS_INTERVAL };

// Output buffers alive at once: the one being filled plus what downstream
// queues hold. Each is a full frame (18 MB at 4608x2592), hence the cap.
#define POOL_MIN_BUFFERS 4
#define POOL_MAX_BUFFERS 8

static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM", "I420")));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS(GST_VIDEO_CAPS_MAKE_WITH_FEATURES("memory:NVMM", "NV12")));

// Stops undistorting for the rest of the run and says why, once.
static void gst_nvundistort_disable(GstNvUndistort* self, const gchar* reason) {
  GST_ELEMENT_WARNING(self, LIBRARY, FAILED,
                      ("%s -- UNDISTORTION DISABLED, frames pass through distorted", reason), (NULL));
  nvundistort::engine_free(self->engine);
  self->engine = NULL;
}

static void gst_nvundistort_set_property(GObject* object, guint id, const GValue* value,
                                         GParamSpec* pspec) {
  GstNvUndistort* self = GST_NVUNDISTORT(object);
  switch (id) {
    case PROP_PARAMS_FILE:
      g_free(self->params_file);
      self->params_file = g_value_dup_string(value);
      break;
    case PROP_STATS_INTERVAL:
      self->stats_interval = g_value_get_uint(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
  }
}

static void gst_nvundistort_get_property(GObject* object, guint id, GValue* value,
                                         GParamSpec* pspec) {
  GstNvUndistort* self = GST_NVUNDISTORT(object);
  switch (id) {
    case PROP_PARAMS_FILE:
      g_value_set_string(value, self->params_file);
      break;
    case PROP_STATS_INTERVAL:
      g_value_set_uint(value, self->stats_interval);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, pspec);
  }
}

static void gst_nvundistort_finalize(GObject* object) {
  g_free(GST_NVUNDISTORT(object)->params_file);
  G_OBJECT_CLASS(gst_nvundistort_parent_class)->finalize(object);
}

static gboolean gst_nvundistort_start(GstBaseTransform* trans) {
  GstNvUndistort* self = GST_NVUNDISTORT(trans);
  self->info_logged = FALSE;
  self->stats_frames = 0;
  self->stats_ms = 0.0;
  if (!self->params_file || !*self->params_file) {
    gst_nvundistort_disable(self, "params-file is not set");
    return TRUE;
  }
  self->engine = nvundistort::engine_new(self->params_file);
  if (!self->engine) gst_nvundistort_disable(self, "unusable params-file (see stderr)");
  return TRUE;
}

static gboolean gst_nvundistort_stop(GstBaseTransform* trans) {
  GstNvUndistort* self = GST_NVUNDISTORT(trans);
  nvundistort::engine_free(self->engine);
  self->engine = NULL;
  return TRUE;
}

// Same caps on both sides except the format: I420 in, NV12 out.
static GstCaps* gst_nvundistort_transform_caps(GstBaseTransform*, GstPadDirection direction,
                                               GstCaps* caps, GstCaps* filter) {
  GstCaps* other = gst_caps_copy(caps);
  const gchar* format = direction == GST_PAD_SINK ? "NV12" : "I420";
  for (guint i = 0; i < gst_caps_get_size(other); ++i)
    gst_structure_set(gst_caps_get_structure(other, i), "format", G_TYPE_STRING, format, NULL);
  if (filter) {
    GstCaps* filtered = gst_caps_intersect_full(filter, other, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref(other);
    other = filtered;
  }
  return other;
}

static gboolean gst_nvundistort_set_caps(GstBaseTransform* trans, GstCaps* incaps, GstCaps*) {
  GstNvUndistort* self = GST_NVUNDISTORT(trans);
  if (!self->engine) return TRUE;
  // The output pool is replaced after a caps change, and the decoder's may be.
  nvundistort::engine_forget_surfaces(self->engine);

  gint w = 0, h = 0;
  const GstStructure* s = gst_caps_get_structure(incaps, 0);
  gst_structure_get_int(s, "width", &w);
  gst_structure_get_int(s, "height", &h);
  const gint cw = nvundistort::engine_width(self->engine), ch = nvundistort::engine_height(self->engine);
  if (w != cw || h != ch) {
    gchar* reason = g_strdup_printf("frames are %dx%d but %s is for %dx%d", w, h, self->params_file, cw, ch);
    gst_nvundistort_disable(self, reason);
    g_free(reason);
  }
  return TRUE;
}

static gboolean gst_nvundistort_get_unit_size(GstBaseTransform*, GstCaps*, gsize* size) {
  *size = sizeof(NvBufSurface);  // what an NVMM GstBuffer holds
  return TRUE;
}

// Output buffers always come from our own DeepStream pool: NVMM surfaces in the
// negotiated output format.
static gboolean gst_nvundistort_decide_allocation(GstBaseTransform* trans, GstQuery* query) {
  GstCaps* caps = NULL;
  gst_query_parse_allocation(query, &caps, NULL);
  if (!caps) return FALSE;

  GstBufferPool* pool = gst_nvds_buffer_pool_new();
  GstStructure* config = gst_buffer_pool_get_config(pool);
  gst_buffer_pool_config_set_params(config, caps, sizeof(NvBufSurface), POOL_MIN_BUFFERS,
                                    POOL_MAX_BUFFERS);
  gst_structure_set(config, "memtype", G_TYPE_UINT, (guint)NVBUF_MEM_DEFAULT, "gpu-id", G_TYPE_UINT,
                    0u, "batch-size", G_TYPE_UINT, 1u, NULL);
  if (!gst_buffer_pool_set_config(pool, config)) {
    GST_ERROR_OBJECT(trans, "output pool rejected its configuration");
    gst_object_unref(pool);
    return FALSE;
  }
  if (gst_query_get_n_allocation_pools(query) > 0)
    gst_query_set_nth_allocation_pool(query, 0, pool, sizeof(NvBufSurface), POOL_MIN_BUFFERS,
                                      POOL_MAX_BUFFERS);
  else
    gst_query_add_allocation_pool(query, pool, sizeof(NvBufSurface), POOL_MIN_BUFFERS,
                                  POOL_MAX_BUFFERS);
  gst_object_unref(pool);
  return TRUE;
}

static GstFlowReturn gst_nvundistort_transform(GstBaseTransform* trans, GstBuffer* inbuf,
                                               GstBuffer* outbuf) {
  GstNvUndistort* self = GST_NVUNDISTORT(trans);
  GstMapInfo in_map, out_map;
  if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ)) return GST_FLOW_ERROR;
  if (!gst_buffer_map(outbuf, &out_map, GST_MAP_WRITE)) {
    gst_buffer_unmap(inbuf, &in_map);
    return GST_FLOW_ERROR;
  }
  NvBufSurface* in = (NvBufSurface*)in_map.data;
  NvBufSurface* out = (NvBufSurface*)out_map.data;
  GstFlowReturn ret = GST_FLOW_OK;
  const gint64 t0 = g_get_monotonic_time();

  if (self->engine && !nvundistort::engine_process(self->engine, in, out))
    gst_nvundistort_disable(self, nvundistort::engine_error(self->engine));
  if (!self->engine) {
    NvBufSurfTransformParams params = {};
    if (NvBufSurfTransform(in, out, &params) != NvBufSurfTransformError_Success) {
      GST_ELEMENT_ERROR(self, STREAM, FAILED, ("I420 -> NV12 conversion failed"), (NULL));
      ret = GST_FLOW_ERROR;
    }
  }
  out->numFilled = 1;

  if (self->engine) {
    if (!self->info_logged) {
      self->info_logged = TRUE;
      GST_INFO_OBJECT(self, "surfaces: %s", nvundistort::engine_info(self->engine));
    }
    if (self->stats_interval > 0) {
      self->stats_ms += (g_get_monotonic_time() - t0) / 1000.0;
      if (++self->stats_frames % self->stats_interval == 0) {
        g_printerr("[nvundistort] %.2f ms/frame (mean of %u)\n",
                   self->stats_ms / self->stats_interval, self->stats_interval);
        self->stats_ms = 0.0;
      }
    }
  }

  gst_buffer_unmap(outbuf, &out_map);
  gst_buffer_unmap(inbuf, &in_map);
  return ret;
}

static void gst_nvundistort_class_init(GstNvUndistortClass* klass) {
  GObjectClass* gobject_class = G_OBJECT_CLASS(klass);
  GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
  GstBaseTransformClass* trans_class = GST_BASE_TRANSFORM_CLASS(klass);

  gobject_class->set_property = gst_nvundistort_set_property;
  gobject_class->get_property = gst_nvundistort_get_property;
  gobject_class->finalize = gst_nvundistort_finalize;

  g_object_class_install_property(
      gobject_class, PROP_PARAMS_FILE,
      g_param_spec_string("params-file", "Parameter file",
                          "Parameter file written by tools/undistort_params.py (read at start)",
                          NULL, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      gobject_class, PROP_STATS_INTERVAL,
      g_param_spec_uint("stats-interval", "Stats interval",
                        "Print the mean time per frame on stderr every N frames (0 = never)", 0,
                        G_MAXUINT, 0, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata(element_class, "Lens undistortion", "Filter/Effect/Video",
                                        "Undistorts NVMM frames on the GPU (OpenCV rational + "
                                        "tangential model)", "Giorgos Ballos");
  gst_element_class_add_static_pad_template(element_class, &sink_template);
  gst_element_class_add_static_pad_template(element_class, &src_template);

  trans_class->start = GST_DEBUG_FUNCPTR(gst_nvundistort_start);
  trans_class->stop = GST_DEBUG_FUNCPTR(gst_nvundistort_stop);
  trans_class->transform_caps = GST_DEBUG_FUNCPTR(gst_nvundistort_transform_caps);
  trans_class->set_caps = GST_DEBUG_FUNCPTR(gst_nvundistort_set_caps);
  trans_class->get_unit_size = GST_DEBUG_FUNCPTR(gst_nvundistort_get_unit_size);
  trans_class->decide_allocation = GST_DEBUG_FUNCPTR(gst_nvundistort_decide_allocation);
  trans_class->transform = GST_DEBUG_FUNCPTR(gst_nvundistort_transform);
}

static void gst_nvundistort_init(GstNvUndistort*) {}

static gboolean plugin_init(GstPlugin* plugin) {
  GST_DEBUG_CATEGORY_INIT(gst_nvundistort_debug, "nvundistort", 0, "lens undistortion");
  return gst_element_register(plugin, "nvundistort", GST_RANK_NONE, GST_TYPE_NVUNDISTORT);
}

#define PACKAGE "nvundistort"
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, nvundistort,
                  "Lens undistortion of NVMM frames on the GPU", plugin_init, "1.0", "MIT/X11",
                  "nvundistort", "https://github.com/gballos/undistort-element-gstreamer")
