/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#ifndef RKT_ML_H
#define RKT_ML_H

#include <util/u_dynarray.h>

#include "rkt_device.h"

// http://nvdla.org/hw/v1/ias/unit_description.html#convolution-buffer
/* TEST (iav RE, 2026-08-22): RK3568 banks are 64 KiB, not 32.  Derived from
 * the vendor command stream: the first convolution of mobilenet_v1 (224 wide,
 * 7 data banks) is split into bands of 127 and 98 input rows.  At 14 entries
 * per row that first band needs 128 slices, which only fits if a bank holds
 * 256 entries of 256 bytes.  With 32768 mesa computed half the capacity and
 * split the same layer into five bands instead of two. */
/* RK3568 (RE 2026-08-22): 8 banks of 32 KiB, 32-byte entries (1024 per
 * bank).  Verified against all 51 mobilenet_v1 vendor tasks: data banks =
 * ceil(entries_per_slice * H / 1024), weight banks = 8 - data banks. */
/* One DPU LUT domain for every SiLU layer: x in [-8, 8) real, one LUT
 * unit = 1/2048, 32 units per table entry, y = 2 * silu(x) in LUT units
 * (32767 at 8.0; beyond that the LO overflow slope continues y = 2x,
 * below -8 silu is 0 to well under an output LSB).  The LO table starts
 * inside the LE range because its first entries misbehave (RE-LOG
 * Test 75). */
#define RKT_LUT_SCALE    (1.0f / 2048.0f)
#define RKT_LUT_LO_START (-512)

#define CBUF_BANK_SIZE        32768
/* CBUF_BANKS is SoC-specific:
 *   RK3588: 12 banks (384 KiB CBUF)
 *   RK3568:  8 banks (256 KiB CBUF) — per RK3568 TRM Part2 section 9.1
 * TODO: select per-SoC at runtime via rkt_device. */
#define CBUF_BANKS            8
#define CBUF_ENTRIES_PER_BANK 1024
#define CBUF_ENTRY_SIZE       (CBUF_BANK_SIZE / CBUF_ENTRIES_PER_BANK)
#define FEATURE_ATOMIC_SIZE   16
#define WEIGHT_ATOMIC_SIZE    32
#define ATOMIC_K_SIZE         16

struct split_task {
   /* RK3568 depthwise with more than 32 channels runs as one task per
    * 32-channel group (vendor: full tasks + address-delta tasks); all
    * buffer addresses shift by this group index. */
   unsigned channel_group;
   unsigned num;

   unsigned top_slice;
   unsigned bottom_slice;
   unsigned num_overlap_slices;
   unsigned num_retain_slices;
   unsigned convolutions;

   unsigned pad_top;
   unsigned pad_bottom;
   unsigned pad_left;
   unsigned pad_right;

   unsigned stride_x;
   unsigned stride_y;

   unsigned input_width;
   unsigned input_height;
   unsigned input_channels;
   unsigned input_channels_real;
   unsigned input_zero_point;
   float input_scale;
   unsigned input_data_entries;
   int input_line_stride;
   int input_surface_stride;
   unsigned input_offset;

   unsigned output_width;
   unsigned output_height;
   unsigned output_channels;
   unsigned output_channels_real;
   unsigned output_zero_point;
   float output_scale;
   int output_surface_stride;
   unsigned output_offset;

   unsigned weights_width;
   unsigned weights_height;
   unsigned weights_kernels;
   unsigned weights_zero_point;
   float weights_scale;

   unsigned input_banks;
   unsigned weights_banks;

   unsigned atomic_count;
   unsigned surfaces_per_row;

   unsigned regcfg_amount;
   uint32_t regcfg_addr;
};

struct rkt_operation {
   struct pipe_resource *regcmd;
   /* PC task-DMA descriptor array (RK3568 vendor 40-byte layout). */
   struct pipe_resource *task_descs;
   struct pipe_resource *weights;
   struct pipe_resource *biases;

   bool depthwise;
   bool reuse_weights_cbuf;
   unsigned truncate_bits;
   unsigned padding_top;
   unsigned padding_bottom;
   unsigned padding_left;
   unsigned padding_right;
   unsigned stride;

   bool addition_input;
   int addition_offset;
   float addition_scale;
   bool addition_relu;
   bool relu;

   /* RK3568 PPU max pooling (RE 2026-08-23, vendor resnet18 task 1): the
    * operation is a bare PPU/PPU_RDMA register chunk started with a
    * broadcast OP_EN 0x60, linked into the graph's PC chain like any
    * other stream but NOT counted as a task (the vendor's TASK_NUMBER
    * only counts the 0x1f convolution chunks).  Reuses weights_width and
    * weights_height for the pooling kernel, padding and stride fields
    * for the rest. */
   bool is_pool;
   /* Average pooling instead of max: OPERATION_MODE method bits 0 and the
    * RECIP_KERNEL registers = 65536/kernel (vendor probe-AP: 7 -> 9362). */
   bool pool_avg;

   unsigned input_index;
   unsigned input_width;
   unsigned input_height;
   unsigned input_channels;
   uint8_t input_zero_point;
   float input_scale;

   unsigned output_index;
   unsigned output_width;
   unsigned output_height;
   unsigned output_channels;
   /* Non-zero: pad the kernel count to this (RK3568 fused adds with
    * C % 32 != 0 run as align(C, 32)-kernel convolutions with zero
    * weights/biases in the tail; the unpack still uses output_channels). */
   unsigned output_channels_pad;
   uint8_t output_zero_point;
   float output_scale;

   unsigned weights_width;
   unsigned weights_height;
   uint8_t weights_zero_point;
   float weights_scale;
   /* Per-channel weight quantization: the BS stream carries the full
    * [bias][ow][mul] triple, weights_scale holds max(scales) and the
    * per-channel remainder lives in the mul entries. */
   bool per_channel;

   int add_tensor;

   /* Byte offset added to the output tensor's base address.  Channel
    * concatenation writes each producer into its slice of the shared
    * output BO: in the planar 8-channel surface layout a producer's
    * surfaces are self-contained, so the concat is just a base offset. */
   unsigned dst_offset;

   struct util_dynarray tasks; /* struct split_task */

   /* Fused x * sigmoid(x) (SiLU) through the DPU lookup table (vendor
    * ConvExSwish, RE-LOG Tests 73/75): the convolution runs the BN
    * multiplier + LUT + OUT_CVT path instead of the bypass; the tables
    * themselves reach the unit through the kernel (drm_rocket_job
    * lut_data). */
   bool silu;
   /* Value of one LUT-domain unit (the table spans [-16384, 16384]). */
   float lut_scale;
   /* BN multiplier mapping the accumulator into the LUT domain:
    * x_lut = acc * lut_mul >> lut_shift. */
   unsigned lut_mul;
   unsigned lut_shift;
   int16_t lut_le[513];
   int16_t lut_lo[513];
};

/* Dimensions of a concatenation output tensor.  Several operations then
 * share that output_index (each with its own dst_offset and channel
 * count), so readback cannot take the dims from find_producer(). */
struct rkt_concat_shape {
   unsigned index;
   unsigned width;
   unsigned height;
   unsigned channels;
};

struct rkt_ml_subgraph {
   struct pipe_ml_subgraph base;

   struct pipe_context *context;
   struct util_dynarray operations; /* rkt_operation */
   struct util_dynarray tensors;    /* pipe_resource* */
   struct util_dynarray concat_shapes; /* rkt_concat_shape */
   /* DPU lookup tables handed to the kernel with the job (LE then LO,
    * 515 words each): every SiLU convolution shares the one table in
    * the global LUT domain. */
   bool has_lut;
   uint16_t lut_words[1030];
};

/* RK3568 feature surfaces are padded to whole 32-byte CBUF entries: the
 * per-surface pixel count is aligned to 4 (vendor: 7x7 maps use surface
 * stride 52 everywhere -- CNA DMA_CON2, DPU DST_SURF_STRIDE, SURFACE_ADD;
 * a 1x1 FC-shaped map keeps stride 1). */
static inline unsigned
rkt_surf_px(unsigned px)
{
   return px > 1 ? align(px, 4) : px;
}

bool
rkt_ml_operation_supported(struct pipe_ml_device *pdevice, const struct pipe_ml_operation *operation);

struct pipe_ml_subgraph *
rkt_ml_subgraph_create(struct pipe_ml_device *pdevice,
                       const struct pipe_ml_operation *poperations,
                       unsigned count);

void rkt_ml_subgraph_invoke(struct pipe_context *pcontext,
                            struct pipe_ml_subgraph *psubgraph,
                            unsigned inputs_count, unsigned input_idxs[],
                            void *inputs[], bool is_signed[]);

void rkt_ml_subgraph_read_outputs(struct pipe_context *pcontext,
                                  struct pipe_ml_subgraph *psubgraph,
                                  unsigned outputs_count,
                                  unsigned output_idxs[], void *outputs[],
                                  bool is_signed[]);

void rkt_ml_subgraph_destroy(struct pipe_ml_device *pdevice,
                             struct pipe_ml_subgraph *psubgraph);

struct rkt_resource *rkt_get_tensor(struct rkt_ml_subgraph *subgraph,
                                    unsigned idx);

bool rkt_is_depthwise(const struct pipe_ml_operation *poperation);

void rkt_dump_buffer(const uint8_t *ptr, char *name, int operation_nr,
                     int suboperation_nr, int offset, unsigned size);

#endif /* RKT_ML_H */
