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
   uint8_t output_zero_point;
   float output_scale;

   unsigned weights_width;
   unsigned weights_height;
   uint8_t weights_zero_point;
   float weights_scale;

   int add_tensor;

   struct util_dynarray tasks; /* struct split_task */
};

struct rkt_ml_subgraph {
   struct pipe_ml_subgraph base;

   struct pipe_context *context;
   struct util_dynarray operations; /* rkt_operation */
   struct util_dynarray tensors;    /* pipe_resource* */
};

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
