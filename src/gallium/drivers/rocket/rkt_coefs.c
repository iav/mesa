/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include "util/u_inlines.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rkt_coefs.h"
#include "rkt_ml.h"

struct pipe_resource *
rkt_fill_weights(struct rkt_ml_subgraph *subgraph,
                 const struct pipe_ml_operation *poperation,
                 unsigned pad_kernels)
{
   struct pipe_context *pcontext = subgraph->context;
   unsigned weights_width = poperation->conv.weight_tensor->dims[1];
   unsigned weights_height = poperation->conv.weight_tensor->dims[2];
   unsigned input_channels = poperation->input_tensors[0]->dims[3];
   unsigned input_channels_real = poperation->input_tensors[0]->dims[3];
   unsigned output_channels = poperation->output_tensors[0]->dims[3];
   unsigned output_channels_real = poperation->output_tensors[0]->dims[3];
   unsigned weights_size;
   uint8_t zero_point = poperation->conv.weight_tensor->zero_point;
   struct pipe_transfer *transfer_out;
   void *map = poperation->conv.weight_tensor->data;
   uint8_t(*weights_in)[weights_width][weights_height][input_channels] = map;
   struct pipe_resource *rsc;
   uint8_t *weights_out;

   input_channels = MAX2(input_channels, FEATURE_ATOMIC_SIZE);

   /* TEST (iav RE 2026-08-22): RK3568 half-width weight atomic experiment */
   unsigned watom = getenv("RKT_WK") ? atoi(getenv("RKT_WK")) : WEIGHT_ATOMIC_SIZE;

   output_channels = align(output_channels, 2);
   if (rkt_is_depthwise(poperation))
      output_channels = 1;

   weights_size = weights_width * weights_height * output_channels *
                  align(input_channels, WEIGHT_ATOMIC_SIZE) * 2;

   rsc =
      pipe_buffer_create(pcontext->screen, 0, PIPE_USAGE_DEFAULT, weights_size);
   weights_out = pipe_buffer_map(pcontext, rsc, PIPE_MAP_WRITE, &transfer_out);

   unsigned input_channel_groups = watom;
   if (rkt_is_depthwise(poperation))
      input_channel_groups *= 2;

   unsigned input_channels_1 =
      DIV_ROUND_UP(input_channels, input_channel_groups);
   unsigned input_channels_2 = MIN2(input_channels, input_channel_groups);

   unsigned n = 0;
   if (rkt_is_depthwise(poperation) && input_channels_real > 32) {
      /* RK3568 depthwise with C>32: one 32-channel group per task; the
       * weight buffer is [group][tap (row-major)][32 channels], so the
       * per-group DCOMP_ADDR0 offset of KH*KW*32 bytes lands on a group
       * boundary (vendor mobilenet_v1 t8: DCOMP_ADDR0 = 288). */
      for (unsigned g = 0; g < DIV_ROUND_UP(input_channels_real, 32); g++)
         for (int y = 0; y < weights_height; y++)
            for (int x = 0; x < weights_width; x++)
               for (unsigned c = 0; c < 32; c++) {
                  unsigned ic = g * 32 + c;
                  weights_out[n++] = ic < input_channels_real
                                        ? weights_in[0][y][x][ic] - 0x80
                                        : 0;
               }
      goto packed;
   }
   if (!rkt_is_depthwise(poperation)) {
      /* RK3568 regular-conv weight layout, solved by probe-model RE
       * (2026-08-22, Test 43): known-weight models converted with
       * rknn-toolkit2 for rk3568 carry the packed buffer in the .rknn, and
       * marker weights give the exact byte positions:
       *
       *   pos(oc, ky, kx, ic) = (oc / 16) * KH*KW*16*icb
       *                       + (ky*KW + kx) * 16*icb
       *                       + (oc %% 16) * icb + ic,   icb = align(Cin, 16)
       *
       * i.e. kernel groups of SIXTEEN (not 32 as on RK3588), tap-major inside
       * a group, 16 kernels per tap block, ic last.  Bytes are w - 0x80. */
      /* ic dimension is cut into 32-channel slices that sit ABOVE the
       * taps: the order inside a 16-kernel group is [ic-slice][tap][oc%16]
       * [ic%32] (probe-DENSE128, a 3x3 Cin=128 dense probe, byte-exact
       * 0/36864 against the vendor packing; the old slice-under-tap order
       * was indistinguishable on 1x1 convolutions, which is all mobilenet
       * v1/v2 exercise with more than one slice). */
      /* ARGB / few-channel input (Cin<=8, probe-RGB byte-exact): the kernel
       * row is 8 bytes -- ic then zero padding to 8. */
      /* Compact tails, generalized (mobilenet_v2 RE 2026-08-23, WPOKE map on
       * op8 Cin=144/Cout=24): the tail ic slice (Cin % 32) uses
       * align(rem, 16)-byte kernel rows (16 bytes for a 16-channel tail,
       * poke 2064 -> oc1), and the tail kernel group (Cout % 16) is stored
       * compactly for EVERY conv, not just FC-shaped ones.  Groups follow
       * each other without padding; the WEIGHT_BYTES register already
       * matches this compact size (op8: 24 * 144 = 3456). */
      /* Kernel padding for fused adds (C % 32): the group geometry runs
       * over the padded count; the memset background (zero_point - 0x80)
       * is exactly a zero weight, so only the real kernels are written. */
      unsigned geom_kernels = pad_kernels ? pad_kernels : output_channels_real;
      unsigned slices = DIV_ROUND_UP(input_channels_real, 32);
      unsigned kgroups = DIV_ROUND_UP(geom_kernels, 16);
      unsigned rem_ic = input_channels_real % 32;
      /* The 8-byte kernel row belongs to the packed-RGB (ARGB) first-layer
       * mode only (probe-RGB byte-exact, Cin=3).  A plain conv with
       * Cin=4..8 announces 16 bytes per kernel in WEIGHT_SIZE1
       * (conv-tiny probes) and reads the rows at that stride. */
      unsigned row_tail = input_channels_real <= 3
                             ? 8
                             : rem_ic ? MIN2(align(rem_ic, 16), 32) : 32;
      /* bytes of one kernel row across all ic slices */
      unsigned rowsum = (slices - 1) * 32 + row_tail;

      memset(weights_out, zero_point - 0x80, weights_size);
      unsigned goff = 0;
      for (unsigned g = 0; g < kgroups; g++) {
         unsigned rows = MIN2(16, geom_kernels - g * 16);
         unsigned gtapblk = rows * rowsum;
         for (unsigned oc = g * 16;
              oc < MIN2(g * 16 + rows, output_channels_real); oc++) {
            for (unsigned ky = 0; ky < weights_width; ky++) {
               for (unsigned kx = 0; kx < weights_height; kx++) {
                  unsigned tap = ky * weights_height + kx;
                  for (unsigned ic = 0; ic < input_channels_real; ic++) {
                     unsigned s = ic / 32;
                     unsigned rowbytes = (s == slices - 1) ? row_tail : 32;
                     unsigned pos = goff +
                                    s * weights_width * weights_height *
                                       rows * 32 +
                                    tap * rows * rowbytes +
                                    (oc % 16) * rowbytes + (ic % 32);
                     weights_out[pos] = weights_in[oc][ky][kx][ic] - 0x80;
                  }
               }
            }
         }
         goff += weights_width * weights_height * gtapblk;
      }
      n = goff;
      assert(n <= weights_size);
      goto packed;
   }
   /* TEST (iav RE 2026-08-22): RKT_WLAY=1 — candidate RK3568 layout
    * [kernel group of 16][tap y*3+x][oc2 0..15][ic]. */
   if (getenv("RKT_WLAY") && atoi(getenv("RKT_WLAY")) == 2) {
      /* плоская укладка [oc][ty][tx][ic] для картирования */
      for (int oc = 0; oc < output_channels; oc++)
         for (int y = 0; y < weights_height; y++)
            for (int x = 0; x < weights_width; x++)
               for (int ic = 0; ic < input_channels; ic++)
                  weights_out[n++] =
                     (oc < output_channels_real && ic < input_channels_real)
                        ? weights_in[oc][x][y][ic] - 0x80 : 0;
      goto packed;
   }
   if (getenv("RKT_WLAY")) {
      for (int oc1 = 0; oc1 < DIV_ROUND_UP(output_channels, 16); oc1++) {
         for (int y = 0; y < weights_height; y++) {
            for (int x = 0; x < weights_width; x++) {
               for (int oc2 = 0; oc2 < 16; oc2++) {
                  for (int ic = 0; ic < input_channels; ic++) {
                     unsigned oc = oc1 * 16 + oc2;
                     if (oc >= output_channels_real || ic >= input_channels_real)
                        weights_out[n++] = 0;
                     else
                        weights_out[n++] = weights_in[oc][x][y][ic] - 0x80;
                  }
               }
            }
         }
      }
      goto packed;
   }
   for (int oc1 = 0; oc1 < DIV_ROUND_UP(output_channels, watom);
        oc1++) {
      for (int ic1 = 0; ic1 < input_channels_1; ic1++) {
         for (int x = 0; x < weights_width; x++) {
            for (int y = 0; y < weights_height; y++) {
               for (int oc2 = 0; oc2 < MIN2(output_channels, watom);
                    oc2++) {
                  for (int ic2 = 0; ic2 < input_channels_2; ic2++) {
                     unsigned oc = oc1 * watom + oc2;
                     unsigned ic = ic1 * input_channel_groups + ic2;
                     if (output_channels_real > 2 &&
                         oc >= align(output_channels_real, 2))
                        continue;

                     if (oc >= output_channels_real)
                        weights_out[n++] = 0x0;
                     else if (ic >= input_channels_real) {
                        if (ic2 < 16 || (input_channels_real % 32) > 16)
                           weights_out[n++] =
                              zero_point - 0x80; /* TODO: Why is the blob converting to
                                                    signed? It should be unsigned. */
                     } else
                        weights_out[n++] = weights_in[oc][x][y][ic] -
                                           0x80; /* TODO: Why is the blob converting to
                                                    signed? It should be unsigned. */
                  }
               }
            }
         }
      }
   }

packed:
   if (getenv("RKT_WDUMP")) {
      FILE *f = fopen(getenv("RKT_WDUMP"), "wb");
      if (f) { fwrite(weights_out, 1, weights_size, f); fclose(f); }
   }
   /* TEST (iav RE 2026-08-22): RKT_WPOKE=<byte offset> — обнулить всю
    * упаковку и поставить один байт 127: выход покажет, какой (oc,tap,ic)
    * этот байт кормит. */
   if (getenv("RKT_WPOKE")) {
      unsigned off = strtoul(getenv("RKT_WPOKE"), NULL, 0);
      memset(weights_out, 0, weights_size);
      if (off < weights_size)
         weights_out[off] = 127;
   }

   if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS)) {
      static int task = 0;
      rkt_dump_buffer(weights_out, "weights", 0, task++, 0, weights_size);
   }

   pipe_buffer_unmap(pcontext, transfer_out);

   return rsc;
}

static int32_t
calculate_bias_correction(struct rkt_ml_subgraph *subgraph,
                          const struct pipe_ml_operation *poperation,
                          unsigned oc, void *map)
{
   unsigned input_channels = poperation->input_tensors[0]->dims[3];
   unsigned input_zero_point = poperation->input_tensors[0]->zero_point;
   unsigned weights_width = poperation->conv.weight_tensor->dims[1];
   unsigned weights_height = poperation->conv.weight_tensor->dims[2];
   unsigned weight_zero_point = poperation->conv.weight_tensor->zero_point;
   uint8_t(*weights)[weights_width][weights_height][input_channels] = map;

   int32_t correction = 0;
   if (rkt_is_depthwise(poperation)) {
      for (unsigned x = 0; x < weights_width; x++) {
         for (unsigned y = 0; y < weights_height; y++) {
            correction += (weights[0][x][y][oc] - weight_zero_point) *
                          (input_zero_point - 0x80);
         }
      }
   } else {
      for (unsigned x = 0; x < weights_width; x++) {
         for (unsigned y = 0; y < weights_height; y++) {
            for (unsigned ic = 0; ic < input_channels; ic++) {
               correction += (weights[oc][x][y][ic] - weight_zero_point) *
                             (input_zero_point - 0x80);
            }
         }
      }
   }

   return correction;
}

struct pipe_resource *
rkt_fill_biases(struct rkt_ml_subgraph *subgraph,
                const struct pipe_ml_operation *poperation,
                unsigned *truncate_bits)
{
   struct pipe_context *pcontext = subgraph->context;
   unsigned output_channels = poperation->output_tensors[0]->dims[3];
   unsigned weights_size = poperation->conv.weight_tensor->dims[1];
   struct pipe_transfer *transfer_out;
   int32_t *biases_in = (int32_t *)poperation->conv.bias_tensor->data;
   void *weights = poperation->conv.weight_tensor->data;
   struct pipe_resource *rsc;
   uint32_t *biases;

   /* RK3568 vendor requant (RE 2026-08-22): the DPU BS stage takes all three
    * per-channel operands from the BRDMA stream, 8 bytes per channel packed in
    * groups of four channels: [bias int32 x4][ow int16 x4][mul uint16 x4].
    *   bias = quantized bias minus the input-zero-point correction (acc domain)
    *   ow   = weight_zero_point - 0x80 (weights are stored as w - 0x80;
    *          hardware computes acc - ow * sum(x))
    *   mul  = per-channel scale multiplier, shift fixed at 14 (BS_MUL_CFG);
    *          with per-tensor quantization this is exactly 1 << 14, and the
    *          whole conv_scale lives in OUT_CVT_SCALE/SHIFT as the vendor does.
    * Stream length follows the padded channel count the RDMA fetches. */
   unsigned padded_channels = align(output_channels, 32);
   bool bias_only_stream = true; /* per-tensor scheme (vendor probe-D2) */
   unsigned weight_zero_point = poperation->conv.weight_tensor->zero_point;
   uint8_t *stream;

   *truncate_bits = 0;

   rsc = pipe_buffer_create(pcontext->screen, 0, PIPE_USAGE_DEFAULT,
                            padded_channels * 8);
   stream = pipe_buffer_map(pcontext, rsc, PIPE_MAP_WRITE, &transfer_out);
   biases = (uint32_t *)stream;

   if (bias_only_stream) {
      int32_t *b32 = (int32_t *)stream;
      for (unsigned oc = 0; oc < padded_channels; oc++) {
         if (oc >= output_channels) { b32[oc] = 0; continue; }
         int32_t corr =
            calculate_bias_correction(subgraph, poperation, oc, weights);
         b32[oc] = biases_in[oc] - corr;
      }
      goto stream_done;
   }
   for (unsigned g = 0; g < padded_channels / 4; g++) {
      int32_t *bias32 = (int32_t *)(stream + g * 32);
      int16_t *ow16 = (int16_t *)(stream + g * 32 + 16);
      uint16_t *mul16 = (uint16_t *)(stream + g * 32 + 24);

      for (unsigned j = 0; j < 4; j++) {
         unsigned oc = g * 4 + j;

         if (oc >= output_channels) {
            bias32[j] = 0;
            ow16[j] = 0;
            mul16[j] = 1 << 14;
         ow16[j] = 0; /* PROBE4 */
            continue;
         }

         int32_t corr =
            calculate_bias_correction(subgraph, poperation, oc, weights);
         bias32[j] = biases_in[oc] - corr;
         /* hardware ADDS ow*sum(x): ow = 0x80 - wzp (Test 42) */
         ow16[j] = 0x80 - weight_zero_point;
         mul16[j] = 1 << 14;
         if (getenv("RKT_MULSH"))
            mul16[j] = 1 << (14 - atoi(getenv("RKT_MULSH")));
         if (getenv("RKT_WPOKE")) {
            bias32[j] = 0; ow16[j] = 0; mul16[j] = 1 << 14;
         }
         /* TEST probes: RKT_PROBE selects synthetic stream contents */
         if (getenv("RKT_PROBE")) {
            int pr = atoi(getenv("RKT_PROBE"));
            uint16_t *slots = (uint16_t *)(stream + g * 32);
            unsigned k;
            switch (pr) {
            case 1: /* staircase bias, ow 0, mul 1<<14 */
               bias32[j] = (oc + 1) * 2048; ow16[j] = 0; mul16[j] = 1 << 14;
               break;
            case 5: /* globally unique small u16 in every slot */
               for (k = 0; k < 16; k++)
                  slots[k] = 4 * (g * 16 + k + 1);
               break;
            }
         }
      }
   }

stream_done:
   if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS)) {
      static int task = 0;
      rkt_dump_buffer((uint8_t *)biases, "biases", 0, task++, 0,
                      padded_channels * 8);
   }

   pipe_buffer_unmap(pcontext, transfer_out);

   return rsc;
}
