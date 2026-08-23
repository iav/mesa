/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include "rkt_task.h"
#include "rkt_ml.h"

static unsigned
calc_entries_per_slice(struct rkt_operation *operation)
{
   /* RK3568 (RE 2026-08-22): the feature atomic is 8 channels (8 bytes) and a
    * CBUF entry is 32 bytes (4 atomics) -- half the RK3588 sizes.  Verified
    * against all 51 mobilenet_v1 vendor tasks: entries = W * ceil(C/8) / 4
    * reproduces every DATA_ENTRIES value (C = 8..512, W = 7..224). */
   unsigned atomics_per_entry = 4;
   /* Depthwise with C>32 runs as 32-channel-group tasks -- CBUF holds one
    * group at a time. */
   unsigned eff_channels = operation->depthwise
                              ? MIN2(operation->input_channels, 32)
                              : operation->input_channels;
   unsigned total_c_atomics = DIV_ROUND_UP(eff_channels, 8);
   unsigned last_c_atomics = total_c_atomics % atomics_per_entry;
   unsigned int_c_entries =
      (total_c_atomics / atomics_per_entry) * operation->input_width;
   unsigned frac_c_entries =
      (last_c_atomics == 3)
         ? operation->input_width
         : DIV_ROUND_UP(last_c_atomics * operation->input_width,
                        atomics_per_entry);

   return int_c_entries + frac_c_entries;
}

static unsigned
calc_input_banks(struct rkt_operation *operation)
{
   unsigned entries_per_slice = calc_entries_per_slice(operation);
   return DIV_ROUND_UP(entries_per_slice * operation->input_height,
                       CBUF_ENTRIES_PER_BANK);
}

static unsigned
calc_weights_banks(struct rkt_operation *operation)
{
   unsigned bpe = sizeof(uint8_t);
   unsigned bytes = operation->weights_width * operation->weights_height *
                    operation->input_channels * bpe;
   unsigned entries;
   unsigned banks;

   if (!operation->depthwise)
      bytes *= operation->output_channels;
   entries = DIV_ROUND_UP(bytes, CBUF_ENTRY_SIZE);
   banks = DIV_ROUND_UP(entries, CBUF_ENTRIES_PER_BANK);

   /* TEST (iav RE, 2026-08-21): the vendor gives this layer one weight bank and
    * seven data banks (CBUF_CON0 0x17); with the extra bank mesa keeps two and
    * six, and the 224-wide RGB layer then stalls on its second band. */
   /* banks++; */

   return banks;
}

static unsigned
calc_line_stride(unsigned width)
{
   return width * ATOMIC_K_SIZE * sizeof(uint8_t);
}

static void
fill_task(struct rkt_ml_subgraph *subgraph,
          struct rkt_operation *operation,
          struct split_task *task)
{
   task->stride_x = operation->stride;
   task->stride_y = operation->stride;

   task->input_width = operation->input_width;
   if (task->input_width == 8 &&
       (operation->addition_input || operation->add_tensor != -1))
      task->input_width *= 2;

   task->input_height = operation->input_height;
   task->input_channels =
      operation->input_channels == 3
         ? 8
         : align(MAX2(operation->input_channels, FEATURE_ATOMIC_SIZE),
                 FEATURE_ATOMIC_SIZE);
   task->input_channels_real = operation->input_channels;
   /* Depthwise with C>32: each task covers one 32-channel group (vendor
    * t6..t9 of mobilenet_v1). */
   if (operation->depthwise && operation->input_channels > 32) {
      task->input_channels = 32;
      task->input_channels_real = 32;
   }
   task->input_zero_point = operation->input_zero_point;
   task->input_scale = operation->input_scale;

   task->output_width = operation->output_width;
   task->output_height = operation->output_height;

   task->output_channels_real = operation->output_channels;
   /* Announced DPU output channels: align(Cout, 16) with NO 32 floor
    * (mobilenet_v2 RE 2026-08-23).  The old align-to-32 made the DPU wait
    * for surfaces that never come on Cout=16 convs, so every band task
    * after the first produced nothing (op2).  Matches the vendor's FC task
    * too (mobilenet_v1 t50: 1001 -> 1008).  RKT_OUTALIGN overrides. */
   task->output_channels = align(MAX2(operation->output_channels, 16), 16);
   if (getenv("RKT_OUTALIGN")) {
      unsigned a = atoi(getenv("RKT_OUTALIGN"));
      task->output_channels = align(operation->output_channels, a);
   }
   if (operation->depthwise && operation->input_channels > 32) {
      task->output_channels_real = 32;
      task->output_channels = 32;
   }
   /* TEST (iav RE, 2026-08-21): RK3568 wants the real channel count on
    * depthwise.  The vendor stream carries 32 channels for every 32-channel
    * depthwise layer of mobilenet_v1 (DPU_DATA_CUBE_CHANNEL 0x001f001f),
    * while this doubling made mesa announce 64 to the DPU, which then waited
    * for data that never arrived and stalled right after CNA. */
   if (operation->depthwise && false) {
      if (task->output_channels_real <= 32)
         task->output_channels *= 2;
      task->output_channels = align(task->output_channels, 64);
   }

   task->output_zero_point = operation->output_zero_point;
   task->output_scale = operation->output_scale;

   if (task->input_channels_real == 1 &&
       (task->output_channels_real > 1 ||
        (operation->addition_input || operation->add_tensor != -1))) {
      task->input_width = MAX2(task->input_width, FEATURE_ATOMIC_SIZE);
      task->input_line_stride =
         MAX2(calc_line_stride(operation->input_width) / FEATURE_ATOMIC_SIZE,
              FEATURE_ATOMIC_SIZE);

      if (operation->input_channels == 32 && operation->input_width == 80) {
         task->input_line_stride *= 4;
         task->input_surface_stride = (float)task->input_line_stride *
                                      (((float)task->input_height / 4) - 1);
      } else
         task->input_surface_stride =
            (float)task->input_line_stride * (((float)task->input_height) - 1);
   } else if (operation->input_channels == 3) {
      /* ARGB input: packed RGB rows aligned to 8 bytes; strides are in
       * 8-byte units (vendor probe-RGB: W=28 -> line 11, surf 784). */
      task->input_line_stride =
         DIV_ROUND_UP(operation->input_width * 3, 8);
      task->input_surface_stride =
         operation->input_width * operation->input_height;
   } else {
      /* RK3568: strides are in 8-byte units; the numbers coincide with the
       * old 16-byte-cell reading (W and W*H) for the feature layout. */
      task->input_line_stride =
         calc_line_stride(operation->input_width) / FEATURE_ATOMIC_SIZE;
      task->input_surface_stride =
         task->input_line_stride * task->input_height;
   }

   if (task->input_width == 8 &&
       (operation->addition_input || operation->add_tensor != -1)) {
      task->input_line_stride /= 2;
      task->input_surface_stride = 112;
   }

   int output_line_stride = calc_line_stride(operation->output_width);
   task->output_surface_stride = output_line_stride * task->output_height;
   task->output_surface_stride /= FEATURE_ATOMIC_SIZE;

   if (task->input_channels_real == 1)
      task->input_data_entries = task->input_width * task->input_height;
   else if (task->input_channels_real == 3)
      /* ARGB: entries = ceil(W*8/32) rounded up to even (vendor: 28->8,
       * 224->56). */
      task->input_data_entries =
         align(DIV_ROUND_UP(task->input_width * 8, 32), 2);
   else
      /* RK3568: entries = W * ceil(C/8) / 4 (8-ch atomics, 32-byte entries);
       * matches the vendor stream for every mobilenet_v1 task. */
      task->input_data_entries = DIV_ROUND_UP(
         task->input_width * DIV_ROUND_UP(task->input_channels, 8), 4);

   task->weights_width = operation->weights_width;
   task->weights_height = operation->weights_height;
   task->weights_zero_point = operation->weights_zero_point;
   task->weights_scale = operation->weights_scale;

   if (operation->depthwise)
      task->weights_kernels = 1;
   else if (operation->input_width == 1 && operation->input_height == 1)
      /* FC-shaped: the vendor keeps the exact kernel count (1001). */
      task->weights_kernels = operation->output_channels;
   else
      task->weights_kernels = align(operation->output_channels, 2);

   /* RK3568 (vendor librknnrt 1.5.2, RE 2026-08-20): DPU SURFACE_ADD is the
    * output surface size in 16-byte cells = Wout * Hout, i.e. equal to
    * output_surface_stride.  The previous *2 was an RK3588-era magic
    * multiplier (elements-vs-cells mismatch). */
   task->surfaces_per_row = task->output_width * task->output_height;
   if (operation->depthwise)
      task->surfaces_per_row *= 2;
}

void
rkt_split_tasks(struct rkt_ml_subgraph *subgraph,
                struct rkt_operation *operation)
{
   /* Function mostly taken from NVDLA */
   unsigned entries_per_slice = calc_entries_per_slice(operation);
   unsigned input_banks_required = calc_input_banks(operation);
   unsigned weights_banks_required = calc_weights_banks(operation);
   unsigned available_weights_banks = weights_banks_required;
   unsigned available_input_banks = CBUF_BANKS - weights_banks_required;
   unsigned pad_top;
   unsigned pad_bottom;
   unsigned pad_left;
   unsigned pad_right;

   pad_top = operation->padding_top;
   pad_bottom = operation->padding_bottom;
   pad_left = operation->padding_left;
   pad_right = operation->padding_right;

   if (weights_banks_required + 1 < CBUF_BANKS) {
      /* Full weights, partial input */
      operation->reuse_weights_cbuf = true;
   } else {
      /* Partial weights, partial input */
      operation->reuse_weights_cbuf = false;
      available_input_banks = 7;
      available_weights_banks = CBUF_BANKS - available_input_banks;
   }

   if (input_banks_required <= available_input_banks) {
      /* Full weights, full input */

      struct split_task task = {0};

      task.num = 0;
      fill_task(subgraph, operation, &task);
      task.input_banks = input_banks_required;
      task.weights_banks = CBUF_BANKS - task.input_banks;
      task.input_height = operation->input_height;

      task.pad_top = pad_top;
      task.pad_bottom = pad_bottom;
      task.pad_left = pad_left;
      task.pad_right = pad_right;

      task.atomic_count = task.output_width * task.output_height;

      util_dynarray_append(&operation->tasks, task);

      if (operation->depthwise && operation->input_channels > 32) {
         unsigned groups = DIV_ROUND_UP(operation->input_channels, 32);
         for (unsigned g = 1; g < groups; g++) {
            struct split_task copy = task;
            copy.channel_group = g;
            copy.num = g;
            util_dynarray_append(&operation->tasks, copy);
         }
      }

      return;
   }

   struct split_task task = {0};
   unsigned available_slices =
      (CBUF_ENTRIES_PER_BANK * available_input_banks) / entries_per_slice;
   /* TEST: RKT_MAXROWS=N caps the band height -- forces more bands to probe
    * multi-task pipelines. */
   if (getenv("RKT_MAXROWS"))
      available_slices = MIN2(available_slices,
                              (unsigned)atoi(getenv("RKT_MAXROWS")));

   task.num = 0;
   fill_task(subgraph, operation, &task);
   task.input_banks = available_input_banks;
   task.weights_banks = available_weights_banks;

   task.top_slice = 0;
   task.bottom_slice = available_slices - 1;

   task.pad_top = pad_top;
   task.pad_left = pad_left;
   task.pad_right = pad_right;

   util_dynarray_append(&operation->tasks, task);

   for (unsigned slice = operation->weights_height - pad_top - 1;
        slice < operation->input_height;) {
      memset(&task, 0, sizeof(task));

      struct split_task *prev_task = util_dynarray_element(
         &operation->tasks, struct split_task,
         util_dynarray_num_elements(&operation->tasks, struct split_task) - 1);

      while (slice <= prev_task->bottom_slice) {
         slice += operation->stride;
      }
      if (slice > prev_task->bottom_slice) {
         slice -= operation->stride;
      }

      task.num = util_dynarray_num_elements(&operation->tasks, struct split_task);
      fill_task(subgraph, operation, &task);
      task.top_slice = MIN2(slice, prev_task->bottom_slice) -
                       (operation->weights_height - 1) + operation->stride;
      task.bottom_slice = task.top_slice + available_slices - 1;
      task.pad_left = pad_left;
      task.pad_right = pad_right;

      // check if current task is the last one
      if (task.bottom_slice >= operation->input_height - 1) {
         task.bottom_slice = operation->input_height - 1;
         task.pad_bottom = pad_bottom;
         util_dynarray_append(&operation->tasks, task);
         break;
      }

      slice = task.top_slice + operation->weights_height - 1;
      util_dynarray_append(&operation->tasks, task);
   }

   struct split_task *last_task = util_dynarray_element(
      &operation->tasks, struct split_task,
      util_dynarray_num_elements(&operation->tasks, struct split_task) - 1);
   if (last_task->top_slice >= operation->input_height ||
       last_task->bottom_slice >= (operation->input_height + pad_bottom)) {
      (void)util_dynarray_pop_ptr(&operation->tasks, struct split_task);
   }

   // determine overlap slices between 2 split chunks
   for (int i = 1;
        i < util_dynarray_num_elements(&operation->tasks, struct split_task);
        i++) {
      struct split_task *prev_task =
         util_dynarray_element(&operation->tasks, struct split_task, i - 1);
      struct split_task *cur_task =
         util_dynarray_element(&operation->tasks, struct split_task, i);

      if (prev_task->bottom_slice >= cur_task->top_slice) {
         cur_task->num_overlap_slices =
            prev_task->bottom_slice - cur_task->top_slice + 1;
         prev_task->num_retain_slices = cur_task->num_overlap_slices;
      } else {
         cur_task->num_overlap_slices = 0;
         prev_task->num_retain_slices = 0;
      }
   }

   unsigned output_height_processed = 0;
   for (int i = 0;
        i < util_dynarray_num_elements(&operation->tasks, struct split_task);
        i++) {
      struct split_task *cur_task =
         util_dynarray_element(&operation->tasks, struct split_task, i);

      unsigned slice = cur_task->top_slice + (operation->weights_height - 1) -
                       cur_task->pad_top;

      while (slice <= cur_task->bottom_slice + cur_task->pad_bottom) {
         slice += operation->stride;
         cur_task->convolutions++;
      }

      cur_task->bottom_slice =
         MIN2(cur_task->bottom_slice, operation->input_height - 1);

      cur_task->input_height = cur_task->bottom_slice - cur_task->top_slice + 1;

      cur_task->output_width = (cur_task->input_width + cur_task->pad_left +
                                cur_task->pad_right - operation->weights_width) /
                                  operation->stride +
                               1;
      cur_task->output_height =
         (cur_task->input_height + cur_task->pad_top + cur_task->pad_bottom -
          operation->weights_height) /
            operation->stride +
         1;
      cur_task->atomic_count = cur_task->output_width * cur_task->output_height;

      /* TEST (iav RE, 2026-08-22): trim the band to the rows the convolution
       * actually consumes.  Taking every available CBUF slice leaves a spare
       * input row whenever the slice count is not congruent to the kernel
       * height modulo the stride -- for the 224-wide first layer of
       * mobilenet_v1 mesa fed CNA 128 rows where 63 output rows need
       * (63 - 1) * 2 + 3 = 127.  The vendor stream carries exactly 127. */
      {
         unsigned consumed = (cur_task->output_height - 1) * operation->stride +
                             operation->weights_height;
         unsigned pad = cur_task->pad_top + cur_task->pad_bottom;

         consumed = consumed > pad ? consumed - pad : 0;
         if (consumed > 0 && cur_task->input_height > consumed)
            cur_task->input_height = consumed;
      }

      /* RK3568: planar 8-channel surfaces, 8 bytes per pixel -- a band
       * offset is rows * W * 8 (the per-surface stride register covers the
       * upper channel groups). */
      cur_task->input_offset =
         (operation->input_channels == 3
             ? DIV_ROUND_UP(operation->input_width * 3, 8) * 8
             : operation->input_width * 8) *
         cur_task->top_slice;
      cur_task->output_offset =
         operation->output_width * 8 * output_height_processed;

      cur_task->input_banks = available_input_banks;
      cur_task->weights_banks = available_weights_banks;

      output_height_processed += cur_task->output_height;
   }

   /* Depthwise with C>32: duplicate every band once per 32-channel group;
    * rkt_fill_regcmd shifts the buffer addresses by the group index. */
   if (operation->depthwise && operation->input_channels > 32) {
      unsigned groups = DIV_ROUND_UP(operation->input_channels, 32);
      unsigned bands =
         util_dynarray_num_elements(&operation->tasks, struct split_task);
      for (unsigned g = 1; g < groups; g++) {
         for (unsigned b = 0; b < bands; b++) {
            struct split_task copy = *util_dynarray_element(
               &operation->tasks, struct split_task, b);
            copy.channel_group = g;
            copy.num =
               util_dynarray_num_elements(&operation->tasks, struct split_task);
            util_dynarray_append(&operation->tasks, copy);
         }
      }
   }
}
