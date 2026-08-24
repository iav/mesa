/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include "rkt_regcmd.h"
#include "rkt_ml.h"
#include "rkt_registers.h"

static void
emit_raw(struct util_dynarray *regs, uint32_t target, uint32_t reg,
         uint32_t value)
{
   uint64_t packed_value = 0;
   packed_value = ((uint64_t)target) << 48;
   packed_value |= ((uint64_t)value) << 16;
   packed_value |= (uint64_t)reg;

   util_dynarray_append(regs, packed_value);

   /* TEST (iav RE, 2026-08-21): RKT_DUMP=1 prints the emitted command stream so
    * it can be diffed against the vendor stream extracted from a .rknn. */
   if (getenv("RKT_DUMP"))
      fprintf(stderr, "rkt regcmd %016llx\n", (unsigned long long)packed_value);
}

/* TEST (iav/droid RE session): optional overrides with the vendor-exact
 * values for THE reference conv2d (80x80x16 -> 40x40x128, 5x5 s2) from the
 * MR !42134 byte-exact diff table.  Enabled with RKT_VENDOR_OVR=1.
 */
#include <stdlib.h>
#include <stdio.h>
static const struct { uint32_t reg; uint32_t val; } rkt_test_ovr[] = {
   { 0x1010, 0x00000070 },  /* CNA_CONV_CON2 */
   { 0x1018, 0x00000000 },  /* CNA_CONV_CON4 */
   { 0x1044, 0x00500028 },  /* CNA_CBUF_CON1 */
   { 0x1078, 0x00171c07 },  /* CNA_DMA_CON0 */
   /* 0x107c, 0x1080, 0x40c0 removed: now produced by proper formulas in
    * rkt_task.c (line_stride=Win, surf_stride=Win*Hin, surf_add=Wout*Hout). */
};

static void
emit(struct util_dynarray *regs, uint32_t reg, uint32_t value)
{
   uint32_t target = rkt_get_target(reg) + 0x1;
   if (getenv("RKT_VENDOR_OVR")) {
      for (unsigned i = 0; i < ARRAY_SIZE(rkt_test_ovr); i++) {
         if (rkt_test_ovr[i].reg == reg) {
            value = rkt_test_ovr[i].val;
            break;
         }
      }
   }
   emit_raw(regs, target, reg, value);
}

#define EMIT(offset, value) emit(regs, offset, value);

static void
fill_first_regcmd(struct rkt_ml_subgraph *subgraph,
                  const struct rkt_operation *operation,
                  struct util_dynarray *regs, unsigned task_num)
{
   struct split_task *task =
      util_dynarray_element(&operation->tasks, struct split_task, task_num);
   unsigned num_tasks =
      util_dynarray_num_elements(&operation->tasks, struct split_task);
   unsigned output_zero_point = task->output_zero_point;
   unsigned weights_zero_point = task->weights_zero_point;
   unsigned offset = output_zero_point - 0x80;

   uint32_t con0 = CNA_CBUF_CON0_WEIGHT_BANK(task->weights_banks) |
                   CNA_CBUF_CON0_DATA_BANK(task->input_banks);
   if (task_num > 0 && operation->reuse_weights_cbuf) {
      /* A new depthwise 32-channel group brings new weights -- no reuse on
       * the first task of each group (vendor t8: CBUF_CON0 without the
       * reuse bit). */
      struct split_task *prev = util_dynarray_element(
         &operation->tasks, struct split_task, task_num - 1);
      /* TEST (iav RE 2026-08-23): RKT_NO_WREUSE=1 disables the CBUF weight
       * reuse bit on follow-up band tasks — probing the band-2 constant
       * output on small-weight convs (mobilenet_v2 op2, 512 B). */
      if (prev->channel_group == task->channel_group &&
          !getenv("RKT_NO_WREUSE"))
         con0 |= CNA_CBUF_CON0_WEIGHT_REUSE(1);
   }

   /* BSP-order S_POINTER wakes for ALL sub-units MUST come FIRST in the
    * regcmd (verified by BSP YOLOv5s side-by-side: slots 0-5 of BSP's
    * regcmd are S_POINTER wakes for CNA, CMAC, ACCU, then CBUF_CON0,
    * then DPU_S_POINTER, DPU_RDMA_S_POINTER). */
   EMIT(REG_CNA_S_POINTER, CNA_S_POINTER_POINTER_PP_MODE(1) |
                              CNA_S_POINTER_EXECUTER_PP_EN(1) |
                              CNA_S_POINTER_POINTER_PP_EN(1));
   /* CMAC (formerly "mystery" 0x400 routing target — added to XML in this patch series). */
   emit_raw(regs, 0x401, 0x2004, 0xe);
   EMIT(REG_CORE_S_POINTER, CORE_S_POINTER_POINTER_PP_MODE(1) |
                              CORE_S_POINTER_EXECUTER_PP_EN(1) |
                              CORE_S_POINTER_POINTER_PP_EN(1));

   /* Vendor slot order (probe-D2/mobilenet_v1 streams): CNA S_PTR, CMAC
    * S_PTR, CORE S_PTR, then CBUF_CON0, THEN the DPU/DPU_RDMA wakes. */
   EMIT(REG_CNA_CBUF_CON0, con0);

   EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) |
                              DPU_S_POINTER_EXECUTER_PP_EN(1) |
                              DPU_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_DPU_RDMA_RDMA_S_POINTER,
        DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |
           DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |
           DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));

   EMIT(REG_CNA_CBUF_CON0, con0);

   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_CTRL, 0);

   uint32_t con1 = 0x0;
   if (task->input_channels_real == 1) {
      con1 |= CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) |
              CNA_CONV_CON1_ARGB_IN(8);
   }

   /* ARGB (Cin=3) first-layer mode: packed RGB input (vendor: 0xa000). */
   if (task->input_channels_real == 3)
      con1 |= CNA_CONV_CON1_ARGB_IN(10);

   if (operation->depthwise)
      con1 |= CNA_CONV_CON1_CONV_MODE(3);

   /* Single CONV_CON1 emit (removed duplicate that mesa originally had). */
   EMIT(REG_CNA_CONV_CON1, con1);
   /* CONV_CON2: KERNEL_GROUP per TRM (page 418) = (weights_kernels / 32 - 1)
    * for int8 mode (32 kernels per group). FEATURE_GRAINS per TRM formula
    * = y_stride + weight_height + 1 (suggested by TRM). The old "+50" value
    * was a hack — TRM-correct formula tested 2026-05-21. */
   /* TEST (iav RE, 2026-08-21): the vendor leaves KERNEL_GROUP at zero in all
    * 51 tasks of mobilenet_v1, including layers with 512 and 1001 kernels.
    * Mesa's kernels/32 - 1 underflows to 0xff on depthwise, where kernels is
    * 1 -- and depthwise is exactly what stalls after CNA. */
   /* FC-shaped (1x1 spatial input): the vendor uses FEATURE_GRAINS=1. */
   /* FEATURE_GRAINS grows on narrow outputs (mobilenet_v2 RE 2026-08-23):
    * with the plain stride+kh value a 7x7 conv starts the MAC before the
    * first weight group is in CBUF and oc0-15 read garbage (op49, run-to-
    * run unstable).  The vendor's 51 mobilenet_v1 tasks all match
    *   stride_y + kh + stride_y * ((Wout < 28) + (Wout < 14))
    * (1x1: 2/3/4 for W>=28/14/7; dw s1: 4/5/6; dw s2: 5/7/9), except the
    * Wout==1 cases (FC=1, avgpool=kh) which keep their special values.
    * RKT_GRAINS overrides for probing. */
   unsigned grains;
   if (task->input_width == 1 && task->input_height == 1)
      grains = 1;
   else {
      grains = task->stride_y + task->weights_height;
      if (task->output_width > 1 && !getenv("RKT_GRAINS_OLD"))
         grains += task->stride_y * ((task->output_width < 28 ? 1 : 0) +
                                     (task->output_width < 14 ? 1 : 0));
   }
   if (getenv("RKT_GRAINS"))
      grains = atoi(getenv("RKT_GRAINS"));
   EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(grains));
   EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_X_STRIDE(task->stride_x) |
                              CNA_CONV_CON3_CONV_Y_STRIDE(task->stride_y));
   /* CONV_CON4: RGB_BYTELENGTH per TRM (page 418). Required to be non-zero
    * to start CNA pipeline (verified by experiment on RK3568). For ARGB
    * mode (3-channel RGB) this is the byte length of the input image.
    * For non-ARGB modes the value isn't used by the conv path but the
    * register MUST be non-zero or CNA's OP_ENABLE entry is rejected.
    * Use input_width * input_height * input_channels as a safe per-task
    * input byte count. */
   /* TEST (iav RE, 2026-08-21): the vendor writes RGB_BYTELENGTH only for the
    * ARGB first layer and leaves this at zero everywhere else. */
   /* ARGB: the vendor writes W * H * 9 here for the RGB first layer. */
   EMIT(REG_CNA_CONV_CON4,
        task->input_channels_real == 3
           ? CNA_CONV_CON4_RGB_BYTELENGTH(task->input_width *
                                          task->input_height * 9)
           : (con1 ? CNA_CONV_CON4_RGB_BYTELENGTH(task->input_width *
                                                  task->input_height *
                                                  task->input_channels)
                   : 0));
   EMIT(REG_CNA_DATA_SIZE0,
        CNA_DATA_SIZE0_DATAIN_WIDTH(task->input_width) |
           CNA_DATA_SIZE0_DATAIN_HEIGHT(task->input_height));

   EMIT(REG_CNA_DATA_SIZE1,
        CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(task->input_channels_real - 1) |
           CNA_DATA_SIZE1_DATAIN_CHANNEL(task->input_channels));

   EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(task->output_width));
   EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(task->atomic_count));
   EMIT(REG_CNA_WEIGHT_SIZE0, task->weights_width * task->weights_height *
                                 task->input_channels * task->weights_kernels);
   EMIT(REG_CNA_WEIGHT_SIZE1,
        task->weights_width * task->weights_height * task->input_channels);
   EMIT(REG_CNA_WEIGHT_SIZE2,
        CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(task->weights_width) |
           CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(task->weights_height) |
           CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(task->weights_kernels));

   EMIT(REG_CNA_CBUF_CON0, con0);

   /* TEST (iav RE, 2026-08-21): the upper half of CBUF_CON1 carries the input
    * width in the vendor stream (0x00500028 for the 80-wide reference conv). */
   emit_raw(regs, CNA | 0x1, REG_CNA_CBUF_CON1,
            ((task->input_channels_real == 3 ? align(task->input_width, 32)
                                             : task->input_width)
             << 16) |
               (getenv("RKT_ENTV")
                   ? task->input_width * task->input_channels / 32
                   : task->input_data_entries));

   if (task->input_channels_real == 3) {
      /* ARGB CVT: (x - 128) * 16384 >> 14 = x - 128 -- shift the raw uint8
       * input into the signed domain the rest of the pipeline expects; the
       * alpha lane gets scale 1.  (0xe38e1 = truncate 14 on all three RGB
       * lanes, vendor-exact.) */
      emit_raw(regs, CNA | 0x1, REG_CNA_CVT_CON0, 0x000e38e1);
      EMIT(REG_CNA_CVT_CON1,
           CNA_CVT_CON1_CVT_SCALE0(16384) | CNA_CVT_CON1_CVT_OFFSET0(0xff80));
      EMIT(REG_CNA_CVT_CON2,
           CNA_CVT_CON2_CVT_SCALE1(16384) | CNA_CVT_CON2_CVT_OFFSET1(0xff80));
      EMIT(REG_CNA_CVT_CON3,
           CNA_CVT_CON3_CVT_SCALE2(16384) | CNA_CVT_CON3_CVT_OFFSET2(0xff80));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
   } else if (task->input_channels_real == 1) {
      unsigned truncate = 14;
      unsigned scale = 16384;
      unsigned offset = 65408;

      if (operation->addition_input || operation->add_tensor != -1) {
         truncate = 15;
         scale = 32388;
      }

      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_TRUNCATE_3(truncate) |
                                CNA_CVT_CON0_CVT_TRUNCATE_2(truncate) |
                                CNA_CVT_CON0_CVT_TRUNCATE_1(truncate) |
                                CNA_CVT_CON0_CVT_TRUNCATE_0(truncate));
      EMIT(REG_CNA_CVT_CON1,
           CNA_CVT_CON1_CVT_SCALE0(scale) | CNA_CVT_CON1_CVT_OFFSET0(offset));
      EMIT(REG_CNA_CVT_CON2,
           CNA_CVT_CON2_CVT_SCALE1(scale) | CNA_CVT_CON2_CVT_OFFSET1(offset));
      EMIT(REG_CNA_CVT_CON3,
           CNA_CVT_CON3_CVT_SCALE2(scale) | CNA_CVT_CON3_CVT_OFFSET2(offset));
      EMIT(REG_CNA_CVT_CON4,
           CNA_CVT_CON4_CVT_SCALE3(scale) | CNA_CVT_CON4_CVT_OFFSET3(offset));
   } else {
      /* TEST (iav RE, 2026-08-21): the vendor writes 0xa here for all 42
       * non-ARGB tasks of mobilenet_v1 -- DATA_SIGN and CVT_TYPE set, but
       * CVT_BYPASS clear.  The converter stays in the path. */
      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) |
                                CNA_CVT_CON0_CVT_TYPE(1));
      EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
      EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
      EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
   }

   EMIT(REG_CNA_FC_CON0, 0);
   EMIT(REG_CNA_FC_CON1, 0);
   /* TEST (iav RE, 2026-08-21): on RK3568 the pad value lives in the upper half
    * of PAD_CON0, not in the separate PAD_CON1 register registers.xml lists --
    * the vendor never writes 0x1184 at all.  Its PAD_CON0 is 0xff800000 /
    * 0xff800011 for every layer whose input zero point is 0 (that is -128 once
    * the uint8 tensor is read as int8) and 0x00000000 for the network input,
    * whose zero point is 128.  So the value is input_zero_point - 0x80, which
    * is exactly what mesa already computed for PAD_CON1. */
   EMIT(REG_CNA_PAD_CON0,
        ((uint32_t)((task->input_zero_point - 0x80) & 0xffff) << 16) |
           CNA_PAD_CON0_PAD_LEFT(task->pad_left) |
           CNA_PAD_CON0_PAD_TOP(task->pad_top));
   EMIT(REG_CNA_FEATURE_DATA_ADDR,
        rkt_get_tensor(subgraph, operation->input_index)->phys_addr +
           task->input_offset +
           task->channel_group *
              rkt_surf_px(operation->input_width * operation->input_height) *
              32);
   EMIT(REG_CNA_FC_CON2, 0);
   /* DMA_CON0: FETCH_PIXEL_LEN (bits 15:8) is the per-surface feature fetch
    * length. Mesa previously omitted this field, leaving it 0 which causes
    * CNA to fetch zero feature data per surface on RK3568. Set to input
    * width as a safe default — matches per-row fetch count. */
   /* TEST (iav RE, 2026-08-21): the vendor writes the same 0x00171c07 here in
    * all 44 convolution tasks of mobilenet_v1 and all 23 of resnet18, across
    * every geometry -- FETCH_PIXEL_LEN is a constant 28, not input_width, and
    * both burst lengths are 7, not 15. */
   /* RK3568 (vendor librknnrt 1.5.2, RE 2026-08-22): write the vendor value
    * verbatim.  Bit 20 is RESERVED in the XML, so assembling this register
    * from named fields silently drops it (0x071c07 instead of 0x171c07),
    * and without bit 20 the MAC array never sees real data: the output of
    * every regular convolution is an input-independent constant. */
   EMIT(REG_CNA_DMA_CON0, 0x00171c07);
   EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(task->input_line_stride));
   EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(task->input_surface_stride));

   if (task->input_width == 1 && task->input_height == 1) {
      /* FC-shaped: the CNA DMA reads the 1x1xC input as (C/8) x 1 x 8
       * (vendor t50: 128 x 1, channel 8 for C=1024). */
      EMIT(REG_CNA_FC_DATA_SIZE0,
           CNA_FC_DATA_SIZE0_DMA_WIDTH(task->input_channels / 8) |
              CNA_FC_DATA_SIZE0_DMA_HEIGHT(1));
      EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(8));
   } else {
      EMIT(REG_CNA_FC_DATA_SIZE0,
           CNA_FC_DATA_SIZE0_DMA_WIDTH(operation->input_width) |
              CNA_FC_DATA_SIZE0_DMA_HEIGHT(task->input_height));

      EMIT(REG_CNA_FC_DATA_SIZE1,
           CNA_FC_DATA_SIZE1_DMA_CHANNEL(task->input_channels));
   }
   /* RK3568 DCOMP layout (per BSP regcmd capture 2026-05-22):
    *   0x1110..0x112c = DCOMP_ADDR0..7  (8 weight chunk base pointers)
    *   0x1130..0x114c = DCOMP_AMOUNT0..7 (8 weight chunk byte amounts)
    * RK3588 mesa has 1 ADDR + 16 AMOUNTs at 0x1140+. Use emit_raw for
    * RK3568 BSP-correct layout: emit ADDR0 with weight base, ADDR1-7 = 0,
    * AMOUNT0-7 = 0 (decompression disabled). */
   EMIT(REG_CNA_DCOMP_CTRL, 0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   {
      uint32_t weight_pa = rkt_resource(operation->weights)->phys_addr +
                           task->channel_group * task->weights_width *
                              task->weights_height * 32;
      emit_raw(regs, CNA | 0x1, 0x1110, weight_pa);  /* DCOMP_ADDR0 */
      emit_raw(regs, CNA | 0x1, 0x1114, 0);          /* DCOMP_ADDR1 */
      emit_raw(regs, CNA | 0x1, 0x1118, 0);          /* DCOMP_ADDR2 */
      emit_raw(regs, CNA | 0x1, 0x111c, 0);          /* DCOMP_ADDR3 */
      emit_raw(regs, CNA | 0x1, 0x1120, 0);          /* DCOMP_ADDR4 */
      emit_raw(regs, CNA | 0x1, 0x1124, 0);          /* DCOMP_ADDR5 */
      emit_raw(regs, CNA | 0x1, 0x1128, 0);          /* DCOMP_ADDR6 */
      emit_raw(regs, CNA | 0x1, 0x112c, 0);          /* DCOMP_ADDR7 */
      emit_raw(regs, CNA | 0x1, 0x1130, 0);          /* DCOMP_AMOUNT0 */
      emit_raw(regs, CNA | 0x1, 0x1134, 0);          /* DCOMP_AMOUNT1 */
      emit_raw(regs, CNA | 0x1, 0x1138, 0);          /* DCOMP_AMOUNT2 */
      emit_raw(regs, CNA | 0x1, 0x113c, 0);          /* DCOMP_AMOUNT3 */
   if (getenv("RKT_STRICT")) {
      unsigned r;
      for (r = 0x1210; r <= 0x1230; r += 4)
         emit_raw(regs, CNA | 0x1, r, 0);
   }
      emit_raw(regs, CNA | 0x1, 0x1140, 0);          /* DCOMP_AMOUNT4 */
      emit_raw(regs, CNA | 0x1, 0x1144, 0);          /* DCOMP_AMOUNT5 */
      emit_raw(regs, CNA | 0x1, 0x1148, 0);          /* DCOMP_AMOUNT6 */
      emit_raw(regs, CNA | 0x1, 0x114c, 0);          /* DCOMP_AMOUNT7 */
   }

   if (task->input_channels_real == 1) {
      EMIT(REG_CNA_CVT_CON5, 65535);
   } else {
      EMIT(REG_CNA_CVT_CON5, 0);
   }

   int32_t pad_con1;
   if (task->weights_width >= 3 && task->input_zero_point == 0x0)
      pad_con1 = 0xffff8080;
   else
      pad_con1 = task->input_zero_point - 0x80;

   if (operation->addition_input || operation->add_tensor != -1)
      pad_con1 = 0xffffff80;

   if (operation->depthwise && task->input_zero_point == 0x8b)
      pad_con1 = 0x0b0b;

   /* The vendor writes no PAD_CON1; the pad value went into PAD_CON0 above. */
   (void)pad_con1;

   uint32_t misc_cfg = CORE_MISC_CFG_QD_EN(1);
   if (operation->depthwise)
      misc_cfg |= CORE_MISC_CFG_DW_EN(1);

   /* TEST (iav RE, 2026-08-21): on RK3568 the CORE block map is one slot lower
    * than registers.xml claims -- the xml has a spurious MAC_GATING at 0x300c
    * which shifts everything after it.  The vendor command stream (all 44
    * convolution tasks of mobilenet_v1) writes:
    *   0x300c = MISC_CFG      (0, or DW_EN=2 for depthwise; QD_EN never set)
    *   0x3010 = DATAOUT_SIZE_0 ((height-1) << 16 | (width-1))
    *   0x3014 = DATAOUT_SIZE_1 (channels - 1)
    *   0x302c = 0             (what the xml calls 0x3030)
    * CLIP_TRUNCATE (real 0x3018) is not written at all.
    * With the xml offsets mesa told CORE its output was 3 wide and 1 high,
    * so CORE produced nothing and the DPU interrupt never arrived. */
   /* TEST (iav RE, 2026-08-21): CMAC MISC_CFG sits at 0x200c, the same slot
    * within its block as CORE's at 0x300c.  The vendor sets QD_EN there for
    * every task and adds DW_EN for depthwise; mesa never wrote it at all. */
   emit_raw(regs, 0x401, 0x200c, operation->depthwise ? 0x3 : 0x1);
   emit_raw(regs, CORE | 0x1, 0x300c, operation->depthwise ? 0x2 : 0x0);
   emit_raw(regs, CORE | 0x1, 0x3010,
            ((task->output_height - 1) << 16) | (task->output_width - 1));
   emit_raw(regs, CORE | 0x1, 0x3014, task->output_channels - 1);
   emit_raw(regs, CORE | 0x1, 0x302c, 0);

   /* DPU_FEATURE_MODE_CFG per RK3568 TRM page 433:
    * - BURST_LEN: 0=burst4, 1=burst8, 2=burst16.
    * - OUTPUT_MODE: TRM documents 0=PPU and 2=outside, but BSP YOLOv5s uses
    *   value 4 (0x108 = bit 8 burst_len=2 + bit 3 = output_mode=4).
    *   Match BSP empirically — TRM is incomplete on output_mode codes.
    * - CONV_MODE: 0=Direct, 3=Depthwise.
    */
   uint32_t feat_mode_cfg =
      DPU_FEATURE_MODE_CFG_BURST_LEN(2) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(4);
   if (operation->depthwise)
      feat_mode_cfg |= DPU_FEATURE_MODE_CFG_CONV_MODE(3);
   /* FC-shaped: single output pixel with many channels -- the DPU writes
    * align(K,16)/8 consecutive 8-byte surfaces (vendor t50: NONALIGN with
    * SURF_LEN 126 for 1001 kernels). */
   if (task->input_width == 1 && task->input_height == 1)
      feat_mode_cfg |= DPU_FEATURE_MODE_CFG_NONALIGN(1) |
                       DPU_FEATURE_MODE_CFG_SURF_LEN(task->output_channels / 8);

   EMIT(REG_DPU_FEATURE_MODE_CFG, feat_mode_cfg);
   /* RK3568 vendor requant (RE 2026-08-22): 0xe0 = BS_MUL_SHIFT_VALUE_NEG=14,
    * pairs with the shift-14 multiplier stage in BS_MUL_CFG below --
    * only engaged for the per-channel full BS stream. */
   EMIT(REG_DPU_DATA_FORMAT, operation->per_channel ? 0xe0 : 0);
   EMIT(REG_DPU_OFFSET_PEND, 0);
   EMIT(REG_DPU_DST_BASE_ADDR,
        rkt_get_tensor(subgraph, operation->output_index)->phys_addr +
           operation->dst_offset + task->output_offset +
           task->channel_group *
              rkt_surf_px(operation->output_width * operation->output_height) *
              32);
   /* TEST (iav RE, 2026-08-21): the vendor stream carries output_width *
    * output_height / 2 here, half of what mesa emits, on all 44 convolution
    * tasks of mobilenet_v1 (checked against the full output tensor, not the
    * per-task band).  DPU_SURFACE_ADD at 0x40c0 keeps the undivided value. */
   /* DST_SURF_STRIDE is in BYTES: Wout*Hout*8 (one 8-byte pixel per
    * surface row position).  The generated helper shifts by 4, so the old
    * value/2 form was the same number for even Wout*Hout -- but a 1x1
    * output needs 8, which the shifted field cannot express (this is why
    * the vendor writes a raw 8 for the FC and avgpool tasks). */
   emit_raw(regs, DPU | 0x1, REG_DPU_DST_SURF_STRIDE,
            task->output_surface_stride * 8);
   EMIT(REG_DPU_DATA_CUBE_WIDTH,
        DPU_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_DATA_CUBE_HEIGHT,
        DPU_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));
   EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0);
   EMIT(REG_DPU_DATA_CUBE_CHANNEL,
        DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(task->output_channels_real - 1) |
           DPU_DATA_CUBE_CHANNEL_CHANNEL(task->output_channels - 1));
   /* RK3568 vendor requant (RE 2026-08-22): bias, scale multiplier and weight
    * zero point all come per channel from the BRDMA coefficient stream (see
    * rkt_fill_biases).  0x148: BS_ALU_SRC=1 (bias from stream), relu bypass,
    * multiplier stage active.  0xe01: multiplier from stream, shift 14. */
   if (operation->per_channel) {
      /* Full [bias][ow][mul] BS stream (vendor scheme, RE 2026-08-22):
       * BS_CFG 0x148, multiplier from the stream with shift 14 (0xe01),
       * OW source = stream (0x125 conv / 0x36d dw), OW_OP unused. */
      EMIT(REG_DPU_BS_CFG, getenv("RKT_BSCFG")
                              ? (uint32_t)strtoul(getenv("RKT_BSCFG"), NULL, 16)
                              : 0x148);
      EMIT(REG_DPU_BS_ALU_CFG, 0);
      emit_raw(regs, DPU | 0x1, REG_DPU_BS_MUL_CFG, 0xe01);
      EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0);
      emit_raw(regs, DPU | 0x1, REG_DPU_BS_OW_CFG,
               operation->depthwise ? 0x36d : 0x125);
      EMIT(REG_DPU_BS_OW_OP, 0);
   } else {
      EMIT(REG_DPU_BS_CFG, getenv("RKT_BSCFG") ? (uint32_t)strtoul(getenv("RKT_BSCFG"), NULL, 16) : 0x158);
      EMIT(REG_DPU_BS_ALU_CFG, 0);
      EMIT(REG_DPU_BS_MUL_CFG, 0);
      EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0);

      if (operation->depthwise) {
         EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(3) |
                                    DPU_BS_OW_CFG_SIZE_E_1(3) |
                                    DPU_BS_OW_CFG_SIZE_E_0(3));
      } else {
         EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) |
                                    DPU_BS_OW_CFG_SIZE_E_1(1) |
                                    DPU_BS_OW_CFG_SIZE_E_0(1));
      }

      EMIT(REG_DPU_BS_OW_OP, DPU_BS_OW_OP_OW_OP(0x80 - weights_zero_point));
   }

   if (!getenv("RKT_NOWDMA")) {
      EMIT(REG_DPU_WDMA_SIZE_0,
           DPU_WDMA_SIZE_0_CHANNEL_WDMA(task->output_channels - 1));
      EMIT(REG_DPU_WDMA_SIZE_1,
           DPU_WDMA_SIZE_1_HEIGHT_WDMA(task->output_height - 1) |
              DPU_WDMA_SIZE_1_WIDTH_WDMA(task->output_width - 1));
   }
   if (operation->relu) {
      /* Fused relu/relu6 the vendor way (mobilenet_v1 t0: BN_CFG 0x92,
       * RELUX_CMP 6/(si*sw)): the BN stage clamps the accumulator to
       * [0, CMP] before the EW and OUT_CVT stages.  Output saturation
       * alone only implements the lower clamp when the output zero point
       * is 0 -- on models where the relu output is quantized with a
       * nonzero zero point the negative accumulator survived saturation
       * (resnet18 probes, RE 2026-08-23).  CMP is the largest value the
       * u8 output can express, which on (0,6)-quantized tensors equals
       * the vendor's relu6 bound; rounded up so u8 saturation still
       * performs the exact upper clamp. */
      float acc_max = (255 - task->output_zero_point) * task->output_scale /
                      (task->input_scale * task->weights_scale);
      EMIT(REG_DPU_BN_CFG,
           DPU_BN_CFG_BN_RELUX_EN(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
              DPU_BN_CFG_BN_ALU_BYPASS(1));
      EMIT(REG_DPU_BN_RELUX_CMP_VALUE,
           (uint32_t)MIN2(ceilf(acc_max), 2147483520.0f));
   } else {
      EMIT(REG_DPU_BN_CFG,
           DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
              DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
   }
   EMIT(REG_DPU_BN_ALU_CFG, 0);
   EMIT(REG_DPU_BN_MUL_CFG, 0);
   if (!operation->relu)
      EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0);

   if (operation->add_tensor != -1) {
      /* RK3568 fused residual add (vendor resnet18 capture 2026-08-23,
       * tasks 3/5/8/10/13/15/18/20): EW_CFG is the raw vendor word
       * 0x900000d0 and ERDMA_CFG is raw 0x40000000 (neither decodes with
       * the RK3588 bitfields).  The EW RDMA reads the second input from its
       * 8ch-planar surfaces and the EW converter maps it into the
       * accumulator domain:
       *   x2' = (x2_s8 + (0x80 - zp2)) * scale >> shift,
       *   scale / 2^shift = s2 / (si * sw)
       * -- the vendor's task 3 encodes that ratio (366.0) as 23423 >> 6
       * with the same 15-bit mantissa rule OUT_CVT uses.  OUT_CVT then
       * finishes with the usual conv_scale = si*sw/so for the sum. */
      /* The vendor word has EW_RELU_BYPASS (bit 9) clear when the ADD
       * carries a fused RELU (resnet18-style conv+add+RELU blocks);
       * mobilenet_v2's residual adds are linear and set the bypass. */
      emit_raw(regs, DPU | 0x1, REG_DPU_EW_CFG,
               0x900000d0 | (operation->addition_relu ? 0 : 1 << 9));
      EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, operation->addition_offset);

      float ew_scale_f = operation->addition_scale /
                         (task->input_scale * task->weights_scale);
      uint32_t esb = fui(ew_scale_f);
      unsigned eshift = 127 + 14 - (esb >> 23);
      unsigned escale = ((esb >> 9) & 0x7fff) | (1 << 14);
      EMIT(REG_DPU_EW_CVT_SCALE_VALUE,
           DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT(eshift) |
              DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(escale));
      EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0x0);
   } else {
      EMIT(REG_DPU_EW_CFG,
           DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |
              DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) |
              DPU_EW_CFG_EW_BYPASS(1));
      EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
      EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
      EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);
   }

   {
      EMIT(REG_DPU_OUT_CVT_OFFSET, offset);

      float conv_scale =
         (task->input_scale * task->weights_scale) / task->output_scale;
      uint32_t scale_bits = fui(conv_scale);
      /* Taken from
       * https://github.com/pytorch/QNNPACK/blob/master/src/qnnpack/requantization.h#L130
       */
      unsigned shift = 127 + 31 - 32 - (scale_bits >> 23) + 16;

      if (operation->truncate_bits > 0)
         shift--;

      /* 15-bit mantissa with the implicit leading one folded into bit
       * 14 (the "|= 1 << 14" below only fires when the top explicit
       * mantissa bit is clear).  Rounding a mantissa of 0x3fff up to
       * 0x4000 sets that bit from the ROUNDING, so the |= is a no-op
       * and the value is 2^e * 1.0 at exponent e when it should be
       * 2^e * 2.0 -- half the intended scale.  A conv_scale a hair
       * under a power of two hits exactly that (RE 2026-08-24, int8
       * addonly probe: 0.0039062488 came out as 0x4000 >> 23 instead of
       * 0x4000 >> 22).  Carry the round-up into the exponent. */
      unsigned scale = ((scale_bits >> 9) & 0x7fff) + 1;
      if (scale == 0x4000 && ((scale_bits >> 9) & 0x4000) == 0) {
         shift--;
      } else if (scale == 0x8000) {
         scale = 0x4000;
         shift--;
      }
      if (scale < 1 << 14)
         scale |= 1 << 14;

      if (getenv("RKT_MULSH"))
         shift -= atoi(getenv("RKT_MULSH"));
      EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale));
      EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift - 1));
   }

   EMIT(REG_DPU_EW_OP_VALUE_0, 0);
   EMIT(REG_DPU_EW_OP_VALUE_1, 0);
   EMIT(REG_DPU_EW_OP_VALUE_2, 0);
   EMIT(REG_DPU_EW_OP_VALUE_3, 0);
   EMIT(REG_DPU_EW_OP_VALUE_4, 0);
   EMIT(REG_DPU_EW_OP_VALUE_5, 0);
   EMIT(REG_DPU_EW_OP_VALUE_6, 0);
   EMIT(REG_DPU_EW_OP_VALUE_7, 0);
   EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(task->surfaces_per_row));
   emit_raw(regs, DPU | 0x1, 0x40c4, 0);
   EMIT(REG_DPU_LUT_ACCESS_CFG, 0);
   EMIT(REG_DPU_LUT_ACCESS_DATA, 0);
   EMIT(REG_DPU_LUT_CFG, 0);
   EMIT(REG_DPU_LUT_INFO, 0);
   EMIT(REG_DPU_LUT_LE_START, 0);
   EMIT(REG_DPU_LUT_LE_END, 0);
   EMIT(REG_DPU_LUT_LO_START, 0);
   EMIT(REG_DPU_LUT_LO_END, 0);
   EMIT(REG_DPU_LUT_LE_SLOPE_SCALE, 0);
   EMIT(REG_DPU_LUT_LE_SLOPE_SHIFT, 0);
   EMIT(REG_DPU_LUT_LO_SLOPE_SCALE, 0);
   EMIT(REG_DPU_LUT_LO_SLOPE_SHIFT, 0);
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH,
        DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT,
        DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL,
        DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(task->output_channels - 1));

   /* The vendor's fused-add tasks keep SRC_BASE_ADDR at 0 -- the second
    * input goes through the EW RDMA (EW_BASE_ADDR below). */
   EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0);

   /* Bias-only per-tensor stream: DATA_USE=1, 4 bytes per channel.
    * Per-channel full stream: DATA_USE=7, 8 bytes per channel. */
   EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG,
        DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(operation->per_channel ? 7
                                                                      : 1));
   EMIT(REG_DPU_RDMA_RDMA_BS_BASE_ADDR,
        rkt_resource(operation->biases)->phys_addr +
           task->channel_group * 32 * (operation->per_channel ? 8 : 4));
   /* TEST (iav RE, 2026-08-21): DPU_RDMA 0x5024 sits right after BS_BASE_ADDR
    * and is absent from registers.xml, so mesa never writes it.  The vendor
    * stream in every .rknn writes output_channels - 1 there for all 51 tasks
    * of mobilenet_v1 (10 distinct values, all matching).  Without it the bias
    * RDMA has nothing to fetch, which matches our DT_RD being exactly 512
    * bytes (one bias buffer) short of the vendor's. */
   /* 0x5024 is the BS stream length in 8-byte words minus one: 4 bytes
    * per channel for the bias-only stream, 8 for the full per-channel
    * triple (vendor mobilenet task 4: 64 channels, full stream, 0x3f). */
   emit_raw(regs, DPU_RDMA | 0x1, 0x5024,
            task->output_channels * (operation->per_channel ? 8 : 4) / 8 - 1);
   EMIT(REG_DPU_RDMA_RDMA_NRDMA_CFG, 1); /* bit0 = disable, as vendor */
   EMIT(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);

   if (operation->add_tensor != -1) {
      /* Vendor resnet18 add tasks: ERDMA_CFG raw 0x40000000; EW base points
       * straight at the second input's 8ch-planar tensor (+ band offset);
       * 0x503c (EW line stride) = surf stride - 8 and EW_SURF_STRIDE =
       * Wout*Hout*8, both in raw bytes (t3: 0x61f8 / 0x6200). */
      emit_raw(regs, DPU_RDMA | 0x1, REG_DPU_RDMA_RDMA_ERDMA_CFG, 0x40000000);
      EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
           rkt_get_tensor(subgraph, operation->add_tensor)->phys_addr +
              task->output_offset);
      /* The second-input tensor keeps full-height surfaces regardless of
       * banding, so the EW surface stride spans the full tensor height
       * even on a band task (vendor probe-ADDB: 0x6200 on both bands of a
       * 56x56 add). */
      unsigned ew_surf_stride =
         rkt_surf_px(operation->output_width * operation->output_height) * 8;
      emit_raw(regs, DPU_RDMA | 0x1, 0x503c, ew_surf_stride - 8);
      emit_raw(regs, DPU_RDMA | 0x1, REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
               ew_surf_stride);
   } else {
      EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1));
      EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0);
      emit_raw(regs, DPU_RDMA | 0x1, 0x503c, 0);
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0);
   }

   uint32_t rdma_feat_mode_cfg = 0x0;

   {
      /* RK3568 vendor requant (RE 2026-08-22): the vendor runs every conv task
       * with 0x4000/0x4006 here -- BURST_LEN=8, MRDMA_DISABLE clear.  With the
       * three-operand BRDMA stream (DATA_USE=7) the old BURST_LEN=15 |
       * MRDMA_DISABLE combination scrambles which stream halfword lands in
       * which BS operand lane. */
      if (getenv("RKT_FEAT4000"))
         rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(8);
      else
         rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) |
                               DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1);
   }

   if (operation->depthwise)
      rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(3);

   EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feat_mode_cfg);
   EMIT(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);

   /* Banded fused-add tasks (vendor probe-ADDB, 2 bands of a 56x56x128
    * 3x3): SURF_NOTCH = (H_full - H_band) * W * 8 -- the part of each
    * full-height EW surface that lies outside this band, which the EW
    * RDMA skips when hopping to the next surface.  Full-fit tasks get 0
    * (H_band == H_full), matching the vendor's plain and single-task add
    * convolutions. */
   if (operation->add_tensor != -1) {
      /* The part of each EW surface this task does not consume: the rows
       * outside the band plus the surface alignment padding (vendor 7x7
       * full-fit add: notch 24 = (52-49)*8; banded 56x56: 11200). */
      EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH,
           rkt_surf_px(operation->output_width * operation->output_height) *
                 8 -
              task->output_width * task->output_height * 8);
   } else {
      EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0);
   }

   EMIT(REG_DPU_RDMA_RDMA_PAD_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_WEIGHT,
        DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
           DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));

   EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0x0);

   /* Always a real PC_BASE_ADDRESS command word, even for a single-task
    * stream: the cross-operation chain (compile_operation /
    * chain_operations) patches the next stream's address into it, and a
    * bare zero word cannot be patched with |=. */
   EMIT(REG_PC_BASE_ADDRESS, 0);

   EMIT(REG_PC_REGISTER_AMOUNTS, 0);

   /* TRM: before op_en, 64'h0041_xxxx_xxxx_xxxx must be set. */
   util_dynarray_append_typed(regs, uint64_t, 0x0041000000000000);

   /* RK3568 replay bisection (RE 2026-08-22, Test 42): the five units MUST
    * be started atomically with the vendor broadcast write (target 0x81,
    * reg 8, 0x1f) -- starting them one at a time desynchronizes the
    * pipeline and scrambles which BRDMA stream halfword lands in which BS
    * operand lane (41%% of the output bytes of a known-good vendor task go
    * wrong with sequential OP_EN writes, byte-exact with the broadcast). */
   util_dynarray_append_typed(regs, uint64_t, 0x00810000001f0008);
}

void
rkt_fill_regcmd(struct rkt_ml_subgraph *subgraph,
                const struct rkt_operation *operation,
                struct util_dynarray *regs, unsigned task_num)
{
   /*
    * TODO: We should only need to set all the registers on the regcmd for the first
    * task in an operation, but for now set them all to be sure.
    */
   fill_first_regcmd(subgraph, operation, regs, task_num);
}

/* RK3568 PPU max-pool chunk (RE 2026-08-23, vendor resnet18 task 1 /
 * probe-MP): a 29-word PPU + PPU_RDMA register block finished with its
 * own PC tail whose broadcast starts units 5 and 6 (OP_EN 0x60).  The
 * chunk is linked into the PC chain right after the producer's stream;
 * the PPU input is the producer's output surface in memory.  Word order
 * matches the vendor stream exactly. */
#define PPU_TARGET      0x4001ull
#define PPU_RDMA_TARGET 0x8001ull
#define PPU_WORD(tgt, val, reg)                                             \
   util_dynarray_append_typed(                                              \
      regs, uint64_t,                                                       \
      ((uint64_t)(tgt) << 48) | (((uint64_t)(val) & 0xffffffff) << 16) |    \
         (reg))

void
rkt_fill_ppu_regcmd(struct rkt_ml_subgraph *subgraph,
                    const struct rkt_operation *operation,
                    struct util_dynarray *regs)
{
   unsigned in_w = operation->input_width;
   unsigned in_h = operation->input_height;
   unsigned channels = operation->input_channels;
   unsigned out_w = operation->output_width;
   unsigned out_h = operation->output_height;

   /* The PPU derives its window count from the input cube, rounding UP:
    * feed it more rows/columns than the windows consume and it emits an
    * extra output row/column, shearing the destination (mpC, VALID 3x3
    * s2 on 56x56: 28 columns written into a 27-wide cube).  The vendor
    * clips the cube to (out - 1) * stride + kernel - padding (probe
    * MPvalid56: cube 55 of a 56-row map) while the RDMA line/surface
    * strides keep the full-map values. */
   unsigned cube_w = MIN2(in_w, (out_w - 1) * operation->stride +
                                   operation->weights_width -
                                   operation->padding_left -
                                   operation->padding_right);
   unsigned cube_h = MIN2(in_h, (out_h - 1) * operation->stride +
                                   operation->weights_height -
                                   operation->padding_top -
                                   operation->padding_bottom);
   uint32_t src_addr =
      rkt_get_tensor(subgraph, operation->input_index)->phys_addr;
   uint32_t dst_addr =
      rkt_get_tensor(subgraph, operation->output_index)->phys_addr +
      operation->dst_offset;
   unsigned in_surf = rkt_surf_px(in_w * in_h) * 8;
   unsigned out_surf = rkt_surf_px(out_w * out_h) * 8;

   PPU_WORD(PPU_TARGET, 0xe, 0x6004);      /* S_POINTER */
   PPU_WORD(PPU_RDMA_TARGET, 0xe, 0x7004); /* RDMA_S_POINTER */
   PPU_WORD(PPU_TARGET, cube_w - 1, 0x600c);
   PPU_WORD(PPU_TARGET, cube_h - 1, 0x6010);
   PPU_WORD(PPU_TARGET, channels - 1, 0x6014);
   PPU_WORD(PPU_TARGET, out_w - 1, 0x6018);
   PPU_WORD(PPU_TARGET, out_h - 1, 0x601c);
   PPU_WORD(PPU_TARGET, channels - 1, 0x6020);
   /* OPERATION_MODE_CFG: [1:0] method (0 avg / 1 max), [4] the vendor's
    * flying bit (set even though the input goes through PPU_RDMA --
    * RKT_PPU_MODE overrides for experiments). */
   PPU_WORD(PPU_TARGET,
            getenv("RKT_PPU_MODE")
               ? strtol(getenv("RKT_PPU_MODE"), NULL, 0)
               : (operation->pool_avg ? 0x10 : 0x11),
            0x6024);
   PPU_WORD(PPU_TARGET,
            (operation->stride - 1) << 20 | (operation->stride - 1) << 16 |
               (operation->weights_height - 1) << 8 |
               (operation->weights_width - 1),
            0x6034); /* POOLING_KERNEL_CFG */
   /* RECIP_KERNEL = 65536/kernel for average pooling (vendor probe-AP:
    * kernel 7 -> 9362), zero for max. */
   PPU_WORD(PPU_TARGET,
            operation->pool_avg ? 65536 / operation->weights_width : 0,
            0x6038);
   PPU_WORD(PPU_TARGET,
            operation->pool_avg ? 65536 / operation->weights_height : 0,
            0x603c);
   /* POOLING_PADDING_CFG, [3:0] left [7:4] top [11:8] right [15:12]
    * bottom (RE-LOG Test 61, replay bit-mapping): every pooling window
    * must fall inside input+padding on BOTH axes or the PPU starves
    * forever and the whole PC chain wedges -- that was the mp3s2 hang
    * (TFLite SAME pads right/bottom, and we only programmed top/left). */
   PPU_WORD(PPU_TARGET,
            operation->padding_bottom << 12 | operation->padding_right << 8 |
               operation->padding_top << 4 | operation->padding_left,
            0x6040);
   /* Padding value: -128 as 19-bit two's complement -- the minimum int8,
    * neutral for max pooling of the shifted (u8 - 0x80) feature data. */
   PPU_WORD(PPU_TARGET, 0x7ff80, 0x6044);
   PPU_WORD(PPU_TARGET, 0x7ff80, 0x6048);
   PPU_WORD(PPU_TARGET, 0x7ff80, 0x604c);
   PPU_WORD(PPU_TARGET, 0x7ff80, 0x6050);
   PPU_WORD(PPU_TARGET, dst_addr, 0x6070);  /* DST_BASE_ADDR */
   PPU_WORD(PPU_TARGET, out_surf, 0x607c);  /* DST_SURF_STRIDE, bytes */
   PPU_WORD(PPU_TARGET, out_surf, 0x6084);  /* DATA_FORMAT / INDEX_ADD */
   PPU_WORD(PPU_TARGET, 0x3, 0x60dc);       /* MISC_CTRL burst */
   PPU_WORD(PPU_RDMA_TARGET, cube_w - 1, 0x700c);
   PPU_WORD(PPU_RDMA_TARGET, cube_h - 1, 0x7010);
   PPU_WORD(PPU_RDMA_TARGET, channels - 1, 0x7014);
   PPU_WORD(PPU_RDMA_TARGET, 0x1, 0x7018);
   PPU_WORD(PPU_RDMA_TARGET, src_addr, 0x701c); /* SRC_BASE_ADDR */
   PPU_WORD(PPU_RDMA_TARGET, in_w * 8, 0x7024); /* SRC_LINE_STRIDE */
   PPU_WORD(PPU_RDMA_TARGET, in_surf, 0x7028);  /* SRC_SURF_STRIDE */
   PPU_WORD(PPU_RDMA_TARGET, 0, 0x7030);        /* RDMA_DATA_FORMAT */

   /* Experiment knob: explicit per-unit OP_ENABLE words before the tail
    * (the conv units must start via the broadcast, but the PPU pair may
    * need its own). */
   if (getenv("RKT_PPU_UNITEN")) {
      PPU_WORD(PPU_RDMA_TARGET, 0x1, 0x7008);
      PPU_WORD(PPU_TARGET, 0x1, 0x6008);
   }

   /* PC tail: address/amount patched by the cross-operation chain; the
    * broadcast starts PPU + PPU_RDMA (units 5 and 6). */
   EMIT(REG_PC_BASE_ADDRESS, 0);
   EMIT(REG_PC_REGISTER_AMOUNTS, 0);
   util_dynarray_append_typed(regs, uint64_t, 0x0041000000000000);
   util_dynarray_append_typed(regs, uint64_t, 0x0081000000600008);
}
