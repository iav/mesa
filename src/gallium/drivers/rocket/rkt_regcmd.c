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
}

/* TEST (iav/droid RE session): optional overrides with the vendor-exact
 * values for THE reference conv2d (80x80x16 -> 40x40x128, 5x5 s2) from the
 * MR !42134 byte-exact diff table.  Enabled with RKT_VENDOR_OVR=1.
 */
#include <stdlib.h>
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
   if (task_num > 0 && operation->reuse_weights_cbuf)
      con0 |= CNA_CBUF_CON0_WEIGHT_REUSE(1);

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

   /* 2026-05-27: Move DPU + DPU_RDMA S_POINTER wakes here, BEFORE CNA
    * configs (CBUF_CON0, DCOMP, CONV_CON1). BSP YOLOv5s/MobileNetV1 emit
    * all 5 sub-unit S_POINTER wakes CONTIGUOUSLY at slots 0-4, then
    * starts CNA configs. Mesa's previous order interleaved DPU wakes
    * between CNA configs, possibly causing CNA to reject PC broadcasts
    * (wakes not done for all sub-units). Match BSP order. */
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

   if (operation->depthwise)
      con1 |= CNA_CONV_CON1_CONV_MODE(3);

   /* Single CONV_CON1 emit (removed duplicate that mesa originally had). */
   EMIT(REG_CNA_CONV_CON1, con1);
   /* CONV_CON2: KERNEL_GROUP per TRM (page 418) = (weights_kernels / 32 - 1)
    * for int8 mode (32 kernels per group). FEATURE_GRAINS per TRM formula
    * = y_stride + weight_height + 1 (suggested by TRM). The old "+50" value
    * was a hack — TRM-correct formula tested 2026-05-21. */
   EMIT(REG_CNA_CONV_CON2,
        CNA_CONV_CON2_KERNEL_GROUP(task->weights_kernels / 32 - 1) |
        CNA_CONV_CON2_FEATURE_GRAINS(
           task->stride_y + task->weights_height + 1));
   EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_X_STRIDE(task->stride_x) |
                              CNA_CONV_CON3_CONV_Y_STRIDE(task->stride_y));
   /* CONV_CON4: RGB_BYTELENGTH per TRM (page 418). Required to be non-zero
    * to start CNA pipeline (verified by experiment on RK3568). For ARGB
    * mode (3-channel RGB) this is the byte length of the input image.
    * For non-ARGB modes the value isn't used by the conv path but the
    * register MUST be non-zero or CNA's OP_ENABLE entry is rejected.
    * Use input_width * input_height * input_channels as a safe per-task
    * input byte count. */
   EMIT(REG_CNA_CONV_CON4,
        CNA_CONV_CON4_RGB_BYTELENGTH(
           task->input_width * task->input_height * task->input_channels));
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

   EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(task->input_data_entries));

   if (task->input_channels_real == 1) {
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
      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) |
                                CNA_CVT_CON0_CVT_TYPE(1) |
                                CNA_CVT_CON0_CVT_BYPASS(1));
      EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
      EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
      EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
   }

   EMIT(REG_CNA_FC_CON0, 0);
   EMIT(REG_CNA_FC_CON1, 0);
   EMIT(REG_CNA_PAD_CON0, CNA_PAD_CON0_PAD_LEFT(task->pad_left) |
                             CNA_PAD_CON0_PAD_TOP(task->pad_top));
   EMIT(REG_CNA_FEATURE_DATA_ADDR,
        rkt_get_tensor(subgraph, operation->input_index)->phys_addr +
           task->input_offset);
   EMIT(REG_CNA_FC_CON2, 0);
   /* DMA_CON0: FETCH_PIXEL_LEN (bits 15:8) is the per-surface feature fetch
    * length. Mesa previously omitted this field, leaving it 0 which causes
    * CNA to fetch zero feature data per surface on RK3568. Set to input
    * width as a safe default — matches per-row fetch count. */
   EMIT(REG_CNA_DMA_CON0,
        CNA_DMA_CON0_WEIGHT_BURST_LEN(15) |
        CNA_DMA_CON0_FETCH_PIXEL_LEN(task->input_width) |
        CNA_DMA_CON0_DATA_BURST_LEN(15));
   EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(task->input_line_stride));
   EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(task->input_surface_stride));

   EMIT(REG_CNA_FC_DATA_SIZE0,
        CNA_FC_DATA_SIZE0_DMA_WIDTH(operation->input_width) |
           CNA_FC_DATA_SIZE0_DMA_HEIGHT(task->input_height));

   EMIT(REG_CNA_FC_DATA_SIZE1,
        CNA_FC_DATA_SIZE1_DMA_CHANNEL(task->input_channels));
   /* RK3568 DCOMP layout (per BSP regcmd capture 2026-05-22):
    *   0x1110..0x112c = DCOMP_ADDR0..7  (8 weight chunk base pointers)
    *   0x1130..0x114c = DCOMP_AMOUNT0..7 (8 weight chunk byte amounts)
    * RK3588 mesa has 1 ADDR + 16 AMOUNTs at 0x1140+. Use emit_raw for
    * RK3568 BSP-correct layout: emit ADDR0 with weight base, ADDR1-7 = 0,
    * AMOUNT0-7 = 0 (decompression disabled). */
   EMIT(REG_CNA_DCOMP_CTRL, 0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   {
      uint32_t weight_pa = rkt_resource(operation->weights)->phys_addr;
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

   EMIT(REG_CNA_PAD_CON1, pad_con1);

   uint32_t misc_cfg = CORE_MISC_CFG_QD_EN(1);
   if (operation->depthwise)
      misc_cfg |= CORE_MISC_CFG_DW_EN(1);

   EMIT(REG_CORE_MISC_CFG, misc_cfg);
   EMIT(REG_CORE_DATAOUT_SIZE_0,
        CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(task->output_height - 1) |
           CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(task->output_width - 1));
   EMIT(REG_CORE_DATAOUT_SIZE_1,
        CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(15));
   EMIT(REG_CORE_CLIP_TRUNCATE,
        CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(operation->truncate_bits));
   emit_raw(regs, CORE | 0x1, 0x3030, 0);

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

   EMIT(REG_DPU_FEATURE_MODE_CFG, feat_mode_cfg);
   EMIT(REG_DPU_DATA_FORMAT, 0);
   EMIT(REG_DPU_OFFSET_PEND, 0);
   EMIT(REG_DPU_DST_BASE_ADDR,
        rkt_get_tensor(subgraph, operation->output_index)->phys_addr +
           task->output_offset);
   EMIT(REG_DPU_DST_SURF_STRIDE,
        DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(task->output_surface_stride));
   EMIT(REG_DPU_DATA_CUBE_WIDTH,
        DPU_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_DATA_CUBE_HEIGHT,
        DPU_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));
   EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0);
   EMIT(REG_DPU_DATA_CUBE_CHANNEL,
        DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(task->output_channels_real - 1) |
           DPU_DATA_CUBE_CHANNEL_CHANNEL(task->output_channels - 1));
   EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_ALU_ALGO(2) | DPU_BS_CFG_BS_ALU_SRC(1) |
                           DPU_BS_CFG_BS_RELU_BYPASS(1) |
                           DPU_BS_CFG_BS_MUL_BYPASS(1));
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

   EMIT(REG_DPU_WDMA_SIZE_0,
        DPU_WDMA_SIZE_0_CHANNEL_WDMA(task->output_channels - 1));
   EMIT(REG_DPU_WDMA_SIZE_1,
        DPU_WDMA_SIZE_1_HEIGHT_WDMA(task->output_height - 1) |
           DPU_WDMA_SIZE_1_WIDTH_WDMA(task->output_width - 1));
   EMIT(REG_DPU_BN_CFG,
        DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
           DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
   EMIT(REG_DPU_BN_ALU_CFG, 0);
   EMIT(REG_DPU_BN_MUL_CFG, 0);
   EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0);

   if (operation->add_tensor != -1) {
      EMIT(REG_DPU_EW_CFG,
           DPU_EW_CFG_EW_CVT_TYPE(1) | DPU_EW_CFG_EW_DATA_MODE(1) |
              DPU_EW_CFG_EDATA_SIZE(1) | DPU_EW_CFG_EW_ALU_ALGO(2) |
              DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) |
              DPU_EW_CFG_EW_OP_SRC(1));

      /* See http://nvdla.org/hw/v1/ias/precision.html#element-wise */
      EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, operation->addition_offset);

      float add_scale = 0.0;
      if (fabs(operation->addition_scale - 0.090192) < 0.00001) {
         add_scale = 299.671889248;
      } else if (fabs(operation->addition_scale - 0.399250) < 0.00001) {
         add_scale = 1326.499209406;
      } else if (fabs(operation->addition_scale - 0.364902) < 0.00001) {
         add_scale = 780.34375;
      } else if (fabs(operation->addition_scale - 0.422037) < 0.00001) {
         add_scale = 715.5625;
      } else if (fabs(operation->addition_scale - 0.213016) < 0.00001) {
         add_scale = 564.6875;
      } else if (fabs(operation->addition_scale - 0.244231) < 0.00001) {
         add_scale = 499.796875;
      } else if (fabs(operation->addition_scale - 0.283416) < 0.00001) {
         add_scale = 488.203125;
      } else if (fabs(operation->addition_scale - 0.171151) < 0.00001) {
         add_scale = 602.90625;
      } else if (fabs(operation->addition_scale - 0.164588) < 0.00001) {
         add_scale = 271.921875;
      } else if (fabs(operation->addition_scale - 0.204098) < 0.00001) {
         add_scale = 262.90625;
      } else if (fabs(operation->addition_scale - 0.116532) < 0.00001) {
         add_scale = 450.140625;
      } else if (fabs(operation->addition_scale - 0.134499) < 0.00001) {
         add_scale = 212.1953125;
      } else if (fabs(operation->addition_scale - 0.220141) < 0.00001) {
         add_scale = 368.28125;
      } else if (fabs(operation->addition_scale - 0.094560) < 0.00001) {
         add_scale = 416.421875;
      } else if (fabs(operation->addition_scale - 0.093230) < 0.00001) {
         add_scale = 305.421875;
      } else if (fabs(operation->addition_scale - 0.100618) < 0.00001) {
         add_scale = 313.671875;
      } else {
         add_scale = 0.0;
      }

      uint32_t add_scale_bits = fui(add_scale);
      /* Taken from
       * https://github.com/pytorch/QNNPACK/blob/master/src/qnnpack/requantization.h#L130
       */
      unsigned add_shift = 127 + 31 - 32 - (add_scale_bits >> 23) + 16;

      unsigned scale = ((add_scale_bits >> 9) & 0x7fff);
      if (scale < 1 << 14)
         scale |= 1 << 14;

      EMIT(REG_DPU_EW_CVT_SCALE_VALUE,
           DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SHIFT(add_shift - 1) |
              DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(scale));

      EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0x0);

      if (fabs(operation->addition_scale - 0.213016) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0x4);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(25914));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(24));
      } else if (fabs(operation->addition_scale - 0.244231) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0x1);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(28927));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(24));
      } else if (fabs(operation->addition_scale - 0.283416) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0x6);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(26050));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(24));
      } else if (fabs(operation->addition_scale - 0.171151) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0xfffffffd);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(28937));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(24));
      } else if (fabs(operation->addition_scale - 0.164588) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0x1);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(24877));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(23));
      } else if (fabs(operation->addition_scale - 0.204098) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0x0);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(23272));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(23));
      } else if (fabs(operation->addition_scale - 0.116532) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0xfffffff8);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(32292));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(24));
      } else if (fabs(operation->addition_scale - 0.134499) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0xfffffffb);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(24153));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(23));
      } else if (fabs(operation->addition_scale - 0.220141) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0xb);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(27655));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(24));
      } else if (fabs(operation->addition_scale - 0.094560) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0x5);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(20432));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(23));
      } else if (fabs(operation->addition_scale - 0.093230) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0xffffffff);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(25449));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(23));
      } else if (fabs(operation->addition_scale - 0.100618) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, offset);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(16874));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(23));
      } else if (fabs(operation->addition_scale - 0.422037) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0x1);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(22559));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(24));
      } else if (fabs(operation->addition_scale - 0.364902) < 0.00001) {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0x4);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(18589));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(24));
      } else {
         EMIT(REG_DPU_OUT_CVT_OFFSET, 0x6);
         EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(27676));
         EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(25));
      }
   } else {
      EMIT(REG_DPU_EW_CFG,
           DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |
              DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) |
              DPU_EW_CFG_EW_BYPASS(1));
      EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
      EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
      EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);
      EMIT(REG_DPU_OUT_CVT_OFFSET, offset);

      float conv_scale =
         (task->input_scale * task->weights_scale) / task->output_scale;
      // DBG("conv_scale %f\n", conv_scale);
      uint32_t scale_bits = fui(conv_scale);
      /* Taken from
       * https://github.com/pytorch/QNNPACK/blob/master/src/qnnpack/requantization.h#L130
       */
      unsigned shift = 127 + 31 - 32 - (scale_bits >> 23) + 16;

      if (operation->truncate_bits > 0)
         shift--;

      unsigned scale = ((scale_bits >> 9) & 0x7fff) + 1;
      if (scale < 1 << 14)
         scale |= 1 << 14;

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

   if (operation->add_tensor != -1) {
      EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR,
           rkt_get_tensor(subgraph, operation->add_tensor)->phys_addr +
              task->output_offset);
   } else {
      EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0);
   }

   EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG, DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(1));
   EMIT(REG_DPU_RDMA_RDMA_BS_BASE_ADDR,
        rkt_resource(operation->biases)->phys_addr);
   EMIT(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);

   unsigned ew_stride =
      MAX2(operation->output_width * operation->output_height, 12);

   if (operation->add_tensor != -1) {
      EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG,
           DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(1) |
              DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(1));
      unsigned ew_base_offset =
         operation->output_width * operation->output_height * ATOMIC_K_SIZE;
      EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR,
           rkt_get_tensor(subgraph, operation->add_tensor)->phys_addr +
              task->output_offset + ew_base_offset);
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
           DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(ew_stride));
   } else {
      EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1));
      EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0);
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0);
   }

   uint32_t rdma_feat_mode_cfg = 0x0;

   if (operation->add_tensor != -1) {
      rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) |
                            DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(5);
   } else {
      /* TEST (RE 2026-08-21): vendor CNA DMA_CON0 uses burst_len 7 on
       * RK3568 (mesa 15 left CNA DMA incomplete, Test 13); try the same
       * burst limit for DPU_RDMA — bias BRDMA never fetches (DT_RD delta
       * vs vendor = exactly the 512-byte bias buffer). */
      rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(7) |
                            DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1);
   }

   if (operation->depthwise)
      rdma_feat_mode_cfg |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(3);

   EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rdma_feat_mode_cfg);
   EMIT(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);

   unsigned surf_notch =
      ew_stride +
      task->output_width * (operation->output_height - task->output_height);

   if (operation->input_width == 3) {
      surf_notch = 15;
   }

   if (operation->add_tensor != -1) {
      EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH,
           DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(surf_notch));
   } else {
      EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0);
   }

   EMIT(REG_DPU_RDMA_RDMA_PAD_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_WEIGHT,
        DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
           DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));

   if (operation->add_tensor != -1) {
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH,
           DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(surf_notch));
   } else {
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0x0);
   }

   if (num_tasks == 1)
      util_dynarray_append_typed(regs, uint64_t, 0x0);
   else
      EMIT(REG_PC_BASE_ADDRESS, 0);

   EMIT(REG_PC_REGISTER_AMOUNTS, 0);

   /* TRM: before op_en, 64'h0041_xxxx_xxxx_xxxx must be set. */
   util_dynarray_append_typed(regs, uint64_t, 0x0041000000000000);

   /* Individual sub-unit OP_EN writes. The broadcast target=0x81 is
    * RK3588-specific and not supported on RK3568. Direct writes work on both. */
   emit_raw(regs, CNA | 0x1, REG_CNA_OPERATION_ENABLE,
            CNA_OPERATION_ENABLE_OP_EN(1));
   emit_raw(regs, CORE | 0x1, REG_CORE_OPERATION_ENABLE,
            CORE_OPERATION_ENABLE_OP_EN(1));
   emit_raw(regs, DPU_RDMA | 0x1, REG_DPU_RDMA_RDMA_OPERATION_ENABLE,
            DPU_RDMA_RDMA_OPERATION_ENABLE_OP_EN(1));
   emit_raw(regs, DPU | 0x1, REG_DPU_OPERATION_ENABLE,
            DPU_OPERATION_ENABLE_OP_EN(1));
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
