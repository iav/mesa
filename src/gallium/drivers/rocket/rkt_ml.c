/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include "pipe/p_state.h"
#include "util/macros.h"
#include "util/u_dynarray.h"
#include "util/u_inlines.h"

#include <math.h>
#include <xf86drm.h>

#include "drm-uapi/rocket_accel.h"

#include "rkt_coefs.h"
#include "rkt_ml.h"
#include "rkt_regcmd.h"
#include "rkt_task.h"

void
rkt_dump_buffer(const uint8_t *ptr, char *name, int operation_nr,
                int suboperation_nr, int offset, unsigned size)
{
   char buffer[255];

   snprintf(buffer, sizeof(buffer), "mesa-%s-%03u-%03u.bin", name, operation_nr,
            suboperation_nr);

   FILE *f = fopen(buffer, "wb");
   assert(f);
   fwrite(ptr + offset, 1, size, f);
   if (ferror(f)) {
      DBG("Error in writing to file: %s\n", strerror(errno));
   }
   fflush(f);
   fclose(f);
}

static void
create_tensor(struct rkt_ml_subgraph *subgraph, unsigned idx,
              unsigned size)
{
   struct pipe_context *context = subgraph->context;
   struct pipe_resource **tensors = util_dynarray_begin(&subgraph->tensors);

   assert(idx < util_dynarray_num_elements(&subgraph->tensors,
                                           struct pipe_resource *));

   struct pipe_resource *res = tensors[idx];

   if (res != NULL) {
      /* A concatenation output is created at its full size up front;
       * the retargeted producers then "create" it at their own slice
       * size, which only has to fit. */
      assert(size <= pipe_buffer_size(res));
      return;
   }

   res = pipe_buffer_create(context->screen, 0, PIPE_USAGE_DEFAULT, size);
   tensors[idx] = res;
}

struct rkt_resource *
rkt_get_tensor(struct rkt_ml_subgraph *subgraph,
               unsigned idx)
{
   return rkt_resource(
      *util_dynarray_element(&subgraph->tensors, struct pipe_resource *, idx));
}

bool
rkt_is_depthwise(const struct pipe_ml_operation *poperation)
{
   unsigned input_channels = poperation->input_tensors[0]->dims[3];
   unsigned output_channels = poperation->output_tensors[0]->dims[3];

   return poperation->conv.depthwise && input_channels > 1 &&
          output_channels > 1;
}

static unsigned
calc_raw_output_size(struct rkt_operation *operation)
{
   unsigned output_channels_1 =
      DIV_ROUND_UP(operation->output_channels, FEATURE_ATOMIC_SIZE) * 2;
   unsigned output_channels_2 = FEATURE_ATOMIC_SIZE;

   return rkt_surf_px(operation->output_width * operation->output_height) *
          output_channels_1 * output_channels_2;
}

/* Room reserved after an operation's streams for an appended PPU pool
 * chunk (33 words, 128-byte aligned). */
#define POOL_CHUNK_ROOM 384

static void
compile_operation(struct rkt_ml_subgraph *subgraph,
                  struct rkt_operation *operation,
                  struct rkt_operation *pool_op)
{
   struct pipe_context *pcontext = subgraph->context;
   unsigned regcfg_total_size = 0;
   struct util_dynarray *regcfgs;
   struct pipe_transfer *transfer = NULL;
   unsigned num_tasks =
      util_dynarray_num_elements(&operation->tasks, struct split_task);

   /* Pool operations compile as a chunk APPENDED to the previous
    * operation's regcmd BO (pool_op below): the PC cannot fetch a PPU
    * chunk from a foreign BO -- vendor-stream replay wedges with the
    * chunk moved out, runs with it in place (RE-LOG Test 61); regular
    * convolution streams jump across BOs fine. */
   assert(!operation->is_pool);

   regcfgs = calloc(num_tasks, sizeof(struct util_dynarray));

   for (int i = 0; i < num_tasks; i++) {
      regcfgs[i] = UTIL_DYNARRAY_INIT;
      rkt_fill_regcmd(subgraph, operation, &regcfgs[i], i);

      unsigned size =
         util_dynarray_num_elements(&regcfgs[i], uint64_t) * sizeof(uint64_t);
      regcfg_total_size += align(size, 128);
   }

   if (pool_op)
      regcfg_total_size += POOL_CHUNK_ROOM;

   operation->regcmd = pipe_buffer_create(pcontext->screen, 0,
                                          PIPE_USAGE_DEFAULT, regcfg_total_size);
   uint8_t *regcmd =
      pipe_buffer_map(pcontext, operation->regcmd, PIPE_MAP_WRITE, &transfer);

   unsigned regcmd_offset = 0;

   for (int i = 0; i < num_tasks; i++) {
      unsigned size = util_dynarray_num_elements(&regcfgs[i], uint64_t);
      struct split_task *task =
         util_dynarray_element(&operation->tasks, struct split_task, i);

      if (i < num_tasks - 1) {
         /* Patch next address and amount of regs to fetch, positions are relative
          * to end */
         unsigned reg_count = util_dynarray_num_elements(&regcfgs[i], uint64_t);
         uint64_t *next_address_reg =
            util_dynarray_element(&regcfgs[i], uint64_t, reg_count - 4);
         uint64_t *reg_count_reg =
            util_dynarray_element(&regcfgs[i], uint64_t, reg_count - 3);

         uint64_t addr = rkt_resource(operation->regcmd)->phys_addr +
                         regcmd_offset + align(size * sizeof(uint64_t), 128);
         *next_address_reg |= addr << 16;

         /* RK3568: pc_data_amount_scale = 1 -- the next stream's
          * PC_REGISTER_AMOUNTS is simply its word count minus one (the
          * vendor writes amount+3 with amount = words-4; same value).
          * The /2 below was RK3588 (scale = 2). */
         unsigned regs_to_fetch =
            util_dynarray_num_elements(&regcfgs[i + 1], uint64_t) - 1;
         *reg_count_reg |= regs_to_fetch << 16;
      }

      memcpy(regcmd + regcmd_offset, util_dynarray_begin(&regcfgs[i]),
             size * sizeof(uint64_t));
      util_dynarray_fini(&regcfgs[i]);

      task->regcfg_amount = size;
      task->regcfg_addr =
         rkt_resource(operation->regcmd)->phys_addr + regcmd_offset;

      if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS)) {
         static int dump_nr;
         rkt_dump_buffer(regcmd, "regcmd", dump_nr++, i, regcmd_offset,
                         (size + 4) * sizeof(uint64_t));
      }

      regcmd_offset += align(size * sizeof(uint64_t), 128);
   }

   if (pool_op) {
      struct util_dynarray pregs = UTIL_DYNARRAY_INIT;
      struct split_task *ptask =
         util_dynarray_element(&pool_op->tasks, struct split_task, 0);

      rkt_fill_ppu_regcmd(subgraph, pool_op, &pregs);

      unsigned psize = util_dynarray_num_elements(&pregs, uint64_t);
      assert(psize * sizeof(uint64_t) <= POOL_CHUNK_ROOM);
      memcpy(regcmd + regcmd_offset, util_dynarray_begin(&pregs),
             psize * sizeof(uint64_t));
      ptask->regcfg_amount = psize;
      ptask->regcfg_addr =
         rkt_resource(operation->regcmd)->phys_addr + regcmd_offset;
      if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS)) {
         static int pool_dump_nr;
         rkt_dump_buffer(regcmd, "poolcmd", pool_dump_nr++, 0, regcmd_offset,
                         psize * sizeof(uint64_t));
      }
      util_dynarray_fini(&pregs);
   }

   pipe_buffer_unmap(pcontext, transfer);

   for (int i = 0; i < num_tasks; i++) {
      util_dynarray_fini(&regcfgs[i]);
   }

   free(regcfgs);
}

static struct rkt_view *
find_view(struct rkt_ml_subgraph *subgraph, unsigned index)
{
   util_dynarray_foreach (&subgraph->views, struct rkt_view, v)
      if (v->index == index)
         return v;
   return NULL;
}

/* Replace a view input by its source: a channel slice becomes a surface
 * offset into the source tensor, a spatial pad becomes convolution
 * padding.  Views can nest (a slice of a pad, a pad of a slice). */
static void
resolve_views(struct rkt_ml_subgraph *subgraph, struct rkt_operation *op)
{
   struct rkt_view *v;

   while ((v = find_view(subgraph, op->input_index)) != NULL) {
      if (v->is_pad) {
         op->input_width = v->src_width;
         op->input_height = v->src_height;
         op->padding_top += v->pad_top;
         op->padding_bottom += v->pad_bottom;
         op->padding_left += v->pad_left;
         op->padding_right += v->pad_right;
      } else {
         unsigned surf =
            rkt_surf_px(op->input_width * op->input_height) * 8;
         op->src_offset += (v->ch_off / 8) * surf;
         op->src_channels = v->src_channels;
      }
      op->input_index = v->src_index;
   }
}

static void
lower_convolution(struct rkt_ml_subgraph *subgraph,
                  const struct pipe_ml_operation *poperation,
                  struct rkt_operation *operation, unsigned pad_channels)
{
   operation->tasks = UTIL_DYNARRAY_INIT;

   operation->depthwise = rkt_is_depthwise(poperation);
   operation->relu = poperation->conv.relu;
   operation->padding_top = poperation->conv.padding_top;
   operation->padding_bottom = poperation->conv.padding_bottom;
   operation->padding_left = poperation->conv.padding_left;
   operation->padding_right = poperation->conv.padding_right;
   operation->stride = poperation->conv.stride_x;

   /* Axis convention (RE 2026-08-24, layer-rect probes): width is the
    * CONTIGUOUS memory row -- NHWC dims[2] -- and height is dims[1], the
    * axis the band splitter cuts.  This matches the vendor stream (its
    * 98x224 probe announces DATAIN_WIDTH=224).  The old dims[1]-as-width
    * reading was self-consistent only on square maps: the input packer
    * already wrote memory rows, so every non-square map sheared (and the
    * output unpacker even wrote past the user buffer). */
   operation->input_index = poperation->input_tensors[0]->index;
   operation->input_width = poperation->input_tensors[0]->dims[2];
   operation->input_height = poperation->input_tensors[0]->dims[1];
   operation->input_channels = poperation->input_tensors[0]->dims[3];
   operation->input_zero_point = poperation->input_tensors[0]->zero_point;
   operation->input_scale = poperation->input_tensors[0]->scale;
   operation->src_channels = operation->input_channels;
   operation->src_offset = 0;
   resolve_views(subgraph, operation);

   operation->output_index = poperation->output_tensors[0]->index;
   operation->output_width = poperation->output_tensors[0]->dims[2];
   operation->output_height = poperation->output_tensors[0]->dims[1];
   operation->output_channels = poperation->output_tensors[0]->dims[3];
   operation->output_zero_point = poperation->output_tensors[0]->zero_point;
   operation->output_scale = poperation->output_tensors[0]->scale;

   operation->weights_width = poperation->conv.weight_tensor->dims[2];
   operation->weights_height = poperation->conv.weight_tensor->dims[1];
   operation->weights_zero_point = poperation->conv.weight_tensor->zero_point;
   operation->weights_scale = poperation->conv.weight_tensor->scale;

   /* Per-channel weight quantization (the vendor two-stage scheme): the
    * layer-wide OUT_CVT runs off the LARGEST channel scale and the BS
    * mul entries renormalize each channel to it (rkt_fill_biases). */
   if (poperation->conv.weight_tensor->scales != NULL) {
      unsigned nch = operation->depthwise
                        ? poperation->conv.weight_tensor->dims[3]
                        : poperation->conv.weight_tensor->dims[0];
      float s_wmax = 0.0f;
      for (unsigned c = 0; c < nch; c++)
         s_wmax = MAX2(s_wmax, poperation->conv.weight_tensor->scales[c]);
      operation->weights_scale = s_wmax;
      operation->per_channel = true;
   }

   operation->output_channels_pad = pad_channels;
   operation->weights = rkt_fill_weights(subgraph, poperation, pad_channels);
   operation->biases =
      rkt_fill_biases(subgraph, poperation, &operation->truncate_bits);
}

/* Average pooling as a depthwise convolution: every weight is q=255 with
 * scale 1/(fw*fh*255) and zero point 0, so each tap contributes exactly
 * 1/(fw*fh) -- the regular requantization pipeline does the rest.  The
 * packed weight bytes come out as 0x7f, matching the vendor stream. */
static void
lower_pooling(struct rkt_ml_subgraph *subgraph,
              const struct pipe_ml_operation *ppool,
              struct rkt_operation *operation)
{
   unsigned fw = ppool->pooling.filter_width;
   unsigned fh = ppool->pooling.filter_height;
   unsigned channels = ppool->input_tensors[0]->dims[3];
   struct pipe_ml_operation conv = {0};
   struct pipe_tensor weight_tensor = {0};
   struct pipe_tensor bias_tensor = {0};
   uint8_t *wdata = malloc(fw * fh * channels);
   int32_t *bdata = calloc(channels, sizeof(int32_t));

   memset(wdata, 0xff, fw * fh * channels);

   weight_tensor.dims[0] = 1;
   weight_tensor.dims[1] = fh;
   weight_tensor.dims[2] = fw;
   weight_tensor.dims[3] = channels;
   weight_tensor.scale = 1.0f / (fw * fh * 255);
   weight_tensor.zero_point = 0;
   weight_tensor.data = wdata;

   bias_tensor.dims[3] = channels;
   bias_tensor.data = (uint8_t *)bdata;

   conv.type = PIPE_ML_OPERATION_TYPE_CONVOLUTION;
   conv.input_tensors = ppool->input_tensors;
   conv.output_tensors = ppool->output_tensors;
   conv.conv.weight_tensor = &weight_tensor;
   conv.conv.bias_tensor = &bias_tensor;
   conv.conv.stride_x = ppool->pooling.stride_x;
   conv.conv.stride_y = ppool->pooling.stride_y;
   conv.conv.depthwise = true;
   conv.conv.dilation_width_factor = 1;
   conv.conv.dilation_height_factor = 1;

   lower_convolution(subgraph, &conv, operation, 0);

   free(wdata);
   free(bdata);
}

/* Max pooling on the PPU unit: no weights, no requantization -- just the
 * geometry.  The operation compiles to a bare PPU/PPU_RDMA chunk (see
 * rkt_fill_ppu_regcmd) linked into the PC chain after its producer. */
static void
lower_max_pooling(struct rkt_ml_subgraph *subgraph,
                  const struct pipe_ml_operation *ppool,
                  struct rkt_operation *operation)
{
   operation->tasks = UTIL_DYNARRAY_INIT;
   operation->is_pool = true;

   operation->input_index = ppool->input_tensors[0]->index;
   operation->input_width = ppool->input_tensors[0]->dims[2];
   operation->input_height = ppool->input_tensors[0]->dims[1];
   operation->input_channels = ppool->input_tensors[0]->dims[3];
   operation->src_channels = operation->input_channels;
   resolve_views(subgraph, operation);
   operation->input_zero_point = ppool->input_tensors[0]->zero_point;
   operation->input_scale = ppool->input_tensors[0]->scale;

   operation->output_index = ppool->output_tensors[0]->index;
   operation->output_width = ppool->output_tensors[0]->dims[2];
   operation->output_height = ppool->output_tensors[0]->dims[1];
   operation->output_channels = ppool->output_tensors[0]->dims[3];
   operation->output_zero_point = ppool->output_tensors[0]->zero_point;
   operation->output_scale = ppool->output_tensors[0]->scale;

   operation->weights_width = ppool->pooling.filter_width;
   operation->weights_height = ppool->pooling.filter_height;
   operation->stride = ppool->pooling.stride_x;
   operation->padding_top = ppool->pooling.padding_top;
   operation->padding_bottom = ppool->pooling.padding_bottom;
   operation->padding_left = ppool->pooling.padding_left;
   operation->padding_right = ppool->pooling.padding_right;
   operation->pool_avg = ppool->pooling.type == PIPE_ML_POOLING_TYPE_AVG;

   operation->add_tensor = -1;
}

/* A fully-connected layer over a 1x1 spatial input is a 1x1 convolution:
 * the TFLite [out, in] weight matrix is bytewise identical to a 1x1 OHWI
 * conv weight tensor [out, 1, 1, in], only the dims move (teflon
 * right-aligns the 2D matrix into [1, 1, out, in]).  Weights larger than
 * the CBUF stream through the big-FC conv path. */
static void
lower_fully_connected(struct rkt_ml_subgraph *subgraph,
                      const struct pipe_ml_operation *pfcon,
                      struct rkt_operation *operation)
{
   struct pipe_ml_operation conv = {0};
   struct pipe_tensor weight_tensor = *pfcon->fcon.weight_tensor;

   weight_tensor.dims[0] = pfcon->fcon.weight_tensor->dims[2];
   weight_tensor.dims[1] = 1;
   weight_tensor.dims[2] = 1;
   weight_tensor.dims[3] = pfcon->fcon.weight_tensor->dims[3];

   conv.type = PIPE_ML_OPERATION_TYPE_CONVOLUTION;
   conv.input_tensors = pfcon->input_tensors;
   conv.output_tensors = pfcon->output_tensors;
   conv.conv.weight_tensor = &weight_tensor;
   conv.conv.bias_tensor = pfcon->fcon.bias_tensor;
   conv.conv.stride_x = 1;
   conv.conv.stride_y = 1;
   conv.conv.relu = pfcon->fcon.relu;
   conv.conv.pointwise = true;
   conv.conv.dilation_width_factor = 1;
   conv.conv.dilation_height_factor = 1;

   lower_convolution(subgraph, &conv, operation, 0);
}

/* Copy (and requantize) one tensor into a slice of a concatenation
 * output: a 1x1 depthwise convolution with all weights q 255 and scale
 * 1/255 (a per-channel identity), the regular BS pipeline requantizes
 * into the concat output's domain.  Only for C % 32 == 0: a depthwise
 * convolution with fewer channels declares align(C, 32) of them to the
 * DPU cube and writes the padding surfaces too, which would clobber the
 * neighbouring slice of the concat output (RE 2026-08-24,
 * layer-concat-mixed: C=16 copy at offset 0 corrupts the C=32 slice
 * behind it).  Everything else falls back to the pointwise formulation
 * with an identity weight matrix.  Used when the input cannot just be
 * written in place by its producer (it comes from outside the
 * partition, has other consumers, or its producer is a PPU pool chunk,
 * which has no requant stage).
 *
 * force_pointwise: a copy that will HOST a fused addition must be
 * pointwise -- with a depthwise host the EW RDMA runs in the depthwise
 * feature mode (RDMA_FEATURE_MODE_CFG 0x7816) and reads the second
 * operand with the wrong surface walk (RE 2026-08-24, addonly probes:
 * the first two 8-channel surfaces come out ~1/4 range low on every
 * non-uniform input, bitwise with a pointwise host). */
static void
lower_identity_copy(struct rkt_ml_subgraph *subgraph,
                    struct pipe_tensor *input,
                    const struct pipe_tensor *output,
                    unsigned dst_offset, bool force_pointwise,
                    struct rkt_operation *operation)
{
   unsigned channels = input->dims[3];
   bool depthwise = channels > 1 && channels % 32 == 0 && !force_pointwise;
   struct pipe_ml_operation conv = {0};
   struct pipe_tensor out_view = *input;
   struct pipe_tensor weight_tensor = {0};
   struct pipe_tensor bias_tensor = {0};
   struct pipe_tensor *in_ptr = input;
   struct pipe_tensor *out_ptr = &out_view;
   uint8_t *wdata = calloc(channels, depthwise ? 1 : channels);
   int32_t *bdata = calloc(channels, sizeof(int32_t));

   /* The output slice keeps the input's geometry but lives in the concat
    * output tensor, in its quantization domain. */
   out_view.index = output->index;
   out_view.scale = output->scale;
   out_view.zero_point = output->zero_point;

   if (depthwise)
      memset(wdata, 0xff, channels);
   else
      for (unsigned c = 0; c < channels; c++)
         wdata[c * channels + c] = 0xff;

   weight_tensor.dims[0] = depthwise ? 1 : channels;
   weight_tensor.dims[1] = 1;
   weight_tensor.dims[2] = 1;
   weight_tensor.dims[3] = channels;
   weight_tensor.scale = 1.0f / 255;
   weight_tensor.zero_point = 0;
   weight_tensor.data = wdata;

   bias_tensor.dims[3] = channels;
   bias_tensor.data = (uint8_t *)bdata;

   conv.type = PIPE_ML_OPERATION_TYPE_CONVOLUTION;
   conv.input_tensors = &in_ptr;
   conv.output_tensors = &out_ptr;
   conv.conv.weight_tensor = &weight_tensor;
   conv.conv.bias_tensor = &bias_tensor;
   conv.conv.stride_x = 1;
   conv.conv.stride_y = 1;
   conv.conv.depthwise = depthwise;
   conv.conv.pointwise = !depthwise;
   conv.conv.dilation_width_factor = 1;
   conv.conv.dilation_height_factor = 1;

   lower_convolution(subgraph, &conv, operation, 0);
   operation->dst_offset = dst_offset;

   free(wdata);
   free(bdata);
}

/* Whether the concatenation at poperations[conc_idx] is the only reader
 * of the tensor: only then may its producer be retargeted to write the
 * concat output slice directly (the tensor itself then never
 * materializes).  NOTE: a tensor that is also a partition OUTPUT cannot
 * be detected here -- it has no reader among the operations. */
static bool
concat_is_sole_consumer(const struct pipe_ml_operation *poperations,
                        unsigned count, unsigned conc_idx, unsigned index)
{
   for (unsigned j = 0; j < count; j++) {
      if (j == conc_idx)
         continue;
      for (unsigned k = 0; k < poperations[j].input_count; k++)
         if (poperations[j].input_tensors[k]->index == index)
            return false;
   }
   return true;
}

static struct rkt_operation *
find_first_consumer(struct rkt_ml_subgraph *subgraph, unsigned tensor_index)
{
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      if (operation->input_index == tensor_index)
         return operation;
   }

   return NULL;
}

static struct rkt_operation *
find_producer(struct rkt_ml_subgraph *subgraph,
              unsigned tensor_index)
{
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      if (operation->output_index == tensor_index)
         return operation;
   }

   return NULL;
}

static unsigned
count_tensors(const struct pipe_ml_operation *poperations,
              unsigned count)
{
   unsigned tensor_count = 0;

   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];
      tensor_count = MAX2(tensor_count, poperation->input_tensors[0]->index);
      tensor_count = MAX2(tensor_count, poperation->output_tensors[0]->index);
      switch (poperation->type) {
      case PIPE_ML_OPERATION_TYPE_CONVOLUTION:
         tensor_count = MAX2(tensor_count, poperation->conv.weight_tensor->index);
         tensor_count = MAX2(tensor_count, poperation->conv.bias_tensor->index);
         break;
      case PIPE_ML_OPERATION_TYPE_ADD:
      case PIPE_ML_OPERATION_TYPE_MUL:
         tensor_count = MAX2(tensor_count, poperation->input_tensors[1]->index);
         break;
      case PIPE_ML_OPERATION_TYPE_POOLING:
         /* lowered to a synthetic depthwise convolution; no extra tensors */
         break;
      case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED:
         tensor_count = MAX2(tensor_count, poperation->fcon.weight_tensor->index);
         tensor_count = MAX2(tensor_count, poperation->fcon.bias_tensor->index);
         break;
      case PIPE_ML_OPERATION_TYPE_CONCATENATION:
         for (unsigned j = 0; j < poperation->input_count; j++)
            tensor_count =
               MAX2(tensor_count, poperation->input_tensors[j]->index);
         break;
      default:
         DBG("poperation->type %d\n", poperation->type);
         UNREACHABLE("Unsupported ML operation type");
      }
   }

   return tensor_count + 1;
}

static bool
tensor_quantization_supported(const struct pipe_tensor *tensor)
{
   /*
    * Per-axis quantization not supported, for details see:
    * https://ai.google.dev/edge/litert/models/quantization_spec#per-axis_vs_per-tensor
    */
   return tensor->scales == NULL && tensor->zero_points == NULL;
}

bool
rkt_ml_operation_supported(struct pipe_ml_device *pdevice,
                           const struct pipe_ml_operation *operation)
{
   bool supported = false;

   /* Everything here is NHWC feature-map arithmetic; lower-rank tensors
    * (yolo's [1, 84, 2100] tail) arrive right-aligned to 4 dims and
    * would be misread as tiny maps with thousands of channels. */
   for (unsigned k = 0; k < operation->input_count; k++)
      if (operation->input_tensors[k]->data == NULL &&
          operation->input_tensors[k]->dims_count != 4)
         return false;
   for (unsigned k = 0; k < operation->output_count; k++)
      if (operation->output_tensors[k]->dims_count != 4)
         return false;
   /* The CNA/CORE width and height fields are 11 bits (DATAIN_WIDTH,
    * DATAOUT_WIDTH ...): yolo's DFL convolution over a 4x2100 map came
    * out scrambled (layer-w2100 probes, Test 76). */
   for (unsigned k = 0; k < operation->input_count; k++)
      if (operation->input_tensors[k]->data == NULL &&
          (operation->input_tensors[k]->dims[1] > 2047 ||
           operation->input_tensors[k]->dims[2] > 2047))
         return false;
   for (unsigned k = 0; k < operation->output_count; k++)
      if (operation->output_tensors[k]->dims[1] > 2047 ||
          operation->output_tensors[k]->dims[2] > 2047)
         return false;

   switch (operation->type) {
   case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {
      struct pipe_tensor *input_tensor = operation->input_tensors[0];
      struct pipe_tensor *weight_tensor = operation->conv.weight_tensor;
      struct pipe_tensor *bias_tensor = operation->conv.bias_tensor;
      struct pipe_tensor *output_tensor = operation->output_tensors[0];

      /* Per-axis quantization is supported for the WEIGHTS of a
       * convolution (the vendor two-stage BS scheme: per-channel mul
       * against max(scales) plus the layer OUT_CVT), as long as every
       * channel shares one zero point (TFLite per-channel weights are
       * symmetric, zp 0).  Bias scales are s_in * s_wc by construction
       * and never enter the pipeline -- the bias DATA is already
       * quantized with them.  Activations must stay per-tensor. */
      bool weights_quant_ok = tensor_quantization_supported(weight_tensor);
      if (!weights_quant_ok && weight_tensor->scales != NULL) {
         unsigned nch = operation->conv.depthwise ? weight_tensor->dims[3]
                                                  : weight_tensor->dims[0];
         weights_quant_ok = true;
         if (weight_tensor->zero_points != NULL)
            for (unsigned c = 0; c < nch; c++)
               weights_quant_ok &=
                  weight_tensor->zero_points[c] == weight_tensor->zero_points[0];
      }
      bool bias_quant_ok = tensor_quantization_supported(bias_tensor) ||
                           bias_tensor->scales != NULL;

      // Dilation not yet implemented
      if (tensor_quantization_supported(input_tensor) &&
          weights_quant_ok && bias_quant_ok &&
          tensor_quantization_supported(output_tensor) &&
          operation->conv.dilation_width_factor == 1 &&
          operation->conv.dilation_height_factor == 1)
         supported = true;

      break;
   }
   case PIPE_ML_OPERATION_TYPE_ADD:
      supported = operation->input_tensors[0]->data == NULL &&
                  operation->input_tensors[1]->data == NULL;
      break;
   case PIPE_ML_OPERATION_TYPE_QUANTIZE: {
      /* int8/uint8 -> int8/uint8 requantization (TFLite puts one on
       * every concat leg whose scale differs from the concat's): folded
       * into the producer's OUT_CVT when it is the only consumer, an
       * identity convolution otherwise. */
      const struct pipe_tensor *in = operation->input_tensors[0];
      const struct pipe_tensor *out = operation->output_tensors[0];
      supported = tensor_quantization_supported(in) &&
                  tensor_quantization_supported(out) &&
                  in->type_size == 1 && out->type_size == 1 &&
                  in->dims[1] == out->dims[1] && in->dims[2] == out->dims[2] &&
                  in->dims[3] == out->dims[3] && in->dims[3] > 0;
      break;
   }
   case PIPE_ML_OPERATION_TYPE_PAD:
      /* Spatial zero-point padding folded into the consuming
       * convolution (yolo's explicit PAD before stride-2 convs). */
      supported = operation->input_tensors[0]->type_size == 1 &&
                  operation->input_tensors[0]->dims[3] > 0 &&
                  operation->pad.before_z == 0 && operation->pad.after_z == 0 &&
                  operation->pad.before_x <= 7 && operation->pad.after_x <= 7 &&
                  operation->pad.before_y <= 7 && operation->pad.after_y <= 7 &&
                  tensor_quantization_supported(operation->input_tensors[0]) &&
                  operation->input_tensors[0]->scale ==
                     operation->output_tensors[0]->scale &&
                  operation->input_tensors[0]->zero_point ==
                     operation->output_tensors[0]->zero_point;
      break;
   case PIPE_ML_OPERATION_TYPE_STRIDED_SLICE: {
      /* A channel slice on an 8-channel surface boundary is pure
       * addressing for the consumer. */
      const struct pipe_tensor *in = operation->input_tensors[0];
      const struct pipe_tensor *out = operation->output_tensors[0];
      supported = in->dims[3] > 0 && out->dims[3] > 0 && in->type_size == 1 &&
                  operation->slice.begin[0] == 0 &&
                  operation->slice.begin[1] == 0 &&
                  operation->slice.begin[2] == 0 &&
                  operation->slice.end[1] == (int)in->dims[1] &&
                  operation->slice.end[2] == (int)in->dims[2] &&
                  operation->slice.begin[3] % 8 == 0 &&
                  operation->slice.end[3] - operation->slice.begin[3] ==
                     (int)out->dims[3] &&
                  operation->slice.strides[0] == 1 &&
                  operation->slice.strides[1] == 1 &&
                  operation->slice.strides[2] == 1 &&
                  operation->slice.strides[3] == 1 &&
                  tensor_quantization_supported(in) &&
                  in->scale == out->scale && in->zero_point == out->zero_point;
      break;
   }
   case PIPE_ML_OPERATION_TYPE_LOGISTIC:
   case PIPE_ML_OPERATION_TYPE_MUL:
      /* Only as the two halves of conv -> x * sigmoid(x) (SiLU), fused
       * into the convolution through the DPU lookup table (vendor
       * ConvExSwish, RE-LOG Test 73).  Teflon establishes the pattern. */
      supported = operation->silu_pattern &&
                  tensor_quantization_supported(operation->input_tensors[0]) &&
                  tensor_quantization_supported(operation->output_tensors[0]);
      break;
   case PIPE_ML_OPERATION_TYPE_POOLING:
      /* Average pooling runs as a depthwise convolution with constant
       * weights (the vendor does the same: mobilenet_v1 t49 is a 7x7/s7
       * depthwise with an all-0x7f weight buffer).  Padding must be zero:
       * with padding TFLite divides by the number of VALID elements at the
       * edges, which a fixed-weight convolution cannot reproduce. */
      supported = operation->pooling.type == PIPE_ML_POOLING_TYPE_AVG &&
                  tensor_quantization_supported(operation->input_tensors[0]) &&
                  tensor_quantization_supported(operation->output_tensors[0]) &&
                  operation->pooling.filter_width <= 7 &&
                  operation->pooling.filter_height <= 7 &&
                  operation->pooling.padding_top == 0 &&
                  operation->pooling.padding_bottom == 0 &&
                  operation->pooling.padding_left == 0 &&
                  operation->pooling.padding_right == 0;
      /* Max pooling runs on the PPU unit (vendor resnet18 task 1;
       * RE-LOG Tests 60-61).  Quantization must pass through unchanged:
       * the PPU chunk carries no requantization stage.  RKT_NO_PPU=1
       * falls back to the CPU. */
      supported |= getenv("RKT_NO_PPU") == NULL &&
                   operation->pooling.type == PIPE_ML_POOLING_TYPE_MAX &&
                   tensor_quantization_supported(operation->input_tensors[0]) &&
                   tensor_quantization_supported(operation->output_tensors[0]) &&
                   operation->input_tensors[0]->scale ==
                      operation->output_tensors[0]->scale &&
                   operation->input_tensors[0]->zero_point ==
                      operation->output_tensors[0]->zero_point &&
                   operation->pooling.filter_width <= 8 &&
                   operation->pooling.filter_height <= 8 &&
                   operation->pooling.stride_x == operation->pooling.stride_y &&
                   operation->pooling.stride_x >= 1 &&
                   operation->pooling.stride_x <= 8 &&
                   operation->pooling.padding_top <= 15 &&
                   operation->pooling.padding_bottom <= 15 &&
                   operation->pooling.padding_left <= 15 &&
                   operation->pooling.padding_right <= 15 &&
                   /* every window must fit input+padding or the PPU
                    * starves and the chain wedges */
                   (operation->output_tensors[0]->dims[1] - 1) *
                         operation->pooling.stride_y +
                         operation->pooling.filter_height <=
                      operation->input_tensors[0]->dims[1] +
                         operation->pooling.padding_top +
                         operation->pooling.padding_bottom &&
                   (operation->output_tensors[0]->dims[2] - 1) *
                         operation->pooling.stride_x +
                         operation->pooling.filter_width <=
                      operation->input_tensors[0]->dims[2] +
                         operation->pooling.padding_left +
                         operation->pooling.padding_right;
      break;
   case PIPE_ML_OPERATION_TYPE_CONCATENATION: {
      struct pipe_tensor *output_tensor = operation->output_tensors[0];
      unsigned channels = 0;

      /* Only channel-wise concatenation of same-geometry 4D tensors.  In
       * the planar 8-channel surface layout it is pure addressing: each
       * input occupies its own run of surfaces in the output BO.  The
       * slice offsets must fall on the 16-channel allocation granularity,
       * so every input but the last needs C % 16 == 0. */
      if (operation->conc.axis != 3 && operation->conc.axis != -1)
         break;

      supported = tensor_quantization_supported(output_tensor);
      for (unsigned i = 0; i < operation->input_count; i++) {
         struct pipe_tensor *input_tensor = operation->input_tensors[i];

         if (i > 0 && channels % 16 != 0)
            supported = false;
         channels += input_tensor->dims[3];

         supported = supported &&
                     tensor_quantization_supported(input_tensor) &&
                     input_tensor->dims[1] == output_tensor->dims[1] &&
                     input_tensor->dims[2] == output_tensor->dims[2];
      }
      break;
   }
   case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED: {
      struct pipe_tensor *input_tensor = operation->input_tensors[0];

      /* Runs as a 1x1 convolution, so the input must already be a single
       * spatial pixel (a flattened [1, N] input qualifies: teflon
       * right-aligns it into [1, 1, 1, N]). */
      supported = tensor_quantization_supported(input_tensor) &&
                  tensor_quantization_supported(operation->fcon.weight_tensor) &&
                  tensor_quantization_supported(operation->fcon.bias_tensor) &&
                  tensor_quantization_supported(operation->output_tensors[0]) &&
                  input_tensor->dims[1] == 1 && input_tensor->dims[2] == 1;
      break;
   }
   default:
      supported = false;
   }

   return supported;
}

/* RK3568 fused adds with C %% 32 != 0: the EW RDMA misreads the second
 * surface pair, so the fused convolution instead runs with its kernel
 * count padded to align(C, 32) (zero weights/biases in the tail).  Find
 * the pad for a convolution that an ADD later fuses into. */
static unsigned
fused_add_pad(const struct pipe_ml_operation *poperations, unsigned count,
              const struct pipe_ml_operation *conv)
{
   unsigned conv_idx = conv - poperations;
   for (unsigned i = 0; i < count; i++) {
      if (poperations[i].type != PIPE_ML_OPERATION_TYPE_ADD)
         continue;
      unsigned c = poperations[i].output_tensors[0]->dims[3];
      if (c % 32 == 0)
         continue;
      unsigned out = conv->output_tensors[0]->index;
      unsigned in0 = poperations[i].input_tensors[0]->index;
      unsigned in1 = poperations[i].input_tensors[1]->index;
      if (out != in0 && out != in1)
         continue;
      /* Only the fuse host (the later of the two producers, see the ADD
       * lowering) grows its kernel count. */
      unsigned other = out == in0 ? in1 : in0;
      bool later = true;
      for (unsigned j = conv_idx + 1; j < count; j++)
         if (poperations[j].output_tensors[0]->index == other)
            later = false;
      if (later)
         return align(c, 32);
   }
   return 0;
}

static float
silu_f(float x)
{
   return x / (1.0f + expf(-x));
}

/* Fuse x * sigmoid(x) into the convolution: the DPU BN multiplier maps
 * the accumulator into the LUT domain [-16384, 16384] (513 entries per
 * table, one entry per 32 units, LUT_INFO index_select 5), the LE table
 * covers x < 0 and LO x >= 0, the LO overflow slope continues y = 2x
 * above the domain, OUT_CVT requantizes the table output.  Domain and
 * scales follow the vendor probe-SILU stream (RE-LOG Test 73): the LUT
 * spans the largest value the output tensor can hold (silu(x) -> x),
 * the table output is 2 * silu(x) in LUT-domain units. */
/* One LUT domain for every layer: x in [-8, 8) real, 32 LUT units =
 * 1/64 per table entry, y = 2 * silu(x) in LUT units (max 32767 at 8.0).
 * Beyond +8 the LO overflow slope continues y = 2x; below -8 silu is 0
 * to well under one output LSB. */
static void
lower_silu(struct rkt_ml_subgraph *subgraph, struct rkt_operation *host,
           const struct pipe_tensor *out)
{
   host->silu = true;
   host->output_index = out->index;
   host->output_zero_point = out->zero_point;
   host->output_scale = out->scale;

   float s_acc = host->input_scale * host->weights_scale;
   host->lut_scale = RKT_LUT_SCALE;

   /* x_lut = acc * mul >> shift with a 15-bit multiplier. */
   float r = s_acc / host->lut_scale;
   unsigned e = 0;
   while (r * (float)(1u << e) < 16384.0f && e < 30)
      e++;
   host->lut_mul = MIN2((unsigned)lrintf(r * (float)(1u << e)), 32767);
   host->lut_shift = e;

   /* The LO table's first entries misbehave (RE-LOG Test 75: for x in
    * [LO_START, LO_START + ~160) the unit returns LO[512] * (x + 8) / 256
    * instead of the entries), so the LO domain starts at -512, inside
    * the LE range, and LUT_CFG gives LE priority in the overlap. */
   for (unsigned i = 0; i < 513; i++) {
      float xle = (-16384.0f + 32.0f * i) * host->lut_scale;
      float xlo = (RKT_LUT_LO_START + 32.0f * i) * host->lut_scale;
      host->lut_le[i] = CLAMP(lrintf(2.0f * silu_f(xle) / host->lut_scale),
                              -32768, 32767);
      host->lut_lo[i] = CLAMP(lrintf(2.0f * silu_f(xlo) / host->lut_scale),
                              -32768, 32767);
   }

   /* The job's LUT payload: entry i at word i (the unit indexes the
    * table by (x - START) >> 5 straight into the word array; the
    * vendor's leading zero word puts every entry one bin late, which
    * costs it ~1.3 LSB on the steep positive branch), the two trailing
    * words repeat the last entry. */
   if (!subgraph->has_lut) {
      const int16_t *tabs[2] = {host->lut_le, host->lut_lo};
      for (unsigned t = 0; t < 2; t++) {
         uint16_t *w = &subgraph->lut_words[t * 515];
         for (unsigned i = 0; i < 513; i++)
            w[i] = (uint16_t)tabs[t][i];
         w[513] = w[514] = (uint16_t)tabs[t][512];
      }
      subgraph->has_lut = true;
   }
}

/* RKT_NO_CHAIN reverts to one job per operation -- except when the graph
 * holds a PPU pool chunk, which only ever executes as a chain link. */
static bool
rkt_chain_disabled(struct rkt_ml_subgraph *subgraph)
{
   if (!getenv("RKT_NO_CHAIN"))
      return false;
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation, op)
      if (op->is_pool)
         return false;
   return true;
}

struct pipe_ml_subgraph *
rkt_ml_subgraph_create(struct pipe_ml_device *pdevice,
                       const struct pipe_ml_operation *poperations,
                       unsigned count)
{
   struct rkt_screen *screen = rkt_ml_device_screen(pdevice);
   struct rkt_ml_device *dev = rkt_ml_device(pdevice);
   struct rkt_ml_subgraph *subgraph;
   unsigned tensor_count;

   if (!dev->context)
      dev->context = screen->pscreen.context_create(&screen->pscreen, NULL, 0);

   subgraph = calloc(1, sizeof(*subgraph));
   subgraph->base.device = pdevice;
   subgraph->context = dev->context;

   tensor_count = count_tensors(poperations, count);

   /* A PPU pool chunk is hosted by the regcmd BO of a non-pool
    * predecessor in the chain; a max pool with no such predecessor (the
    * first operation of the partition, or right after another pool)
    * gets a synthetic identity-copy carrier inserted in front of it,
    * with a synthetic intermediate tensor.  Reserve a slot per pool. */
   unsigned synth_index = tensor_count;
   for (unsigned i = 0; i < count; i++)
      if (poperations[i].type == PIPE_ML_OPERATION_TYPE_POOLING)
         tensor_count++;

   subgraph->tensors = UTIL_DYNARRAY_INIT;
   subgraph->operations = UTIL_DYNARRAY_INIT;
   subgraph->concat_shapes = UTIL_DYNARRAY_INIT;
   subgraph->views = UTIL_DYNARRAY_INIT;
   if (!util_dynarray_resize(&subgraph->tensors, struct pipe_resource *,
                             tensor_count))
      return NULL;
   memset(util_dynarray_begin(&subgraph->tensors), 0, subgraph->tensors.size);

   /* Lower */
   for (int i = 0; i < count; i++) {
      struct rkt_operation operation = {0};
      operation.add_tensor = -1;
      operation.orig_output_index = -1;

      switch (poperations[i].type) {
      case PIPE_ML_OPERATION_TYPE_CONVOLUTION:
         lower_convolution(subgraph, &poperations[i], &operation,
                           fused_add_pad(poperations, count, &poperations[i]));
         util_dynarray_append(&subgraph->operations, operation);
         break;
      case PIPE_ML_OPERATION_TYPE_POOLING: {
         /* Average pooling prefers the PPU too (no synthetic depthwise
          * weights, no CBUF); the depthwise lowering stays as the
          * fallback when the PPU cannot express it (kernel > 8, or a
          * requantizing pool -- the PPU chunk has no requant stage). */
         const struct pipe_ml_operation *pp = &poperations[i];
         unsigned num_ops = util_dynarray_num_elements(&subgraph->operations,
                                                       struct rkt_operation);
         /* The PPU chunk lives in the regcmd BO of the operation right
          * before it, which must not itself be a pool. */
         bool need_carrier =
            num_ops == 0 ||
            util_dynarray_element(&subgraph->operations, struct rkt_operation,
                                  num_ops - 1)
               ->is_pool;
         bool ppu_ok = getenv("RKT_NO_PPU") == NULL &&
                       pp->pooling.filter_width <= 8 &&
                       pp->pooling.filter_height <= 8 &&
                       pp->pooling.stride_x == pp->pooling.stride_y &&
                       pp->pooling.stride_x <= 8 &&
                       pp->input_tensors[0]->scale ==
                          pp->output_tensors[0]->scale &&
                       pp->input_tensors[0]->zero_point ==
                          pp->output_tensors[0]->zero_point;
         if (pp->pooling.type == PIPE_ML_POOLING_TYPE_MAX ||
             (ppu_ok && !need_carrier)) {
            unsigned pool_input = ~0u;
            if (need_carrier) {
               /* Max pooling has no convolution formulation to fall
                * back on: copy the input into a synthetic intermediate
                * tensor with an identity convolution, which then hosts
                * the chunk. */
               struct rkt_operation copy = {0};
               struct pipe_tensor synth = *pp->input_tensors[0];
               copy.add_tensor = -1;
               synth.index = synth_index++;
               lower_identity_copy(subgraph, pp->input_tensors[0], &synth, 0,
                                   false, &copy);
               util_dynarray_append(&subgraph->operations, copy);
               pool_input = synth.index;
            }
            lower_max_pooling(subgraph, pp, &operation);
            if (pool_input != ~0u)
               operation.input_index = pool_input;
         } else {
            /* Average pooling without a chunk carrier just takes the
             * depthwise lowering. */
            lower_pooling(subgraph, pp, &operation);
         }
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }
      case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED:
         lower_fully_connected(subgraph, &poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      case PIPE_ML_OPERATION_TYPE_CONCATENATION: {
         /* Channel concatenation is addressing, not computation: the
          * output BO is created at full size and each input's producer
          * is retargeted to write its own run of surfaces (dst_offset).
          * When an input cannot be written in place -- produced outside
          * the partition, read by other consumers too, or produced by a
          * PPU pool chunk that would need a requant it does not have --
          * an identity depthwise convolution copies (and requantizes) it
          * into the slice instead. */
         const struct pipe_ml_operation *pconc = &poperations[i];
         const struct pipe_tensor *out = pconc->output_tensors[0];
         unsigned surf = rkt_surf_px(out->dims[1] * out->dims[2]) * 8;
         unsigned ch_off = 0;

         create_tensor(subgraph, out->index,
                       rkt_surf_px(out->dims[1] * out->dims[2]) *
                          DIV_ROUND_UP(out->dims[3], FEATURE_ATOMIC_SIZE) * 2 *
                          FEATURE_ATOMIC_SIZE);

         for (unsigned k = 0; k < pconc->input_count; k++) {
            struct pipe_tensor *in = pconc->input_tensors[k];
            struct rkt_operation *prod =
               find_producer(subgraph, in->index);
            bool requant = in->scale != out->scale ||
                           in->zero_point != out->zero_point;
            unsigned dst_offset = (ch_off / 8) * surf;

            if (prod != NULL &&
                concat_is_sole_consumer(poperations, count, i, in->index) &&
                !(prod->is_pool && requant)) {
               prod->output_index = out->index;
               prod->dst_offset = dst_offset;
               if (!prod->is_pool) {
                  prod->output_zero_point = out->zero_point;
                  prod->output_scale = out->scale;
               }
            } else {
               struct rkt_operation copy = {0};
               copy.add_tensor = -1;
               lower_identity_copy(subgraph, in, out, dst_offset, false,
                                   &copy);
               util_dynarray_append(&subgraph->operations, copy);
            }
            ch_off += in->dims[3];
         }

         struct rkt_concat_shape shape = {
            .index = out->index,
            .width = out->dims[2],
            .height = out->dims[1],
            .channels = out->dims[3],
         };
         util_dynarray_append(&subgraph->concat_shapes, shape);
         break;
      }
      case PIPE_ML_OPERATION_TYPE_ADD: {
         /* Fuse tensor addition into a convolution.  The host must be
          * whichever producer runs LAST in the task chain: its EW stream
          * reads the other input's tensor from memory, so that tensor has
          * to be computed already.  In resnet-style downsample blocks both
          * inputs are produced inside the partition (main path and the 1x1
          * skip) and the tflite order puts the skip conv after the main
          * path -- fusing into the first one made the EW read garbage
          * (resnet18 RE 2026-08-23; the vendor fuses into the 1x1 s2 skip
          * convolutions, its tasks 8/13/18). */
         struct rkt_operation *prod0 =
            find_producer(subgraph, poperations[i].input_tensors[0]->index);
         struct rkt_operation *prod1 =
            find_producer(subgraph, poperations[i].input_tensors[1]->index);
         struct rkt_operation *host, *other_op;
         const struct pipe_tensor *other;

         if (prod0 == NULL && prod1 == NULL) {
            /* Both inputs come from outside the partition (a detached
             * residual, e.g. a yolo partition opening with the ADD):
             * synthesize the host as a pointwise identity convolution
             * copying input 0 into the ADD output's domain; the EW
             * stage adds input 1 straight from the job input buffer. */
            struct rkt_operation copy = {0};

            copy.add_tensor = -1;
            lower_identity_copy(subgraph, poperations[i].input_tensors[0],
                                poperations[i].output_tensors[0], 0, true,
                                &copy);
            util_dynarray_append(&subgraph->operations, copy);
            prod0 = util_dynarray_element(
               &subgraph->operations, struct rkt_operation,
               util_dynarray_num_elements(&subgraph->operations,
                                          struct rkt_operation) -
                  1);
         }

         assert(prod0 || prod1);

         if (!prod1 || (prod0 && prod0 > prod1)) {
            host = prod0;
            other_op = prod1;
            other = poperations[i].input_tensors[1];
         } else {
            host = prod1;
            other_op = prod0;
            other = poperations[i].input_tensors[0];
         }

         /* A depthwise host reads the EW operand with the wrong surface
          * walk (see lower_identity_copy), and a SiLU host has its EW
          * stage taken by the lookup table (yolo bottleneck: add(b,
          * silu(conv(...)))); when the other producer is a regular,
          * non-SiLU convolution AND runs later in the chain it can host
          * instead, otherwise fall back to a pointwise identity copy of
          * the output hosting the add. */
         if (host->depthwise || host->silu) {
            if (other_op && !other_op->depthwise && !other_op->silu &&
                other_op > host) {
               struct rkt_operation *t = host;
               host = other_op;
               other_op = t;
               other = other == poperations[i].input_tensors[0]
                          ? poperations[i].input_tensors[1]
                          : poperations[i].input_tensors[0];
            } else {
               struct pipe_tensor dw_out = {0};
               struct rkt_operation copy = {0};
               unsigned k = host == prod0 ? 0 : 1;

               dw_out = *poperations[i].input_tensors[k];
               copy.add_tensor = -1;
               lower_identity_copy(subgraph, &dw_out,
                                   poperations[i].output_tensors[0], 0, true,
                                   &copy);
               util_dynarray_append(&subgraph->operations, copy);
               host = util_dynarray_element(
                  &subgraph->operations, struct rkt_operation,
                  util_dynarray_num_elements(&subgraph->operations,
                                             struct rkt_operation) -
                     1);
            }
         }

         if (other_op == NULL) {
            /* Graph input, or a channel slice (view) of a tensor: the EW
             * stream reads the source at the surface offset. */
            unsigned idx = other->index, off = 0;
            struct rkt_view *v;
            while ((v = find_view(subgraph, idx)) != NULL) {
               assert(!v->is_pad);
               off += (v->ch_off / 8) *
                      rkt_surf_px(host->output_width * host->output_height) * 8;
               idx = v->src_index;
            }
            host->add_tensor = idx;
            host->add_src_offset = off;
            struct rkt_operation *src_op = find_producer(subgraph, idx);
            if (src_op != NULL)
               src_op->addition_input = true;
         } else {
            other_op->addition_input = true;
            host->add_tensor = other_op->output_index;
         }

         host->output_index = poperations[i].output_tensors[0]->index;
         /* The fused task requants into the ADD's output domain, not the
          * convolution's (mobilenet_v2 RE 2026-08-23: with the conv's old
          * zp/scale left here, OUT_CVT used zp 136 instead of the add
          * output's 133 and the whole residual sum came out shifted). */
         host->output_zero_point =
            poperations[i].output_tensors[0]->zero_point;
         host->output_scale = poperations[i].output_tensors[0]->scale;
         host->addition_offset = 0x80 - other->zero_point;
         host->addition_scale = other->scale;
         host->addition_relu = poperations[i].add.relu;

         break;
      }
      case PIPE_ML_OPERATION_TYPE_STRIDED_SLICE: {
         const struct pipe_ml_operation *ps = &poperations[i];
         struct rkt_view v = {
            .index = ps->output_tensors[0]->index,
            .src_index = ps->input_tensors[0]->index,
            .ch_off = ps->slice.begin[3],
            .channels = ps->output_tensors[0]->dims[3],
            .src_channels = ps->input_tensors[0]->dims[3],
         };
         util_dynarray_append(&subgraph->views, v);
         break;
      }
      case PIPE_ML_OPERATION_TYPE_PAD: {
         const struct pipe_ml_operation *pp = &poperations[i];
         /* teflon: before/after_x pad dims[1] (height), _y dims[2]. */
         struct rkt_view v = {
            .index = pp->output_tensors[0]->index,
            .src_index = pp->input_tensors[0]->index,
            .is_pad = true,
            .src_width = pp->input_tensors[0]->dims[2],
            .src_height = pp->input_tensors[0]->dims[1],
            .pad_top = pp->pad.before_x,
            .pad_bottom = pp->pad.after_x,
            .pad_left = pp->pad.before_y,
            .pad_right = pp->pad.after_y,
         };
         util_dynarray_append(&subgraph->views, v);
         break;
      }
      case PIPE_ML_OPERATION_TYPE_QUANTIZE: {
         const struct pipe_ml_operation *pq = &poperations[i];
         const struct pipe_tensor *in = pq->input_tensors[0];
         const struct pipe_tensor *out = pq->output_tensors[0];
         struct rkt_operation *prod = find_producer(subgraph, in->index);

         if (prod != NULL &&
             concat_is_sole_consumer(poperations, count, i, in->index)) {
            /* Retarget the producer: it requantizes into the QUANTIZE
             * output's domain and writes that tensor directly. */
            prod->orig_output_index = prod->output_index;
            prod->orig_output_zero_point = prod->output_zero_point;
            prod->orig_output_scale = prod->output_scale;
            prod->output_index = out->index;
            prod->output_zero_point = out->zero_point;
            prod->output_scale = out->scale;
         } else {
            struct rkt_operation copy = {0};
            copy.add_tensor = -1;
            lower_identity_copy(subgraph, pq->input_tensors[0], out, 0, false,
                                &copy);
            util_dynarray_append(&subgraph->operations, copy);
         }
         break;
      }
      case PIPE_ML_OPERATION_TYPE_LOGISTIC:
         /* Half of a SiLU pattern (rkt_ml_operation_supported): folded
          * into the convolution when its MUL comes by. */
         break;
      case PIPE_ML_OPERATION_TYPE_MUL: {
         /* MUL(conv_out, sigmoid(conv_out)): the convolution runs the
          * DPU LUT path and writes the MUL's output directly. */
         struct rkt_operation *host = NULL;
         for (unsigned k = 0; k < 2 && host == NULL; k++)
            host = find_producer(subgraph, poperations[i].input_tensors[k]->index);
         assert(host && !host->silu && host->add_tensor == -1 && !host->relu &&
                !host->addition_input);
         lower_silu(subgraph, host, poperations[i].output_tensors[0]);
         break;
      }
      default:
         DBG("poperation->type %d\n", poperations[i].type);
         UNREACHABLE("Unsupported ML operation type");
      }
   }

   /* Create input tensors */
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      unsigned input_channels_1 =
         DIV_ROUND_UP(operation->src_channels, FEATURE_ATOMIC_SIZE) * 2;
      unsigned input_channels_2 = FEATURE_ATOMIC_SIZE;
      unsigned input_size =
         rkt_surf_px(operation->input_width * operation->input_height) *
         input_channels_1 * input_channels_2;

      create_tensor(subgraph, operation->input_index, input_size);

      /* A fused-add second input read from outside the partition has no
       * producer to create it; its geometry is the host's OUTPUT one. */
      if (operation->add_tensor != -1 &&
          find_producer(subgraph, operation->add_tensor) == NULL) {
         unsigned add_size =
            rkt_surf_px(operation->output_width * operation->output_height) *
            DIV_ROUND_UP(operation->output_channels, FEATURE_ATOMIC_SIZE) * 2 *
            FEATURE_ATOMIC_SIZE;
         create_tensor(subgraph, operation->add_tensor, add_size);
      }
   }

   /* Create output tensors */
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      struct rkt_resource *res =
         rkt_get_tensor(subgraph, operation->output_index);
      if (res != NULL)
         continue;

      create_tensor(subgraph, operation->output_index,
                    calc_raw_output_size(operation));
   }

   /* Compile.  A pool operation's PPU chunk is emitted into the PREVIOUS
    * operation's regcmd BO (the PC cannot fetch it from a foreign BO),
    * so it is handled while compiling that operation. */
   {
      unsigned n = util_dynarray_num_elements(&subgraph->operations,
                                              struct rkt_operation);
      for (unsigned i = 0; i < n; i++) {
         struct rkt_operation *operation = util_dynarray_element(
            &subgraph->operations, struct rkt_operation, i);
         struct rkt_operation *pool = NULL;

         if (operation->is_pool)
            continue;
         if (i + 1 < n) {
            struct rkt_operation *next = util_dynarray_element(
               &subgraph->operations, struct rkt_operation, i + 1);
            if (next->is_pool) {
               struct split_task task = {0};
               util_dynarray_append(&next->tasks, task);
               pool = next;
            }
         }
         rkt_split_tasks(subgraph, operation);
         compile_operation(subgraph, operation, pool);
      }
   }

   /* Link every operation's last regcmd stream to the next operation's
    * first one through the PC tail, the same way compile_operation links
    * tasks inside an operation: the whole graph then runs as one PC task
    * chain in a single kernel job (one IRQ+fence round trip instead of
    * one per operation, ~0.2 ms each on RK3568). */
   if (!rkt_chain_disabled(subgraph)) {
      unsigned num_ops = util_dynarray_num_elements(&subgraph->operations,
                                                    struct rkt_operation);
      for (unsigned i = 0; i + 1 < num_ops; i++) {
         struct rkt_operation *op = util_dynarray_element(
            &subgraph->operations, struct rkt_operation, i);
         struct rkt_operation *next = util_dynarray_element(
            &subgraph->operations, struct rkt_operation, i + 1);
         struct split_task *last = util_dynarray_element(
            &op->tasks, struct split_task,
            util_dynarray_num_elements(&op->tasks, struct split_task) - 1);
         struct split_task *first =
            util_dynarray_element(&next->tasks, struct split_task, 0);
         /* A pool operation's chunk lives in the PREVIOUS operation's
          * regcmd BO (it has none of its own). */
         struct pipe_resource *chunk_bo =
            op->is_pool ? util_dynarray_element(&subgraph->operations,
                                                struct rkt_operation, i - 1)
                             ->regcmd
                        : op->regcmd;
         struct pipe_transfer *xfer = NULL;
         uint64_t *words =
            pipe_buffer_map(subgraph->context, chunk_bo, PIPE_MAP_READ_WRITE, &xfer);
         unsigned base = (last->regcfg_addr -
                          rkt_resource(chunk_bo)->phys_addr) /
                         sizeof(uint64_t);
         uint64_t *tail = words + base + last->regcfg_amount;

         tail[-4] |= (uint64_t)first->regcfg_addr << 16;
         tail[-3] |= (uint64_t)(first->regcfg_amount - 1) << 16;
         pipe_buffer_unmap(subgraph->context, xfer);
      }
   }

   return &subgraph->base;
}

void
rkt_ml_subgraph_invoke(struct pipe_context *pcontext,
                       struct pipe_ml_subgraph *psubgraph,
                       unsigned inputs_count, unsigned input_idxs[],
                       void *inputs[], bool is_signed[])
{
   struct rkt_screen *screen = rkt_screen(pcontext->screen);
   struct rkt_ml_subgraph *subgraph = (struct rkt_ml_subgraph *)(psubgraph);
   int ret;

   DBG("Processing input\n");

   for (int i = 0; i < inputs_count; i++) {
      struct rkt_operation *operation =
         find_first_consumer(subgraph, input_idxs[i]);
      bool addition_feed = false;

      /* A partition input can also be the second operand of a fused
       * addition (no operation lists it as input_index then): pack it
       * into the add tensor's BO with the HOST's output geometry and
       * the addition operand's zero point. */
      if (operation == NULL) {
         util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                                op) {
            if (op->add_tensor == (int)input_idxs[i]) {
               operation = op;
               addition_feed = true;
               break;
            }
         }
      }

      struct pipe_resource *input =
         &rkt_get_tensor(subgraph, input_idxs[i])->base;
      unsigned input_channels =
         addition_feed ? operation->output_channels : operation->src_channels;
      unsigned output_channels = operation->output_channels;

      struct rkt_resource *input_tensor = rkt_get_tensor(
         subgraph,
         addition_feed ? (unsigned)operation->add_tensor
                       : operation->input_index);
      /* INT8 user buffers are folded into the driver's uint8 domain
       * bytewise (q_u8 = q_i8 XOR 0x80); the zero points were already
       * shifted by +128 when the tensors were created. */
      uint8_t fold = is_signed[i] ? 0x80 : 0;
      if (output_channels == 1 && input_channels == 1 &&
          !operation->addition_input && (operation->add_tensor == -1) &&
          !fold) {
         pipe_buffer_copy(pcontext, &input_tensor->base, input, 0, 0,
                          pipe_buffer_size(input));
      } else {
         unsigned input_width = addition_feed ? operation->output_width
                                              : operation->input_width;
         unsigned input_height = addition_feed ? operation->output_height
                                               : operation->input_height;
         unsigned zero_point = addition_feed
                                  ? 0x80 - operation->addition_offset
                                  : operation->input_zero_point;
         struct pipe_transfer *transfer_out;
         /* NHWC user memory: [height (dims[1])][width (dims[2])][C]. */
         uint8_t(*input_in)[input_width][input_channels] = inputs[i];
         uint8_t *map = pipe_buffer_map(pcontext, &input_tensor->base,
                                        PIPE_MAP_WRITE, &transfer_out);

         DBG("Converting data\n");

         /*
          * From the NVDLA docs: "For int8, one element of data refers to an 8-bit
          * signed integer." But only when transposing do we seem to need to
          * convert to signed. The DMA unit seems to be able to convert from
          * unsigned to signed though.
          */
         if (input_channels == 3) {
            /* ARGB input: packed RGB, raw uint8 (the CNA CVT stage shifts by
             * -128), rows aligned to 8 bytes with zero padding. */
            unsigned line = DIV_ROUND_UP(input_width * 3, 8) * 8;
            for (int y = 0; y < input_height; y++) {
               unsigned n = y * line;
               for (int x = 0; x < input_width; x++)
                  for (int c = 0; c < 3; c++)
                     map[n++] = input_in[y][x][c] ^ fold;
               for (; n < (y + 1) * line;)
                  map[n++] = 0;
            }
         } else if (input_channels == 1) {
            unsigned n = 0;
            for (int y = 0; y < input_height; y++) {
               for (int x = 0; x < MAX2(input_width, FEATURE_ATOMIC_SIZE); x++) {
                  if (x < input_width)
                     map[n++] = input_in[y][x][0] ^ fold;
                  else
                     map[n++] = zero_point;
               }
            }
         } else {
            /* RK3568 feature layout (verified against the live vendor
             * capture, probe-D2 impulses 2026-08-22): 8 bytes per pixel
             * carrying 8 channels, planar 8-channel surfaces of stride
             * W*H*8; all CNA/DPU strides are in 8-byte units. */
            unsigned n = 0;
            unsigned surf_pad =
               rkt_surf_px(input_width * input_height) -
               input_width * input_height;
            for (int u = 0; u < DIV_ROUND_UP(input_channels, 8); u++) {
               for (int y = 0; y < input_height; y++) {
                  for (int x = 0; x < input_width; x++) {
                     for (int c = 0; c < 8; c++) {
                        unsigned input_channel = c + u * 8;
                        if (input_channel < input_channels)
                           map[n++] = (input_in[y][x][input_channel] ^ fold) -
                                      0x80;
                        else
                           map[n++] = zero_point - 0x80;
                     }
                  }
               }
               for (unsigned p = 0; p < surf_pad * 8; p++)
                  map[n++] = zero_point - 0x80;
            }
         }

         if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS))
            rkt_dump_buffer(map, "input", 0, input_idxs[i], 0,
                            pipe_buffer_size(&input_tensor->base));

         DBG("Converted data\n");

         pipe_buffer_unmap(pcontext, transfer_out);
      }
   }
   DBG("Processed input\n");

   DBG("Submitting graph\n");

   struct util_dynarray jobs = UTIL_DYNARRAY_INIT;

   /* Chained submit: the graph's regcmd streams were linked at compile
    * time, so the whole network is ONE job.  External inputs go on the
    * in list, every operation output on the out list; intermediates are
    * both read and written inside the job, and a BO repeated across the
    * two lists wedges the scheduler -- they are covered by the out list
    * entry alone. */
   if (!rkt_chain_disabled(subgraph)) {
      unsigned num_ops = util_dynarray_num_elements(&subgraph->operations,
                                                    struct rkt_operation);
      unsigned total_tasks = 0;
      /* PPU pool chunks are real tasks: the vendor's task table lists
       * them with enable_mask 0x60 and int_mask 0xc00 (probe-MP task 2),
       * and TASK_NUMBER counts them -- leaving them out desynchronizes
       * the PC task pipeline and the PPU core never starts. */
      util_dynarray_foreach (&subgraph->operations, struct rkt_operation, op)
         total_tasks +=
            util_dynarray_num_elements(&op->tasks, struct split_task);

      /* One interrupt per job, the last task's (vendor scheme: the PC
       * applies per-task masks from the descriptor array). */
      struct rkt_operation *last_op = util_dynarray_element(
         &subgraph->operations, struct rkt_operation, num_ops - 1);
      uint32_t pool_int_mask =
         getenv("RKT_POOL_INT_MASK")
            ? strtol(getenv("RKT_POOL_INT_MASK"), NULL, 0)
            : 0xc00;
      uint32_t last_int_mask = last_op->is_pool ? pool_int_mask : 0x300;

      struct drm_rocket_task *tasks = calloc(total_tasks, sizeof(*tasks));
      bool *task_is_pool = calloc(total_tasks, sizeof(bool));
      uint32_t *in_bo_handles = calloc(num_ops * 2, sizeof(uint32_t));
      uint32_t *out_bo_handles = calloc(num_ops, sizeof(uint32_t));
      unsigned num_inputs = 0, num_outputs = 0, ti = 0;

      util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                             op) {
         util_dynarray_foreach (&op->tasks, struct split_task, task) {
            tasks[ti].regcmd = task->regcfg_addr;
            tasks[ti].regcmd_count = task->regcfg_amount;
            task_is_pool[ti] = op->is_pool;
            ti++;
         }

         unsigned reads[2] = {op->input_index, ~0u};
         if (op->add_tensor != -1)
            reads[1] = op->add_tensor;
         for (unsigned r = 0; r < 2; r++) {
            if (reads[r] == ~0u || find_producer(subgraph, reads[r]) != NULL)
               continue;
            uint32_t handle = rkt_get_tensor(subgraph, reads[r])->handle;
            bool seen = false;
            for (unsigned k = 0; k < num_inputs; k++)
               seen |= in_bo_handles[k] == handle;
            if (!seen)
               in_bo_handles[num_inputs++] = handle;
         }

         uint32_t out_handle =
            rkt_get_tensor(subgraph, op->output_index)->handle;
         bool seen = false;
         for (unsigned k = 0; k < num_outputs; k++)
            seen |= out_bo_handles[k] == out_handle;
         if (!seen)
            out_bo_handles[num_outputs++] = out_handle;
      }

      struct pc_task_desc {
         uint32_t flags, op_idx, enable_mask, int_mask, int_clear,
                  int_status, regcfg_amount, regcfg_offset;
         uint64_t regcmd_addr;
      } __attribute__((packed));
      struct pipe_resource *descs_rsc = pipe_buffer_create(
         pcontext->screen, 0, PIPE_USAGE_DEFAULT,
         total_tasks * sizeof(struct pc_task_desc));
      struct pipe_transfer *xfer = NULL;
      struct pc_task_desc *d =
         pipe_buffer_map(pcontext, descs_rsc, PIPE_MAP_WRITE, &xfer);
      for (unsigned k = 0; k < total_tasks; k++) {
         d[k].flags = 0;
         d[k].op_idx = k + 1;
         d[k].enable_mask = task_is_pool[k] ? 0x60 : 0x1f;
         d[k].int_mask = task_is_pool[k] ? 0xc00 : 0x300;
         d[k].int_clear = 0x1ffff;
         d[k].int_status = 0;
         /* The vendor descriptor amount excludes the 4-word PC tail
          * (conv chunks: 133 of 137 words, pool: 29 of 33); the PC adds
          * RKNPU_PC_DATA_EXTRA_AMOUNT itself when it walks the array. */
         d[k].regcfg_amount = tasks[k].regcmd_count - 4;
         d[k].regcfg_offset = 0;
         d[k].regcmd_addr = tasks[k].regcmd;
      }
      pipe_buffer_unmap(pcontext, xfer);
      free(task_is_pool);
      util_dynarray_element(&subgraph->operations, struct rkt_operation, 0)
         ->task_descs = descs_rsc;

      struct drm_rocket_job job = {0};
      job.task_struct_size = sizeof(struct drm_rocket_task);
      job.in_bo_handles = (uint64_t)(uintptr_t)in_bo_handles;
      job.in_bo_handle_count = num_inputs;
      job.out_bo_handles = (uint64_t)(uintptr_t)out_bo_handles;
      job.out_bo_handle_count = num_outputs;
      job.tasks = (uint64_t)(uintptr_t)tasks;
      job.task_count = total_tasks;
      job.task_desc_addr = rkt_resource(descs_rsc)->phys_addr;
      job.last_int_mask = last_int_mask;
      if (subgraph->has_lut) {
         job.lut_data = (uint64_t)(uintptr_t)subgraph->lut_words;
         job.lut_count = 1030;
      }
      util_dynarray_append(&jobs, job);
   } else
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      unsigned num_inputs = operation->add_tensor != -1 ? 2 : 1;
      uint32_t *in_bo_handles = calloc(num_inputs, sizeof(uint32_t));
      uint32_t *out_bo_handles = malloc(sizeof(uint32_t));

      in_bo_handles[0] = rkt_get_tensor(subgraph, operation->input_index)->handle;

      if (operation->add_tensor != -1) {
         in_bo_handles[1] =
            rkt_get_tensor(subgraph, operation->add_tensor)->handle;
         /* A duplicated handle in the job's BO list makes the scheduler
          * wedge silently (same failure mode as one BO in both the in and
          * out lists) -- happens when the convolution input doubles as the
          * second add input (add(x, conv(x))). */
         if (in_bo_handles[1] == in_bo_handles[0])
            num_inputs = 1;
      }

      out_bo_handles[0] =
         rkt_get_tensor(subgraph, operation->output_index)->handle;

      if (operation->reuse_weights_cbuf) {
         /* Submit all tasks to the same core, so weights can be reused */
         unsigned num_tasks =
            util_dynarray_num_elements(&operation->tasks, struct split_task);
         struct drm_rocket_task *tasks = calloc(num_tasks, sizeof(*tasks));
         unsigned task_count = 0;
         util_dynarray_foreach (&operation->tasks, struct split_task, task) {
            tasks[task_count].regcmd = task->regcfg_addr;
            tasks[task_count].regcmd_count = task->regcfg_amount;
            task_count++;
         }
         struct drm_rocket_job job = {0};
         job.task_struct_size = sizeof(struct drm_rocket_task);
         job.in_bo_handles = (uint64_t)(uintptr_t)in_bo_handles;
         job.in_bo_handle_count = num_inputs;
         job.out_bo_handles = (uint64_t)(uintptr_t)out_bo_handles;
         job.out_bo_handle_count = 1;
         job.tasks = (uint64_t)tasks;
         job.task_count = task_count;
         if (subgraph->has_lut) {
            job.lut_data = (uint64_t)(uintptr_t)subgraph->lut_words;
            job.lut_count = 1030;
         }

         /* RK3568 PC task-DMA mode (vendor rknpu_job): build the 40-byte
          * descriptor array so the PC unit walks the whole chain itself --
          * no CPU stepping between tasks, no ping-pong config races. */
         if (task_count > 0) {
            struct pc_task_desc {
               uint32_t flags, op_idx, enable_mask, int_mask, int_clear,
                        int_status, regcfg_amount, regcfg_offset;
               uint64_t regcmd_addr;
            } __attribute__((packed));
            struct pipe_resource *descs_rsc = pipe_buffer_create(
               pcontext->screen, 0, PIPE_USAGE_DEFAULT,
               task_count * sizeof(struct pc_task_desc));
            struct pipe_transfer *xfer = NULL;
            struct pc_task_desc *d =
               pipe_buffer_map(pcontext, descs_rsc, PIPE_MAP_WRITE, &xfer);
            unsigned ti = 0;

            util_dynarray_foreach (&operation->tasks, struct split_task, t) {
               d[ti].flags = 0;
               d[ti].op_idx = ti + 1;
               d[ti].enable_mask = 0x1f;
               d[ti].int_mask = 0x300;
               d[ti].int_clear = 0x1ffff;
               d[ti].int_status = 0;
               d[ti].regcfg_amount = t->regcfg_amount - 1;
               d[ti].regcfg_offset = 0;
               d[ti].regcmd_addr = t->regcfg_addr;
               ti++;
            }
            pipe_buffer_unmap(pcontext, xfer);
            job.task_desc_addr = rkt_resource(descs_rsc)->phys_addr;
            operation->task_descs = descs_rsc;
         }

         util_dynarray_append(&jobs, job);
      } else {
         /* Spread tasks among cores, for parallelism */
         util_dynarray_foreach (&operation->tasks, struct split_task, task) {
            struct drm_rocket_task *ktask = calloc(1, sizeof(*ktask));
            ktask->regcmd = task->regcfg_addr;
            ktask->regcmd_count = task->regcfg_amount;

            struct drm_rocket_job job = {0};
            job.task_struct_size = sizeof(struct drm_rocket_task);
            job.in_bo_handles = (uint64_t)(uintptr_t)in_bo_handles;
            job.in_bo_handle_count = num_inputs;
            job.out_bo_handles = (uint64_t)(uintptr_t)out_bo_handles;
            job.out_bo_handle_count = 1;
            job.tasks = (uint64_t)ktask;
            job.task_count = 1;
            util_dynarray_append(&jobs, job);
         }
      }
   }

   struct drm_rocket_submit submit = {0};
   submit.job_struct_size = sizeof(struct drm_rocket_job);
   submit.jobs = (uint64_t)util_dynarray_begin(&jobs);
   submit.job_count = util_dynarray_num_elements(&jobs, struct drm_rocket_job);

   ret = drmIoctl(screen->fd, DRM_IOCTL_ROCKET_SUBMIT, &submit);
   assert(ret == 0);

   util_dynarray_foreach (&jobs, struct drm_rocket_job, job) {
      free((void *)job->in_bo_handles);
      free((void *)job->out_bo_handles);
      free((void *)job->tasks);
   }
   util_dynarray_fini(&jobs);

   DBG("Submitted graph\n");
}

void
rkt_ml_subgraph_read_outputs(struct pipe_context *pcontext,
                             struct pipe_ml_subgraph *psubgraph,
                             unsigned outputs_count,
                             unsigned output_idxs[], void *outputs[],
                             bool is_signed[])
{
   struct rkt_ml_subgraph *subgraph = (struct rkt_ml_subgraph *)(psubgraph);

   DBG("Processing output\n");

   for (int i = 0; i < outputs_count; i++) {
      unsigned idx = output_idxs[i];
      unsigned ch_off = 0, view_channels = 0;
      struct rkt_view *v;

      /* A partition output that is a channel slice of a tensor: read
       * the source at the surface offset. */
      while ((v = find_view(subgraph, idx)) != NULL) {
         assert(!v->is_pad);
         ch_off += v->ch_off;
         if (!view_channels)
            view_channels = v->channels;
         idx = v->src_index;
      }

      struct rkt_operation *operation = find_producer(subgraph, idx);
      /* A producer whose QUANTIZE was folded in still has to serve the
       * pre-quantization tensor when the partition exports it. */
      bool requant = false;
      if (operation == NULL) {
         util_dynarray_foreach (&subgraph->operations, struct rkt_operation, op)
            if (op->orig_output_index == (int)idx) {
               operation = op;
               requant = true;
               idx = op->output_index;
               break;
            }
      }
      assert(operation != NULL);
      struct rkt_resource *output_tensor = rkt_get_tensor(subgraph, idx);
      struct pipe_transfer *transfer = NULL;
      uint8_t *raw_output;
      unsigned out_w = operation->output_width;
      unsigned out_h = operation->output_height;
      unsigned out_c = operation->output_channels;

      /* A concatenation output has several producers, each knowing only
       * its own slice -- the full dims live in the shape table. */
      util_dynarray_foreach (&subgraph->concat_shapes,
                             struct rkt_concat_shape, cs) {
         if (cs->index == idx) {
            out_w = cs->width;
            out_h = cs->height;
            out_c = cs->channels;
            break;
         }
      }
      if (view_channels)
         out_c = view_channels;
      /* RK3568 DPU output layout (RE 2026-08-22, layer-c28 quadrant map):
       * surfaces of 8 channels, 8 bytes per pixel, surface stride
       * Wout * Hout * 8 -- matches the vendor DST_SURF_STRIDE (0x1880 =
       * 28*28*8 for the 28x28 probe).  The previous 16-channel unpack only
       * looked correct on channel-uniform fills. */
      uint8_t(*output_in)[out_h][out_w][8];
      uint8_t(*output_out)[out_w][out_c];

      DBG("Before pipe_buffer_map\n");
      raw_output = pipe_buffer_map(pcontext, &output_tensor->base, PIPE_MAP_READ,
                                   &transfer);
      DBG("After pipe_buffer_map\n");

      DBG("Converting data\n");

      output_in = (void *)raw_output;
      output_out = (void *)outputs[i];

      if (DBG_ENABLED(ROCKET_DBG_DUMP_BOS))
         rkt_dump_buffer(raw_output, "output", 0, 0, 0, output_tensor->bo_size);

      {
         /* Same planar 8-channel / 8-byte-pixel layout as the input side
          * (verified against the live vendor capture 2026-08-22). */
         uint8_t *raw = (uint8_t *)output_in;
         unsigned rows = out_h;   /* memory rows, dims[1] */
         unsigned cols = out_w;   /* contiguous row length, dims[2] */
         unsigned surf = rkt_surf_px(rows * cols) * 8;
         /* An INT8 user buffer wants q_i8 = q_u8 XOR 0x80 (the driver
          * domain is uint8, see subgraph_invoke). */
         uint8_t fold = is_signed[i] ? 0x80 : 0;
         raw += (ch_off / 8) * surf;
         float rq_scale = requant ? operation->output_scale /
                                       operation->orig_output_scale
                                  : 1.0f;
         for (int oc = 0; oc < out_c; oc++) {
            unsigned g = oc / 8, c = oc % 8;
            for (unsigned y = 0; y < rows; y++) {
               for (unsigned x = 0; x < cols; x++) {
                  uint8_t q = raw[g * surf + (y * cols + x) * 8 + c] + 0x80;
                  if (requant) {
                     float r = (float)operation->orig_output_zero_point +
                               ((int)q - (int)operation->output_zero_point) *
                                  rq_scale;
                     q = (uint8_t)CLAMP(lrintf(r), 0, 255);
                  }
                  output_out[y][x][oc] = q ^ fold;
               }
            }
         }
      }

      DBG("Converted data\n");

      pipe_buffer_unmap(pcontext, transfer);
   }

   DBG("Processed output\n");
}

static void
free_operation(struct rkt_operation *operation)
{
   util_dynarray_fini(&operation->tasks);
   pipe_resource_reference(&operation->regcmd, NULL);
   pipe_resource_reference(&operation->task_descs, NULL);
   pipe_resource_reference(&operation->weights, NULL);
   pipe_resource_reference(&operation->biases, NULL);
}

void
rkt_ml_subgraph_destroy(struct pipe_ml_device *pdevice,
                        struct pipe_ml_subgraph *psubgraph)
{
   struct rkt_ml_subgraph *subgraph = (struct rkt_ml_subgraph *)(psubgraph);

   util_dynarray_foreach (&subgraph->operations, struct rkt_operation, operation)
      free_operation(operation);
   util_dynarray_fini(&subgraph->operations);

   util_dynarray_foreach (&subgraph->tensors, struct pipe_resource *, tensor)
      if (tensor)
         pipe_resource_reference(tensor, NULL);
   util_dynarray_fini(&subgraph->tensors);
   util_dynarray_fini(&subgraph->concat_shapes);
   util_dynarray_fini(&subgraph->views);

   free(subgraph);
}
