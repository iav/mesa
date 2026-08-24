/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include "pipe/p_state.h"
#include "util/macros.h"
#include "util/u_dynarray.h"
#include "util/u_inlines.h"

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
      assert(size == pipe_buffer_size(res));
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

   operation->input_index = poperation->input_tensors[0]->index;
   operation->input_width = poperation->input_tensors[0]->dims[1];
   operation->input_height = poperation->input_tensors[0]->dims[2];
   operation->input_channels = poperation->input_tensors[0]->dims[3];
   operation->input_zero_point = poperation->input_tensors[0]->zero_point;
   operation->input_scale = poperation->input_tensors[0]->scale;

   operation->output_index = poperation->output_tensors[0]->index;
   operation->output_width = poperation->output_tensors[0]->dims[1];
   operation->output_height = poperation->output_tensors[0]->dims[2];
   operation->output_channels = poperation->output_tensors[0]->dims[3];
   operation->output_zero_point = poperation->output_tensors[0]->zero_point;
   operation->output_scale = poperation->output_tensors[0]->scale;

   operation->weights_width = poperation->conv.weight_tensor->dims[1];
   operation->weights_height = poperation->conv.weight_tensor->dims[2];
   operation->weights_zero_point = poperation->conv.weight_tensor->zero_point;
   operation->weights_scale = poperation->conv.weight_tensor->scale;

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
   operation->input_width = ppool->input_tensors[0]->dims[1];
   operation->input_height = ppool->input_tensors[0]->dims[2];
   operation->input_channels = ppool->input_tensors[0]->dims[3];
   operation->input_zero_point = ppool->input_tensors[0]->zero_point;
   operation->input_scale = ppool->input_tensors[0]->scale;

   operation->output_index = ppool->output_tensors[0]->index;
   operation->output_width = ppool->output_tensors[0]->dims[1];
   operation->output_height = ppool->output_tensors[0]->dims[2];
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
         tensor_count = MAX2(tensor_count, poperation->input_tensors[1]->index);
         break;
      case PIPE_ML_OPERATION_TYPE_POOLING:
         /* lowered to a synthetic depthwise convolution; no extra tensors */
         break;
      default:
         DBG("poperation->type %d\n", poperation->type);
         UNREACHABLE("Unsupported ML operation type");
      }
   }

   return tensor_count + 1;
}

static bool
tensor_quantization_supported(struct pipe_tensor *tensor)
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

   switch (operation->type) {
   case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {
      struct pipe_tensor *input_tensor = operation->input_tensors[0];
      struct pipe_tensor *weight_tensor = operation->conv.weight_tensor;
      struct pipe_tensor *bias_tensor = operation->conv.bias_tensor;
      struct pipe_tensor *output_tensor = operation->output_tensors[0];

      // Dilation and per-axis quantization not yet implemented
      if (tensor_quantization_supported(input_tensor) &&
          tensor_quantization_supported(weight_tensor) &&
          tensor_quantization_supported(bias_tensor) &&
          tensor_quantization_supported(output_tensor) &&
          operation->conv.dilation_width_factor == 1 &&
          operation->conv.dilation_height_factor == 1)
         supported = true;

      /* RK3568: a fully-connected-shaped convolution whose weights exceed
       * the 256 KiB CBUF (e.g. the final 1024->1001 1x1 of mobilenet_v1)
       * needs the vendor fp16 FC mode, which is not implemented yet -- the
       * current conv path wedges on it (job timeout). */
      {
         unsigned kernels = weight_tensor->dims[0];
         unsigned wbytes = kernels * weight_tensor->dims[1] *
                           weight_tensor->dims[2] * weight_tensor->dims[3];
         if (wbytes > 7 * CBUF_BANK_SIZE && getenv("RKT_NO_FC"))
            supported = false;
      }

      break;
   }
   case PIPE_ML_OPERATION_TYPE_ADD:
      supported = operation->input_tensors[0]->data == NULL &&
                  operation->input_tensors[1]->data == NULL;
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
                         operation->pooling.stride_x +
                         operation->pooling.filter_width <=
                      operation->input_tensors[0]->dims[1] +
                         operation->pooling.padding_left +
                         operation->pooling.padding_right &&
                   (operation->output_tensors[0]->dims[2] - 1) *
                         operation->pooling.stride_y +
                         operation->pooling.filter_height <=
                      operation->input_tensors[0]->dims[2] +
                         operation->pooling.padding_top +
                         operation->pooling.padding_bottom;
      break;
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
   subgraph->tensors = UTIL_DYNARRAY_INIT;
   subgraph->operations = UTIL_DYNARRAY_INIT;
   if (!util_dynarray_resize(&subgraph->tensors, struct pipe_resource *,
                             tensor_count))
      return NULL;
   memset(util_dynarray_begin(&subgraph->tensors), 0, subgraph->tensors.size);

   /* Lower */
   for (int i = 0; i < count; i++) {
      struct rkt_operation operation = {0};
      operation.add_tensor = -1;

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
         bool ppu_ok = getenv("RKT_NO_PPU") == NULL &&
                       pp->pooling.filter_width <= 8 &&
                       pp->pooling.filter_height <= 8 &&
                       pp->pooling.stride_x == pp->pooling.stride_y &&
                       pp->pooling.stride_x <= 8 &&
                       pp->input_tensors[0]->scale ==
                          pp->output_tensors[0]->scale &&
                       pp->input_tensors[0]->zero_point ==
                          pp->output_tensors[0]->zero_point;
         if (pp->pooling.type == PIPE_ML_POOLING_TYPE_MAX || ppu_ok)
            lower_max_pooling(subgraph, pp, &operation);
         else
            lower_pooling(subgraph, pp, &operation);
         util_dynarray_append(&subgraph->operations, operation);
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

         if (other_op == NULL) {
            /* Graph input */
            host->add_tensor = other->index;
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
      default:
         DBG("poperation->type %d\n", poperations[i].type);
         UNREACHABLE("Unsupported ML operation type");
      }
   }

   /* Create input tensors */
   util_dynarray_foreach (&subgraph->operations, struct rkt_operation,
                          operation) {
      unsigned input_channels_1 =
         DIV_ROUND_UP(operation->input_channels, FEATURE_ATOMIC_SIZE) * 2;
      unsigned input_channels_2 = FEATURE_ATOMIC_SIZE;
      unsigned input_size =
         rkt_surf_px(operation->input_width * operation->input_height) *
         input_channels_1 * input_channels_2;

      create_tensor(subgraph, operation->input_index, input_size);
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
      struct pipe_resource *input =
         &rkt_get_tensor(subgraph, input_idxs[i])->base;
      unsigned input_channels = operation->input_channels;
      unsigned output_channels = operation->output_channels;

      struct rkt_resource *input_tensor =
         rkt_get_tensor(subgraph, operation->input_index);
      if (output_channels == 1 && input_channels == 1 &&
          !operation->addition_input && (operation->add_tensor == -1)) {
         pipe_buffer_copy(pcontext, &input_tensor->base, input, 0, 0,
                          pipe_buffer_size(input));
      } else {
         unsigned input_width = operation->input_width;
         unsigned input_height = operation->input_height;
         unsigned zero_point = operation->input_zero_point;
         struct pipe_transfer *transfer_out;
         uint8_t(*input_in)[input_height][input_channels] = inputs[i];
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
            unsigned line = DIV_ROUND_UP(input_height * 3, 8) * 8;
            for (int x = 0; x < input_width; x++) {
               unsigned n = x * line;
               for (int y = 0; y < input_height; y++)
                  for (int c = 0; c < 3; c++)
                     map[n++] = input_in[x][y][c];
               for (; n < (x + 1) * line;)
                  map[n++] = 0;
            }
         } else if (input_channels == 1) {
            unsigned n = 0;
            for (int x = 0; x < input_width; x++) {
               for (int y = 0; y < MAX2(input_height, FEATURE_ATOMIC_SIZE); y++) {
                  if (y < input_height)
                     map[n++] = input_in[x][y][0];
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
               for (int x = 0; x < input_width; x++) {
                  for (int y = 0; y < input_height; y++) {
                     for (int c = 0; c < 8; c++) {
                        unsigned input_channel = c + u * 8;
                        if (input_channel < input_channels)
                           map[n++] = input_in[x][y][input_channel] - 0x80;
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
            rkt_dump_buffer(map, "input", 0, 0, 0,
                            rkt_get_tensor(subgraph, input_idxs[i])->bo_size);

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

         /* RK3568 PC task-DMA mode (vendor rknpu_job): build the 40-byte
          * descriptor array so the PC unit walks the whole chain itself --
          * no CPU stepping between tasks, no ping-pong config races. */
         if (!getenv("RKT_NO_DESC") && task_count > 0) {
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
               d[ti].regcfg_amount = t->regcfg_amount - 1 +
                  (getenv("RKT_DESC_AM") ? atoi(getenv("RKT_DESC_AM")) : 0);
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

      struct rkt_operation *operation = find_producer(subgraph, output_idxs[i]);
      struct rkt_resource *output_tensor =
         rkt_get_tensor(subgraph, output_idxs[i]);
      struct pipe_transfer *transfer = NULL;
      uint8_t *raw_output;
      /* RK3568 DPU output layout (RE 2026-08-22, layer-c28 quadrant map):
       * surfaces of 8 channels, 8 bytes per pixel, surface stride
       * Wout * Hout * 8 -- matches the vendor DST_SURF_STRIDE (0x1880 =
       * 28*28*8 for the 28x28 probe).  The previous 16-channel unpack only
       * looked correct on channel-uniform fills. */
      uint8_t(*output_in)[operation->output_height][operation->output_width]
                         [8];
      uint8_t(*output_out)[operation->output_width][operation->output_channels];

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
         unsigned rows = operation->output_width;   /* dims[1] */
         unsigned cols = operation->output_height;  /* dims[2] */
         unsigned surf = rkt_surf_px(rows * cols) * 8;
         for (int oc = 0; oc < operation->output_channels; oc++) {
            unsigned g = oc / 8, c = oc % 8;
            for (unsigned y = 0; y < rows; y++) {
               for (unsigned x = 0; x < cols; x++) {
                  output_out[y][x][oc] =
                     raw[g * surf + (y * cols + x) * 8 + c] + 0x80;
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

   free(subgraph);
}
