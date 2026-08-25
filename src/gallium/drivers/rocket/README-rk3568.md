# rocket on RK3568 — a map for the next reader

This branch makes the Mesa `rocket` gallium driver (Teflon/TFLite
delegate) drive the single-core RK3568 NPU.  The RK3588 path it started
from is MR !42134; everything RK3568-specific was derived by comparing
our command streams byte-for-byte against captured vendor (librknnrt)
streams.  This file records what is *not* obvious from the code: the
data flow, the memory conventions, and the list of things that look
arbitrary but are load-bearing.  "Test N" refers to the author's lab
log (RE-LOG.md, in Russian, kept with https://github.com/iav/rk3568-npu);
the register names follow `registers.xml`.

## 1. Data flow

```
teflon (TFLite delegate)
  └─ rkt_ml_operation_supported()           rkt_ml.c   what we take
  └─ rkt_ml_subgraph_create()               rkt_ml.c
       ├─ lower_*()                          one rkt_operation per NPU op:
       │    lower_convolution      conv / depthwise (+fused add, +relu)
       │    lower_pooling          avg pool -> depthwise conv, const weights
       │    lower_max_pooling      max/avg pool -> PPU chunk (pool_avg)
       │    lower_fully_connected  FC -> 1x1 conv over a 1x1 map
       │    lower_identity_copy    1x1 conv with unit weights (concat legs,
       │                           add hosts, pool-first carriers)
       ├─ rkt_split_tasks()                 rkt_task.c   bands x channel groups
       ├─ rkt_fill_weights / rkt_fill_biases rkt_coefs.c  packed weights, BS stream
       ├─ compile_operation()               rkt_ml.c     one regcmd BO per op
       │    rkt_fill_regcmd()               rkt_regcmd.c conv task (CNA/CMAC/CORE/DPU)
       │    rkt_fill_ppu_regcmd()           rkt_regcmd.c PPU chunk, appended to the
       │                                                 PREVIOUS task's BO
       └─ PC chaining: every task's 4-word tail (PC_BASE_ADDRESS,
          PC_REGISTER_AMOUNTS, op-enable, broadcast) is patched with the
          next task's address/amount, across operations too.  The whole
          graph is ONE drm_rocket_job; the PC unit walks the chain and
          raises a single interrupt (last_int_mask, uapi extension).
  └─ rkt_ml_subgraph_invoke()               rkt_ml.c
       packs the input into 8-channel surfaces, submits, waits
  └─ rkt_ml_subgraph_read_outputs()         unpacks the output surfaces
```

BO lists of the job: `in` = tensors with no producer inside the
partition, `out` = every operation's output.  Intermediate tensors go in
NEITHER list and a handle must not appear twice — a duplicate or an
in+out overlap wedges the scheduler silently (Test 41, 59).

## 2. Memory conventions

- **Axes**: width = `dims[2]` (the contiguous memory row), height =
  `dims[1]`.  The RK3588 path mixed them; that was self-consistent only
  on square maps and corrupted every non-square one (Test 68).
- **Feature surfaces**: 8 channels x 8 bytes per pixel, planar per
  surface.  Line stride = W*8 bytes, surface stride = align(W*H, 4)*8
  bytes (the align-4 is what killed the 7x7 flakiness: 49 -> 52,
  Test 57).  CNA/DPU stride registers are in 8-byte units (16 on
  RK3588).  A band task that starts mid-tensor addresses its slice
  through `RDMA_SURF_NOTCH` = surface stride − W*H_band*8 (Test 55).
- **First layer (C=3)**: packed ARGB, one 32-bit pixel, CVT shifts by
  −128 (Test 46).
- **Weights** (`rkt_fill_weights`): 16-kernel groups, tap-major inside
  a group, 32-channel input slices with compact tails for partial
  slices and partial kernel groups (Test 43, 44).  Depthwise: 32-channel
  groups, tap rows strided by align16 (Test 66).
- **int8**: the NPU only ever sees uint8.  Teflon shifts int8 zero
  points by 128 and the delegate XORs the data with 0x80 (Test 71).
  Biases stay int32.

## 3. Requantization — the DPU BS stream

Per output channel the DPU reads an 8-byte record from the BS stream:
`[bias int32][ow int16][mul int16]`.

- per-tensor weights: bias-only stream (`BS_CFG 0x158`), `ow` and the
  multiplier come from registers;
- per-channel weights: full stream (`BS_CFG 0x148`, `BS_MUL_CFG 0xe01`,
  OW source = stream): `ow = 0x80 − wzp`, `mul = round(2^14 *
  s_wc / s_wmax)` renormalizes each channel to the layer scale that
  `OUT_CVT` runs off (Test 71);
- `OUT_CVT` scale is a 15-bit float (exp + 14-bit mantissa with an
  implicit leading 1).  Rounding may carry into the exponent — a scale a
  hair under a power of two otherwise ends up at half value (Test 72).
- `DPU_RDMA 0x5024` (not in the XML) = BS stream length in 8-byte words
  minus one; without it the bias RDMA fetches nothing (Test 71).
- fused add: the second operand comes through the EW stage with its own
  scale (`EW_CFG`, `EW_CVT_*`); a depthwise host is never used for a
  fused add (Test 72).

## 4. Things that look arbitrary but are load-bearing

Each of these cost hours; none is derivable from the TRM.  Change them
only with a vendor capture to compare against.

| what | why | test |
|---|---|---|
| `CNA_DMA_CON0 = 0x00171c07` written verbatim | bit 20 is RESERVED in the XML; assembling from fields drops it and the MAC then emits a constant | 29/30 |
| one broadcast op-enable word starts all units (`0x0041…`/`0x0081…` tail) | per-unit enables race the ping-pong register banks; the broadcast is the vendor's cure for the phase bug | 42 |
| PPU chunk lives in the BO of the PRECEDING task | the PC fetches the pool words from the same descriptor run; a separate BO never signals | 60/61 |
| `POOLING_PADDING_CFG 0x6040` carries all four paddings | a pooling window not covered by input+padding starves the PPU forever and stalls the whole chain | 61 |
| CBUF entry count from the DECLARED channels (align 32) | the bank wraps otherwise; showed up as the mp56 tail | 65 |
| PPU `CUBE_IN` clipped to the rows/columns actually consumed | the window starves on the unclipped cube | 65 |
| depthwise with C<32 declares a full 32-group | with the real 16 announced the pipeline produces nothing | 66 |
| identity copy for a depthwise leg only when C%32==0 | a depthwise copy with C<32 writes padding surfaces over the neighbouring concat slice | 64/69 |
| `FEATURE_GRAINS = stride+kh + stride*((Wout<28)+(Wout<14))` | the plain value starts the MAC before the first weight group is in CBUF on narrow maps | 53 |
| `CORE` block map one slot lower than the XML | the XML has a spurious `MAC_GATING` at 0x300c; `CMAC MISC_CFG` is 0x200c | 28/29 |
| `KERNEL_GROUP` = 0 always | the RK3588 formula underflows on depthwise | 30/31 |
| pad value in the upper half of `PAD_CON0` | `PAD_CON1` is never written by the vendor | 30/31 |
| kernel: core reset + IOMMU re-arm with `ENABLE_PAGING=0` on the wedge | the ping-pong bank state machine sticks after a timeout; a re-arm with paging on hangs the SoC | 40 |
| kernel: GFP_DMA32 for page tables and BOs | the NPU addresses 32 bits; no `mem=4G` needed | 58 |
| MAC clock from SCMI (`scmi_rate`), not the CRU clock | the CRU `npu` clock is decorative; the PVTPLL rate is what the MAC runs at | 67 |

## 5. How to debug

Knobs (all environment variables, read by the delegate):

| knob | effect |
|---|---|
| `RKT_DUMP=1` | print every emitted regcmd word to stderr — feed `tools/rknn_diff.py` against a vendor `.rknn` |
| `RKT_WDUMP=<file>` | dump the packed weight buffer |
| `RKT_NO_CHAIN=1` | one job per operation instead of one per graph |
| `RKT_NO_PPU=1` | pooling through the depthwise lowering / CPU |
| `RKT_MULSH=<n>` | shift the OUT_CVT scale by n — separates scale errors from addressing errors |
| `RKT_MAXROWS=<n>` | cap the band height — forces multi-band pipelines on small probes |
| `RKT_POOL_INT_MASK=<m>` | interrupt mask of the last PPU task |
| `ROCKET_DEBUG=dump_bos` | dump every BO (weights, biases, regcmd, tensors) to files |
| `TEFLON_MAX_OPS` / `TEFLON_MIN_OPS` | delegate a window of graph nodes |

Method that worked, in order of cost:

1. `tests/suite.py <probe.tflite>` — seven input patterns.  Uniform
   fills separate arithmetic from addressing (a wrong stride shows on
   gradients only); channel-coded inputs show which surface went
   where.  A probe at max ≤ 1 LSB is "clean".
2. When a probe is wrong, build the smallest geometric series that
   flips it (rect vs square, C=16/24/28/32, stride 1/2) and read the
   diff map — which rows/channels are hit — before theorizing.
3. `RKT_DUMP` + `tools/rknn_diff.py` against the vendor stream of the
   same layer (`tools/rknn_regcmd.py` extracts it from a `.rknn`).
   Registers that differ are the suspects; registers the vendor writes
   and we don't are the usual culprits.
4. For a vendor reference without the vendor stack running:
   `capture/capture.c` (LD_PRELOAD on the vendor runtime) records the
   regcmd + BOs, `tests/replay.c` replays them through rocket.  A
   vendor stream replayed through our kernel path must match EXACTLY;
   if it does, the environment is clean and the bug is in our stream.
5. Offline `.rknn` probes need rknn-toolkit2 on x86 (qemu-user or a
   cloud x86 box); the generator scripts are in the author's notes.

## 6. Kernel module

`module/` in https://github.com/iav/rk3568-npu is the in-tree
`drivers/accel/rocket` (7.2-rc7) plus: RK3568 compatible and variant
data (dma_bits, pc_data_amount_scale), GFP_DMA32 BOs, the wedge
recovery (core reset + manual IOMMU re-arm in `hw_submit`), the PC
task-DMA descriptor submit with a single final interrupt
(`last_int_mask` in `struct drm_rocket_job` — this branch and the
module must be built from matching revisions), the `scmi_rate`
parameter (800 MHz default).  The IOMMU page-table DMA32 patch is in
the kernel itself (`rockchip-iommu.c`, see the release .debs).
