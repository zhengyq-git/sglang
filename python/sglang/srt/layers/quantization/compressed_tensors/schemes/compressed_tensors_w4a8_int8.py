# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A8 (packed INT4 static-symmetric weights + INT8 dynamic activations)
linear scheme for platforms without a dedicated dense W4A8 fused kernel.

Numerics
--------
The activation path is a real A8 path: activations are quantized to INT8 per
token (``per_token_quant_int8``) and immediately dequantized back to the
compute dtype, so the GEMM sees the A8 rounding error instead of the full
BF16 activation. The W4 weights are unpacked from the compressed-tensors
``pack_to_int32`` layout and scaled per group/channel into the compute dtype.

Performance
-----------
This is a correctness-first fallback: there is no fused W4A8 GEMM, the GEMM
itself runs on the generic compute-dtype path. Expect BF16-GEMM level
performance plus the dequantization overhead, not the throughput of a real
W4A8 kernel.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)

__all__ = ["CompressedTensorsW4A8Int8", "unpack_int32_to_int8"]

logger = logging.getLogger(__name__)


def _normalize_strategy(strategy) -> Optional[str]:
    """compressed-tensors hands over either a plain string ("token") or a
    ``QuantizationStrategy`` enum; normalize both to the bare string value."""
    if strategy is None:
        return None
    return str(strategy).rsplit(".", 1)[-1].lower().strip()

# compressed-tensors `pack-quantized` stores `32 // num_bits` values per int32.
NUM_BITS = 4
PACK_FACTOR = 32 // NUM_BITS
NIBBLE_MASK = (1 << NUM_BITS) - 1
# Symmetric int4 checkpoints store unsigned-offset nibbles; shift back to signed.
INT4_OFFSET = 1 << (NUM_BITS - 1)


def unpack_int32_to_int8(
    weight_packed: torch.Tensor, num_bits: int = NUM_BITS
) -> torch.Tensor:
    """Unpack compressed-tensors ``pack_to_int32`` weights into int8 values.

    ``[..., K // pack_factor]`` int32 -> ``[..., K]`` int8, unpacking the low
    nibble first (little-endian nibble order, matching the CUTLASS repack used
    by the W4AFP8 MoE scheme) and shifting the unsigned-offset values back into
    the signed range via ``- 2 ** (num_bits - 1)``.
    """
    if num_bits != NUM_BITS:
        raise NotImplementedError(
            f"CompressedTensorsW4A8Int8 only handles {NUM_BITS}-bit packing, "
            f"got num_bits={num_bits}."
        )

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    offset = 1 << (num_bits - 1)

    shifts = (
        torch.arange(pack_factor, device=weight_packed.device, dtype=torch.int32)
        * num_bits
    )
    # The right shift is arithmetic on int32, but `& mask` keeps only the
    # requested nibble, so the sign extension is discarded.
    unpacked = (weight_packed.unsqueeze(-1) >> shifts) & mask
    unpacked = (unpacked - offset).to(torch.int8)
    return unpacked.flatten(-2).contiguous()


class CompressedTensorsW4A8Int8(CompressedTensorsLinearScheme):
    """INT4 packed weights (channel or group) + INT8 dynamic activations."""

    def __init__(
        self,
        strategy: str,
        group_size: Optional[int] = None,
        symmetric: bool = True,
        activation_strategy: Optional[str] = QuantizationStrategy.TOKEN.value,
        params_dtype: torch.dtype = torch.bfloat16,
    ):
        if strategy not in (
            QuantizationStrategy.CHANNEL.value,
            QuantizationStrategy.GROUP.value,
        ):
            raise ValueError(
                "CompressedTensorsW4A8Int8 requires channel or group weight "
                f"quantization, got strategy={strategy}."
            )
        if not symmetric:
            raise NotImplementedError(
                "CompressedTensorsW4A8Int8 currently supports symmetric weights "
                "only (checkpoints carrying a weight zero-point are not handled)."
            )

        self.strategy = _normalize_strategy(strategy)
        self.group_size = -1 if group_size is None else group_size
        self.symmetric = symmetric
        self.activation_strategy = _normalize_strategy(activation_strategy)
        self.params_dtype = params_dtype

    @classmethod
    def get_min_capability(cls) -> int:
        # No dedicated kernel: plain torch dequant + generic GEMM, so there is
        # no device-capability requirement (unlike the CUTLASS W4A8 path).
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        if input_size_per_partition % PACK_FACTOR != 0:
            raise ValueError(
                "Packed INT4 weights need the input dim to be a multiple of "
                f"{PACK_FACTOR}, got {input_size_per_partition}."
            )

        # WEIGHT: compressed-tensors stores int4 packed along the input dim.
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // PACK_FACTOR,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            packed_factor=PACK_FACTOR,
            packed_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL.value:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition, 1, dtype=params_dtype
                ),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.group_size > 0, (
                "Group strategy requires a positive group_size, got "
                f"{self.group_size}."
            )
            if input_size_per_partition % self.group_size != 0:
                raise ValueError(
                    f"input_size_per_partition ({input_size_per_partition}) must "
                    f"be divisible by group_size ({self.group_size})."
                )
            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // self.group_size,
                    dtype=params_dtype,
                ),
                output_dim=0,
                input_dim=1,
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)

        # Original (unpacked) weight shape, used to drop the packing padding of
        # the input dim. Checkpoints in pack-quantized format always carry it.
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_shape", weight_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Idempotent: some runners call this more than once per layer.
        if getattr(layer, "is_w4a8_dequantized", False):
            return

        dtype = self.params_dtype
        weight_packed = layer.weight_packed.data
        weight_scale = layer.weight_scale.data

        # int32 packed -> signed int4 values, [out, K_pad]
        weight_int = unpack_int32_to_int8(weight_packed)
        k_pad = weight_int.shape[1]

        # Broadcast the per-channel / per-group scale onto the unpacked K dim.
        if self.strategy == QuantizationStrategy.GROUP.value:
            weight_scale = weight_scale.repeat_interleave(self.group_size, dim=1)
        if weight_scale.shape[1] == 1:
            weight_scale = weight_scale.expand(-1, k_pad)
        weight_scale = weight_scale[:, :k_pad]

        # Dequantize on the fly in fp32 to avoid bf16 accumulation error.
        weight = weight_int.to(torch.float32) * weight_scale.to(torch.float32)

        # Drop the padding that packed int4 may have introduced on the K dim.
        real_k = self._real_input_size(layer, k_pad)
        if real_k is not None and real_k < k_pad:
            weight = weight[:, :real_k]

        layer.weight = Parameter(weight.to(dtype).contiguous(), requires_grad=False)
        layer.is_w4a8_dequantized = True

        # The packed representation is dead weight after dequantization; drop it
        # so it does not double the checkpoint's footprint on the device.
        if hasattr(layer, "weight_packed"):
            delattr(layer, "weight_packed")
        if hasattr(layer, "weight_shape"):
            delattr(layer, "weight_shape")

    @staticmethod
    def _real_input_size(layer: torch.nn.Module, k_pad: int) -> Optional[int]:
        """Real input size from the checkpoint's ``weight_shape``, if usable.

        Returns ``None`` when the tensor is missing or does not describe this
        partition (e.g. it stores the unsharded K under tensor parallelism),
        in which case the unpacked size is kept as is.
        """
        weight_shape = getattr(layer, "weight_shape", None)
        if weight_shape is None or weight_shape.numel() != 2:
            return None
        try:
            real_k = int(weight_shape[1].item())
        except (RuntimeError, ValueError, TypeError):
            return None
        if 0 < real_k <= k_pad:
            return real_k
        return None

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        weight = layer.weight
        dtype = weight.dtype

        if not x.is_contiguous():
            x = x.contiguous()

        if self.activation_strategy == QuantizationStrategy.TOKEN.value:
            # A8 path: per-token int8 quantization, then dequantized back to the
            # compute dtype because the generic GEMM has no int8 input support.
            x_q, x_scale = per_token_quant_int8(x)
        else:
            # Per-tensor dynamic fallback (configs without a token strategy).
            absmax = x.abs().max().clamp(min=1e-10)
            x_scale = (absmax / 127.0).to(torch.float32)
            x_q = torch.round(x.float() / x_scale).clamp_(-128.0, 127.0).to(torch.int8)

        x_dequant = (x_q.to(torch.float32) * x_scale.to(torch.float32)).to(dtype)
        return torch.nn.functional.linear(x_dequant, weight, bias)
