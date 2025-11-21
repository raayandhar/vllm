# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import torch


class ArithmeticCodecMode(str, Enum):
    DISABLED = "disabled"
    ENCODE = "encode"
    DECODE = "decode"


class BitSink:
    """Accumulates bits and exposes byte chunks when 8 bits are available."""

    def __init__(self) -> None:
        self._byte = 0
        self._filled = 0
        self._buffer = bytearray()

    def write_bit(self, bit: int) -> None:
        self._byte = (self._byte << 1) | (bit & 1)
        self._filled += 1
        if self._filled == 8:
            self._buffer.append(self._byte & 0xFF)
            self._byte = 0
            self._filled = 0

    def drain(self) -> bytes:
        if not self._buffer:
            return b""
        chunk = bytes(self._buffer)
        self._buffer.clear()
        return chunk

    def finalize(self) -> bytes:
        """Flush residual bits (padded with zeros)."""
        if self._filled > 0:
            self._byte <<= 8 - self._filled
            self._buffer.append(self._byte & 0xFF)
            self._byte = 0
            self._filled = 0
        return self.drain()


class BitSource:
    """Reads bits from a byte stream, returning zeros when out of data."""

    def __init__(self, initial_bytes: bytes | bytearray | None = None) -> None:
        self._buffer = bytearray(initial_bytes or b"")
        self._byte_index = 0
        self._bit_index = 0

    def append(self, data: bytes | bytearray) -> None:
        if data:
            self._buffer.extend(data)

    def read_bit(self) -> int:
        if self._byte_index >= len(self._buffer):
            return 0
        current = self._buffer[self._byte_index]
        bit = (current >> (7 - self._bit_index)) & 1
        self._bit_index += 1
        if self._bit_index == 8:
            self._bit_index = 0
            self._byte_index += 1
        return bit


@dataclass
class ArithmeticEncoderState:
    precision_bits: int
    low: int = field(init=False)
    high: int = field(init=False)
    pending_bits: int = field(default=0, init=False)
    sink: BitSink = field(default_factory=BitSink)
    finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.low = 0
        self.high = (1 << self.precision_bits) - 1

    @property
    def _half(self) -> int:
        return 1 << (self.precision_bits - 1)

    @property
    def _quarter(self) -> int:
        return 1 << (self.precision_bits - 2)

    @property
    def _three_quarter(self) -> int:
        return self._half + self._quarter

    @property
    def scale(self) -> int:
        return 1 << self.precision_bits

    def _emit_bit(self, bit: int) -> None:
        self.sink.write_bit(bit)
        while self.pending_bits > 0:
            self.sink.write_bit(1 - bit)
            self.pending_bits -= 1

    def update(self, low_count: int, high_count: int) -> None:
        if self.finished:
            raise RuntimeError("Encoder already finalized.")
        range_ = self.high - self.low + 1
        total = self.scale
        self.high = self.low + (range_ * high_count // total) - 1
        self.low = self.low + (range_ * low_count // total)

        while True:
            if self.high < self._half:
                self._emit_bit(0)
            elif self.low >= self._half:
                self._emit_bit(1)
                self.low -= self._half
                self.high -= self._half
            elif self.low >= self._quarter and self.high < self._three_quarter:
                self.pending_bits += 1
                self.low -= self._quarter
                self.high -= self._quarter
            else:
                break
            self.low = self.low * 2
            self.high = self.high * 2 + 1

    def finalize(self) -> bytes:
        if self.finished:
            return b""
        self.pending_bits += 1
        if self.low < self._quarter:
            self._emit_bit(0)
        else:
            self._emit_bit(1)
        self.finished = True
        return self.sink.finalize()

    def drain(self) -> bytes:
        return self.sink.drain()


@dataclass
class ArithmeticDecoderState:
    precision_bits: int
    source: BitSource = field(default_factory=BitSource)
    low: int = field(init=False)
    high: int = field(init=False)
    value: int = field(init=False)

    def __post_init__(self) -> None:
        self.low = 0
        self.high = (1 << self.precision_bits) - 1
        self.value = 0
        for _ in range(self.precision_bits):
            self.value = (self.value << 1) | self.source.read_bit()

    @property
    def _half(self) -> int:
        return 1 << (self.precision_bits - 1)

    @property
    def _quarter(self) -> int:
        return 1 << (self.precision_bits - 2)

    @property
    def _three_quarter(self) -> int:
        return self._half + self._quarter

    @property
    def scale(self) -> int:
        return 1 << self.precision_bits

    def append(self, data: bytes | bytearray) -> None:
        self.source.append(data)

    def decode(self, cdf: torch.Tensor) -> int:
        """Decode symbol using integer cdf with shape [vocab + 1]."""
        total = self.scale
        range_ = self.high - self.low + 1
        scaled_value = ((self.value - self.low + 1) * total - 1) // range_
        # cdf is 1D tensor, first element is 0, last is total.
        # We need highest index where cdf[idx] <= scaled_value < cdf[idx+1]
        hi = torch.searchsorted(cdf[1:], scaled_value, right=False)
        token = int(hi.item())

        low_count = int(cdf[token].item())
        high_count = int(cdf[token + 1].item())
        self.high = self.low + (range_ * high_count // total) - 1
        self.low = self.low + (range_ * low_count // total)

        while True:
            if self.high < self._half:
                pass
            elif self.low >= self._half:
                self.low -= self._half
                self.high -= self._half
                self.value -= self._half
            elif self.low >= self._quarter and self.high < self._three_quarter:
                self.low -= self._quarter
                self.high -= self._quarter
                self.value -= self._quarter
            else:
                break
            self.low = self.low * 2
            self.high = self.high * 2 + 1
            self.value = (self.value * 2) + self.source.read_bit()

        return token


@dataclass
class ArithmeticCodecRuntimeState:
    mode: ArithmeticCodecMode
    precision_bits: int
    initial_bytes: bytes | bytearray | None = None
    encoder: ArithmeticEncoderState | None = None
    decoder: ArithmeticDecoderState | None = None

    def __post_init__(self) -> None:
        if self.mode == ArithmeticCodecMode.ENCODE:
            self.encoder = ArithmeticEncoderState(self.precision_bits)
        elif self.mode == ArithmeticCodecMode.DECODE:
            source = BitSource(self.initial_bytes)
            self.decoder = ArithmeticDecoderState(
                self.precision_bits, source=source
            )

    @property
    def active(self) -> bool:
        return self.mode != ArithmeticCodecMode.DISABLED

    def append_decoder_bytes(self, data: bytes | bytearray) -> None:
        if self.decoder:
            self.decoder.append(data)

    def encode_token(self, cdf: torch.Tensor, token_id: int) -> bytes:
        assert self.encoder is not None, "Encoder state missing."
        low = int(cdf[token_id].item())
        high = int(cdf[token_id + 1].item())
        self.encoder.update(low, high)
        return self.encoder.drain()

    def finalize_encode(self) -> bytes:
        if self.encoder:
            return self.encoder.finalize()
        return b""

    def decode_token(self, cdf: torch.Tensor) -> int:
        assert self.decoder is not None, "Decoder state missing."
        return self.decoder.decode(cdf)


def build_int_cdf(
    logits: torch.Tensor,
    precision_bits: int,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Return integer-valued cumulative distribution function [vocab + 1]."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    logits = logits.clone()
    device = logits.device
    if top_k is not None:
        top_k_tensor = torch.tensor([top_k], device=device, dtype=torch.int32)
    else:
        top_k_tensor = None
    if top_p is not None:
        top_p_tensor = torch.tensor([top_p], device=device, dtype=torch.float32)
    else:
        top_p_tensor = None
    logits = apply_top_k_top_p(
        logits.unsqueeze(0), top_k_tensor, top_p_tensor  # type: ignore[arg-type]
    ).squeeze(0)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float64)
    scale = 1 << precision_bits
    cdf = torch.cumsum(probs, dim=-1)
    cdf = torch.clamp(
        (cdf * scale).floor().to(torch.int64), min=0, max=scale - 1
    )
    cdf = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=cdf.device),
            cdf,
        ]
    )
    cdf[-1] = scale
    return cdf.to("cpu")
