# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm.compression import (
    ArithmeticCodecMode,
    ArithmeticCodecRuntimeState,
    decode_with_fixed_cdf,
    encode_with_fixed_cdf,
)
from vllm.compression.dataset_codec import _logprob_map_to_cdf
from vllm.logprobs import Logprob


def test_logprob_map_to_cdf_roundtrip():
    logprob_map = {
        0: Logprob(logprob=math.log(0.7)),
        1: Logprob(logprob=math.log(0.2)),
        2: Logprob(logprob=math.log(0.1)),
    }
    cdf = _logprob_map_to_cdf(logprob_map, vocab_size=3, precision_bits=12)
    sequence = [0, 1, 2, 0, 0, 1]

    encoded = encode_with_fixed_cdf(cdf, sequence, precision_bits=12)
    decoded = decode_with_fixed_cdf(cdf, len(sequence), precision_bits=12, bitstream=encoded)

    assert decoded == sequence
    assert len(encoded) > 0


def test_encode_helpers_work_with_codec_state():
    cdf = torch.tensor([0, 1024, 3072, 4096], dtype=torch.int64)
    bitstream = encode_with_fixed_cdf(cdf, [1, 0, 2], precision_bits=12)
    decoder = ArithmeticCodecRuntimeState(
        mode=ArithmeticCodecMode.DECODE,
        precision_bits=12,
        initial_bytes=bitstream,
    )
    tokens = [decoder.decode_token(cdf) for _ in range(3)]
    assert tokens == [1, 0, 2]
