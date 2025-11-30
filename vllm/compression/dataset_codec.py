# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utilities for arithmetic coding over fixed datasets.

The helpers here use prompt logprobs (teacher forcing) to encode a known
token stream with an arithmetic coder, and reuse the existing decode-side
sampler to reconstruct the exact token sequence from the bitstream.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from vllm.compression import ArithmeticCodecMode, ArithmeticCodecRuntimeState
from vllm.inputs import TokensPrompt
from vllm.logprobs import Logprob, PromptLogprobs
from vllm.sampling_params import ArithmeticCodecParams, RequestOutputKind, SamplingParams


@dataclass(slots=True)
class DatasetCompressionResult:
    """Container for a compressed dataset payload."""

    bitstream: bytes
    token_count: int
    precision_bits: int
    prefix_token_ids: list[int]
    added_eos: bool


def _logprob_map_to_cdf(
    logprob_map: dict[int, Logprob],
    vocab_size: int,
    precision_bits: int,
) -> torch.Tensor:
    """Convert a {token_id -> logprob} map into an integer CDF.

    The arithmetic codec expects cumulative counts in token-id order.
    """
    if len(logprob_map) != vocab_size:
        raise ValueError(
            f"Expected {vocab_size} logprobs, got {len(logprob_map)}. "
            "Set `prompt_logprobs=-1` so the full vocabulary is returned."
        )

    scale = 1 << precision_bits
    dense_logprobs = torch.full(
        (vocab_size,), float("-inf"), dtype=torch.float64, device="cpu"
    )
    token_ids = torch.tensor(
        list(logprob_map.keys()), dtype=torch.long, device="cpu"
    )
    logprob_values = torch.tensor(
        [lp.logprob for lp in logprob_map.values()],
        dtype=torch.float64,
        device="cpu",
    )
    dense_logprobs[token_ids] = logprob_values

    # Normalize explicitly to guard against numerical drift.
    probs = dense_logprobs.exp()
    total = probs.sum()
    if not torch.isfinite(total) or total == 0:
        raise ValueError("Invalid logprob set; probabilities do not sum to 1.")
    probs = probs / total

    cdf = torch.cumsum(probs, dim=-1)
    cdf = torch.clamp((cdf * scale).floor().to(torch.int64), max=scale)
    cdf = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device="cpu"),
            cdf,
        ]
    )
    cdf[-1] = scale
    return cdf


class DatasetArithmeticCodec:
    """Dataset encoder/decoder built on top of vLLM arithmetic coding."""

    def __init__(
        self,
        llm,
        *,
        precision_bits: int = 16,
        chunk_size: int | None = None,
        context_size: int | None = None,
    ) -> None:
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()
        self.model_config = llm.llm_engine.model_config
        self.precision_bits = precision_bits
        self.vocab_size = self.model_config.get_vocab_size()
        max_model_len = self.model_config.max_model_len
        self.context_size = (
            context_size if context_size is not None else max_model_len - 1
        )
        self.chunk_size = chunk_size if chunk_size is not None else self.context_size

    def compress_text(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> DatasetCompressionResult:
        """Compress arbitrary text as an arithmetic-coded bitstream."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        prefix: list[int] = []
        if add_bos and self.tokenizer.bos_token_id is not None:
            prefix.append(self.tokenizer.bos_token_id)
        if add_eos and self.tokenizer.eos_token_id is not None:
            tokens = tokens + [self.tokenizer.eos_token_id]
        all_tokens = prefix + tokens

        bitstream = self._encode_tokens(all_tokens, len(prefix))
        return DatasetCompressionResult(
            bitstream=bitstream,
            token_count=len(tokens),
            precision_bits=self.precision_bits,
            prefix_token_ids=prefix,
            added_eos=add_eos and self.tokenizer.eos_token_id is not None,
        )

    def decompress_to_text(self, payload: DatasetCompressionResult) -> str:
        """Decode a compressed payload back to text."""
        prompt_tokens = payload.prefix_token_ids or [
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id or 0
        ]
        decode_params = SamplingParams(
            max_tokens=payload.token_count + (1 if payload.added_eos else 0),
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
            arithmetic_codec=ArithmeticCodecParams(
                mode="decode",
                precision_bits=payload.precision_bits,
                initial_state=payload.bitstream,
            ),
        )
        output = self.llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_tokens)],
            sampling_params=decode_params,
            use_tqdm=False,
        )[0]
        if not output.outputs:
            raise RuntimeError("Decode request returned no outputs.")
        token_ids: Sequence[int] = output.outputs[0].token_ids
        decoded = list(prompt_tokens) + list(token_ids)
        decoded = decoded[len(payload.prefix_token_ids) :]  # drop primer
        if payload.added_eos and decoded and decoded[-1] == self.tokenizer.eos_token_id:
            decoded = decoded[:-1]
        return self.tokenizer.decode(decoded)

    def _encode_tokens(self, token_ids: Sequence[int], prefix_len: int) -> bytes:
        state = ArithmeticCodecRuntimeState(
            mode=ArithmeticCodecMode.ENCODE, precision_bits=self.precision_bits
        )
        chunks: list[bytes] = []

        # Slide a bounded context window over the token stream so each request
        # stays within the model's max length while still using teacher forcing.
        cursor = prefix_len
        while cursor < len(token_ids):
            prompt_start = max(0, cursor - self.context_size)
            chunk_end = min(cursor + self.chunk_size - 1, len(token_ids) - 1)
            prompt_tokens = token_ids[prompt_start : chunk_end + 1]
            prompt_logprobs = self._prompt_logprobs(prompt_tokens)

            start_offset = cursor - prompt_start
            for local_idx in range(start_offset, len(prompt_tokens)):
                logprob_map = prompt_logprobs[local_idx]
                if logprob_map is None:
                    raise ValueError("Missing logprobs for a prompt position.")
                cdf = _logprob_map_to_cdf(
                    logprob_map, self.vocab_size, self.precision_bits
                )
                tok = prompt_tokens[local_idx]
                chunk = state.encode_token(cdf, tok)
                if chunk:
                    chunks.append(chunk)
            cursor = chunk_end + 1

        final_chunk = state.finalize_encode()
        if final_chunk:
            chunks.append(final_chunk)
        return b"".join(chunks)

    def _prompt_logprobs(self, prompt_tokens: Sequence[int]) -> PromptLogprobs:
        """Request full prompt logprobs for a token slice."""
        params = SamplingParams(
            max_tokens=0,
            prompt_logprobs=-1,
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        output = self.llm.generate(
            [TokensPrompt(prompt_token_ids=list(prompt_tokens))],
            sampling_params=params,
            use_tqdm=False,
        )[0]
        if output.prompt_logprobs is None:
            raise RuntimeError("prompt_logprobs were not returned; check params.")
        # max_tokens=0 avoids generation; prompt_logprobs[0] is always None.
        return output.prompt_logprobs


def encode_with_fixed_cdf(
    cdf: torch.Tensor, sequence: Iterable[int], precision_bits: int
) -> bytes:
    """Helper used in tests: encode a sequence with a fixed CDF."""
    state = ArithmeticCodecRuntimeState(
        mode=ArithmeticCodecMode.ENCODE, precision_bits=precision_bits
    )
    chunks = []
    for token in sequence:
        chunk = state.encode_token(cdf, int(token))
        if chunk:
            chunks.append(chunk)
    final_chunk = state.finalize_encode()
    if final_chunk:
        chunks.append(final_chunk)
    return b"".join(chunks)


def decode_with_fixed_cdf(
    cdf: torch.Tensor, num_tokens: int, precision_bits: int, bitstream: bytes
) -> list[int]:
    """Helper used in tests: decode tokens using a fixed CDF."""
    state = ArithmeticCodecRuntimeState(
        mode=ArithmeticCodecMode.DECODE,
        precision_bits=precision_bits,
        initial_bytes=bitstream,
    )
    return [state.decode_token(cdf) for _ in range(num_tokens)]
