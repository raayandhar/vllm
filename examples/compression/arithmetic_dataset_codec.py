# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Dataset compression using arithmetic coding and vLLM.

Unlike the generation-focused demo, this script encodes an existing text
corpus with teacher forcing: for each token in the dataset we build the full
next-token distribution, shrink the arithmetic interval, and emit bits. The
decoder reads the bitstream back through the arithmetic sampler to recover
the exact token sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vllm import LLM
from vllm.compression import DatasetArithmeticCodec, DatasetCompressionResult


def _save_metadata(payload: DatasetCompressionResult, meta_path: Path) -> None:
    meta = {
        "token_count": payload.token_count,
        "precision_bits": payload.precision_bits,
        "prefix_token_ids": payload.prefix_token_ids,
        "added_eos": payload.added_eos,
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def _load_metadata(bitstream_path: Path, meta_path: Path) -> DatasetCompressionResult:
    meta: dict[str, Any] = json.loads(meta_path.read_text())
    return DatasetCompressionResult(
        bitstream=bitstream_path.read_bytes(),
        token_count=meta["token_count"],
        precision_bits=meta["precision_bits"],
        prefix_token_ids=meta["prefix_token_ids"],
        added_eos=meta["added_eos"],
    )


def compress(args: argparse.Namespace) -> tuple[DatasetArithmeticCodec, DatasetCompressionResult]:
    model = LLM(
        model=args.model,
        tokenizer=args.model if args.tokenizer is None else args.tokenizer,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    codec = DatasetArithmeticCodec(
        model,
        precision_bits=args.precision_bits,
        chunk_size=args.chunk_size,
        context_size=args.context_size,
    )

    text = Path(args.input).read_text()
    payload = codec.compress_text(
        text,
        add_bos=not args.skip_bos,
        add_eos=not args.skip_eos,
    )
    output_path = Path(args.output)
    output_path.write_bytes(payload.bitstream)
    _save_metadata(payload, output_path.with_suffix(output_path.suffix + ".json"))

    bits = len(payload.bitstream) * 8
    bpt = bits / payload.token_count if payload.token_count else 0.0
    print(f"Wrote {len(payload.bitstream)} bytes ({bits:.0f} bits, {bpt:.3f} bits/token)")
    return codec, payload


def decompress(
    codec: DatasetArithmeticCodec,
    payload: DatasetCompressionResult,
    out_path: Path | None,
    original: str | None = None,
) -> None:
    restored = codec.decompress_to_text(payload)
    if out_path:
        out_path.write_text(restored)
        print(f"Decoded text written to {out_path}")
    if original is not None:
        status = "MATCH" if restored == original else "MISMATCH"
        print(f"Verification: {status}")
        if status == "MISMATCH":
            raise RuntimeError("Decoded text does not match original input.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arithmetic-code a text dataset with vLLM.")
    parser.add_argument("--model", required=True, help="Model name or path.")
    parser.add_argument("--tokenizer", help="Optional tokenizer override.")
    parser.add_argument(
        "--input",
        help="Path to the text to compress. Ignored if --text is provided.",
    )
    parser.add_argument(
        "--text",
        help="Inline text to compress (e.g., a prompt string).",
    )
    parser.add_argument("--output", required=True, help="Where to store the compressed bitstream.")
    parser.add_argument(
        "--precision-bits",
        type=int,
        default=16,
        choices=[16, 24, 32],
        help="Arithmetic coder precision.",
    )
    parser.add_argument("--chunk-size", type=int, help="Tokens to encode per request.")
    parser.add_argument("--context-size", type=int, help="Context window passed to the model.")
    parser.add_argument("--max-model-len", type=int, help="Override model context length.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--skip-bos", action="store_true", help="Do not prepend a BOS token.")
    parser.add_argument("--skip-eos", action="store_true", help="Do not append an EOS token.")
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Immediately decode the bitstream after compressing.",
    )
    parser.add_argument(
        "--decompress-output",
        help="Optional path to write decompressed text. Defaults to stdout-less verification only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.text and not args.input:
        raise SystemExit("Provide either --text or --input.")
    if args.text:
        tmp_path = Path(args.output).with_suffix(".tmp_input.txt")
        tmp_path.write_text(args.text)
        args.input = str(tmp_path)

    codec, payload = compress(args)
    original_text = Path(args.input).read_text()

    if args.decompress:
        decompress(
            codec=codec,
            payload=payload,
            out_path=Path(args.decompress_output) if args.decompress_output else None,
            original=original_text,
        )


if __name__ == "__main__":
    main()
