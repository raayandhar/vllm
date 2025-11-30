# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example: Using Arithmetic Codec Compression with Qwen3-1.7B

This script demonstrates how to use arithmetic coding for compression-aware decoding
with the Qwen3-1.7B model. It shows both encoding (compressing token generation)
and decoding (reconstructing tokens from compressed bitstream).

Usage:
    python examples/compression/arithmetic_codec_qwen3_example.py \
        --model Qwen/Qwen3-1.7B \
        --prompt "What is the capital of France?" \
        --max-tokens 50
"""

import argparse
from typing import Optional

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.sampling_params import ArithmeticCodecParams, RequestOutputKind


def encode_and_decode(
    model: str,
    prompt: str,
    max_tokens: int = 50,
    precision_bits: int = 16,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    gpu_memory_utilization: float = 0.5,
) -> tuple[list[int], bytes, list[int]]:
    """
    Encode a prompt using arithmetic codec and then decode it back.
    
    Returns:
        tuple: (encoded_tokens, bitstream, decoded_tokens)
    """
    # Initialize engine
    engine_args = EngineArgs(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        enforce_eager=True,  # Faster for testing
    )
    engine = LLMEngine.from_engine_args(engine_args)
    
    print(f"\n{'='*70}")
    print("ENCODING PHASE - Generating tokens and compressing them")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}")
    print(f"Precision bits: {precision_bits}")
    print(f"Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}")
    
    # Step 1: Encode mode
    encode_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        output_kind=RequestOutputKind.DELTA,
        arithmetic_codec=ArithmeticCodecParams(
            mode="encode",
            precision_bits=precision_bits,
        ),
    )
    
    request_id = "encode_request"
    engine.add_request(request_id, prompt, encode_params)
    
    encoded_tokens = []
    all_codec_chunks = []
    
    # Process until completion
    step_count = 0
    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        step_count += 1
        
        for output in request_outputs:
            if output.request_id == request_id:
                # Collect tokens
                if output.outputs:
                    for seq_output in output.outputs:
                        if seq_output.token_ids:
                            encoded_tokens.extend(seq_output.token_ids)
                            if len(encoded_tokens) <= 10:
                                print(f"  Step {step_count}: Generated token {seq_output.token_ids[-1]}")
                
                # Collect codec chunks
                if output.codec_chunks:
                    for chunk in output.codec_chunks:
                        if chunk:
                            all_codec_chunks.append(chunk)
                
                if output.finished:
                    print(f"\n✓ Encoding completed in {step_count} steps")
    
    # Build complete bitstream
    bitstream = bytearray()
    for chunk in all_codec_chunks:
        bitstream.extend(chunk)
    
    print(f"\nEncoding Results:")
    print(f"  Generated tokens: {len(encoded_tokens)}")
    print(f"  Compressed size: {len(bitstream)} bytes")
    if len(bitstream) > 0:
        compression_ratio = (len(encoded_tokens) * 2) / len(bitstream)
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  (Assuming ~2 bytes per token on average)")
    else:
        print("  Warning: No compressed data collected")
        return encoded_tokens, bytes(bitstream), []
    
    # Step 2: Decode mode
    print(f"\n{'='*70}")
    print("DECODING PHASE - Reconstructing tokens from bitstream")
    print(f"{'='*70}")
    
    # Create a new engine instance for decoding
    decode_engine = LLMEngine.from_engine_args(engine_args)
    
    decode_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        output_kind=RequestOutputKind.DELTA,
        arithmetic_codec=ArithmeticCodecParams(
            mode="decode",
            precision_bits=precision_bits,
            initial_state=bytes(bitstream),
        ),
    )
    
    decode_request_id = "decode_request"
    decode_engine.add_request(decode_request_id, prompt, decode_params)
    
    decoded_tokens = []
    
    # Process until completion
    step_count = 0
    while decode_engine.has_unfinished_requests():
        request_outputs = decode_engine.step()
        step_count += 1
        
        for output in request_outputs:
            if output.request_id == decode_request_id:
                if output.outputs:
                    for seq_output in output.outputs:
                        if seq_output.token_ids:
                            decoded_tokens.extend(seq_output.token_ids)
                            if len(decoded_tokens) <= 10:
                                print(f"  Step {step_count}: Decoded token {seq_output.token_ids[-1]}")
                
                if output.finished:
                    print(f"\n✓ Decoding completed in {step_count} steps")
    
    print(f"\nDecoding Results:")
    print(f"  Decoded tokens: {len(decoded_tokens)}")
    
    return encoded_tokens, bytes(bitstream), decoded_tokens


def verify_roundtrip(
    encoded_tokens: list[int],
    decoded_tokens: list[int],
    verbose: bool = True,
) -> bool:
    """Verify that the roundtrip encoding/decoding worked correctly."""
    print(f"\n{'='*70}")
    print("VERIFICATION - Comparing encoded vs decoded tokens")
    print(f"{'='*70}")
    
    min_len = min(len(encoded_tokens), len(decoded_tokens))
    matches = sum(
        1 for i in range(min_len) if encoded_tokens[i] == decoded_tokens[i]
    )
    
    if verbose:
        print(f"Encoded tokens (first 20): {encoded_tokens[:20]}")
        print(f"Decoded tokens (first 20): {decoded_tokens[:20]}")
        print(f"\nToken-by-token comparison:")
        print(f"  Total tokens: {min_len}")
        print(f"  Matches: {matches}")
        print(f"  Mismatches: {min_len - matches}")
        
        if min_len > 0:
            match_rate = (matches / min_len) * 100
            print(f"  Match rate: {match_rate:.1f}%")
    
    # For deterministic arithmetic coding, tokens should match exactly
    if matches == min_len and len(encoded_tokens) == len(decoded_tokens):
        print("\n✓ Roundtrip verification PASSED - All tokens match!")
        return True
    elif matches > 0:
        print(f"\n⚠ Roundtrip verification PARTIAL - {matches}/{min_len} tokens match")
        return False
    else:
        print("\n✗ Roundtrip verification FAILED - No tokens match")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test arithmetic codec compression with Qwen3-1.7B"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name or path (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Input prompt (default: 'What is the capital of France?')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)",
    )
    parser.add_argument(
        "--precision-bits",
        type=int,
        default=16,
        choices=[16, 24, 32],
        help="Arithmetic codec precision bits (default: 16)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6, recommended for Qwen3)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling parameter (default: 20)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization (default: 0.5)",
    )
    
    args = parser.parse_args()
    
    try:
        encoded_tokens, bitstream, decoded_tokens = encode_and_decode(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            precision_bits=args.precision_bits,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        
        if decoded_tokens:
            success = verify_roundtrip(encoded_tokens, decoded_tokens)
            return 0 if success else 1
        else:
            print("\n✗ Decoding failed - no tokens decoded")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
