# Arithmetic Coding Integration

## Goals
We want to introduce an `ArithmeticEncoder`/`ArithmeticDecoder` pair that can consume the per-token distributions produced by vLLM and emit a compact bitstream (encode) or drive generation deterministically from an existing bitstream (decode). The codec must be usable from the Python API, streaming outputs, and—longer term—the OpenAI-compatible HTTP surface. This document outlines the code changes needed and how the codec fits into the current sampling stack.

## Where To Integrate
1. **SamplingParams plumbing (`vllm/sampling_params.py`)** – add an `ArithmeticCodecParams` struct so callers can opt into encode/decode mode, provide I/O handles, and control precision. Similar to `StructuredOutputsParams`, the new struct will be validated by `Processor._validate_supported_sampling_params`.
2. **Metadata wiring (`vllm/v1/sample/metadata.py`, `vllm/v1/sample/logits_processor/state.py`)** – extend `SamplingMetadata` to carry codec flags, cumulative probability precision, and per-request persistent state so the sampler can reuse encoder buffers across engine steps.
3. **Sampler changes (`vllm/v1/sample/sampler.py`)** – insert an `ArithmeticSampler` sub-module that lives right after `apply_logits_processors` and before `TopKTopPSampler`. When arithmetic mode is on, `sample()` will delegate to the codec instead of greedy/random sampling and will return both the sampled token IDs and an encoded `torch.uint8` chunk.
4. **Output handling (`vllm/v1/engine/output_processor.py`, `vllm/outputs.py`)** – extend `RequestOutput` to surface the byte stream (e.g., a new `compressed_chunks: list[bytes]` field) and plumb the chunk list through streaming so clients can persist or transmit it.
5. **Request lifecycle (`vllm/v1/engine/processor.py`, `vllm/v1/engine/detokenizer.py`)** – ensure encode/decode requests cannot mix with multi-modal prompts unsupported by the codec, and reset codec state when a request finishes.

## New Modules
- `vllm/compression/arithmetic.py`: pure-PyTorch implementation of interval updates, including helper kernels to build cumulative distribution functions from logits on GPU. This module exposes `ArithmeticEncoderState`, `ArithmeticDecoderState`, and stateless `update_interval()` utilities compiled with `torch.compile`.
- `vllm/v1/sample/ops/arithmetic.py`: lightweight wrapper invoked by the sampler that maps batched logits into parallel arithmetic updates. This mirrors the structure of `topk_topp_sampler.py` so the sampler can switch between them by inspecting `sampling_metadata.codec_mode`.

## Encoding Path
1. **Parameterization** – the user sets `SamplingParams.codec={"mode": "encode", "precision_bits": 32, "sink": BinaryIO}`. `Processor._validate_supported_sampling_params` checks mutual exclusivity with nucleus/temperature sampling, structured outputs, and speculative decoding.
2. **Metadata build** – when `Processor` assembles `EngineCoreRequest`, it instantiates an `ArithmeticCodecState` (per sequence) with low/high intervals set to `[0, 1)` using integer arithmetic and records it inside the request’s sampling metadata.
3. **Sampler execution** – inside `Sampler.sample` we branch:  
   - Build normalized probabilities from the processed logits (`torch.log_softmax` + `torch.cumsum`) with the precision requested.  
   - Call `ArithmeticEncoder.update(state, probs, target_token_id)` for each request in the batch to shrink the interval and emit carry bits whenever the interval straddles a boundary.  
   - Collect emitted bytes into `SamplerOutput.codec_chunks` (a new field) alongside the usual `sampled_token_ids`.
4. **Output streaming** – `OutputProcessor` detects codec chunks in each `SamplerOutput` and pushes them into `RequestOutput.compressed_chunks`; the REST/gRPC layer forwards them immediately so a receiver can save the byte stream without waiting for the request to finish.

## Decoding Path
1. **Parameterization** – the user supplies `SamplingParams.codec={"mode": "decode", "source": BinaryIO, "expected_tokens": N}`. Validation ensures we have enough metadata (e.g., the decoder must know when to stop emitting tokens either via EOS, stop sequences, or `expected_tokens`).
2. **State seeding** – `Processor` loads the first machine word from the user-provided bitstream into `ArithmeticDecoderState.value`.
3. **Sampler behavior** – instead of sampling from logits, the sampler feeds the logits-derived cumulative probabilities plus the current `value` into `ArithmeticDecoder.decode_next(state, cdf)`, which returns the token index lying inside the encoded interval. The sampled token is appended to the sequence AND the decoder advances its `value` so subsequent steps are consistent.
4. **Stop criteria** – the standard finish reasons (EOS, length, stop strings) continue to live in `OutputProcessor`, ensuring decode requests integrate seamlessly with the scheduler.

## Interaction With Existing Processors
- `LogitsProcessor`s still run before arithmetic coding, so constraints such as min tokens, repetition penalties, or structured output masks are preserved.
- `TopK/TopP` and temperature are bypassed in arithmetic mode because the codec expects the full categorical distribution; Processor validation will enforce this (e.g., reject `top_p != 1`).
- For logprob collection, we can reuse the `LogprobsTensors` path because arithmetic encode/decode keeps track of the actual sampled token; we only need to ensure `raw_logprobs` are computed prior to codec normalization.

## Testing & Validation
1. Unit tests in `tests/compression/test_arithmetic.py` covering round-trip encode/decode for CPU and CUDA, different precisions, and edge cases (min/max token IDs, forced bad words, etc.).
2. Integration tests under `tests/samplers/test_arithmetic_sampler.py` to confirm scheduler batches can mix codec and non-codec requests safely.
3. End-to-end harness in `examples/compression/arithmetic_codec_demo.py` showing how to encode a prompt, transmit the bitstream, and decode it back into text with `LLMEngine`.

## Follow-Up
- Extend the OpenAI-compatible server (`vllm/entrypoints/openai/api_server.py`) to accept codec params over HTTP.
- Add monitoring hooks so `StatLoggerManager` can report average bits/token when the codec is active.
