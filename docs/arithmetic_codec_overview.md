# Arithmetic Codec Implementation

## Runtime plumbing
- `SamplingParams` accepts `ArithmeticCodecParams` (see `vllm/sampling_params.py`). The struct records `mode`, `precision_bits` (16/24/32), and the optional `initial_state` byte stream used to seed decoding.
- When a request is created (`vllm/v1/request.py`), the sampling params are inspected and an `ArithmeticCodecRuntimeState` is attached to the request. The runtime state (defined in `vllm/compression/arithmetic.py`) owns either an `ArithmeticEncoderState` or an `ArithmeticDecoderState`.
- During batch preparation (`vllm/v1/worker/gpu_model_runner.py` and `gpu_input_batch.py`), the runtime state is passed to the sampler through `SamplingMetadata.codec_states` so that sampling can cooperate with the codec for each active request.

## Encoding path
1. After logits are post-processed, the sampler identifies rows that carry an arithmetic runtime state. (`vllm/v1/sample/sampler.py`)
2. `build_int_cdf()` (same file) runs on the exact logits that sampling used—after temperature scaling, argmax-invariant processors, and top-k/top-p masking—and converts the normalized distribution into an integer CDF. The tensor has length `vocab + 1` and its last element equals `2^precision_bits`.
3. For encode mode, the sampler calls `state.encode_token(cdf, token_id)` for the sampled token. Internally, `ArithmeticEncoderState.update()` shrinks the `[low, high)` interval, emits carry bits using `BitSink`, and returns any newly available bytes.
4. The byte chunks returned by the encoder are attached to the current `EngineCoreOutput`. The scheduler (`vllm/v1/core/sched/scheduler.py`) forwards them to the request state, and the output processor (`vllm/v1/engine/output_processor.py`) exposes them through `RequestOutput.codec_chunks`.
5. When a request finishes, the scheduler flushes the encoder via `finalize_encode()` so the user receives the final, padded bytes.

## Decoding path
1. The user provides the compressed bitstream via `ArithmeticCodecParams(initial_state=...)`. `ArithmeticDecoderState` seeds its `BitSource` with these bytes and pre-fills the current value register.
2. During sampling, the same integer CDF construction is performed. Instead of respecting the RNG sample, the sampler calls `state.decode_token(cdf)`.
3. `ArithmeticDecoderState.decode()` finds the symbol whose interval contains the scaled value, updates `[low, high)`, consumes bits from the stream, and returns the recovered token id.
4. The sampler overwrites the sampled token with the decoded token so the rest of the engine sees a deterministic sequence. No codec chunks are produced in decode mode.

## Streaming to the example
- `examples/compression/arithmetic_codec_qwen3_example.py` enables `RequestOutputKind.DELTA` so each `RequestOutput` only contains the newly generated token ids. It records those tokens, stitches together `codec_chunks`, and feeds the resulting byte stream back into a second run with `mode="decode"`.
- Because the runtime encoder/decoder share the same precision and CDF construction, the decoded tokens must exactly match the encoded sequence; a mismatch indicates either a bitstream truncation or the wrong sampling settings.
