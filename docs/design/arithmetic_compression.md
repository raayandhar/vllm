# Teacher-Forced Top-K Probability Export for Arithmetic Coding

This note lays out how to add a first-class "compression mode" to vLLM so we can run teacher-forced passes over a dataset, pull the top-k (e.g., k=100) probability mass per token, renormalize it, and feed it into an arithmetic coder. The plan keeps us inside the normal vLLM scheduling/batching path, reuses GPU-side sampling logic, and exposes the data through the existing streaming infrastructure.

## Goals and constraints
- Input: tokenized dataset; no new tokens are sampled (teacher forcing).
- Output: for each next-token position, a small (k=100) set of token IDs with normalized probs matching the same top-k/top-p/temperature/logit-processor stack we would use for sampling.
- Efficiency: stay on GPU for logits filtering, avoid full-vocab transfers, stream results so we do not hold whole-corpus tensors in host memory, and work with chunked prefill.
- UX: requestable via Python/OpenAI server knobs; fits `SamplingParams` instead of a separate codepath.

## Existing building blocks
- `SamplingParams.prompt_logprobs`: already asks the engine to emit top-k logprobs for prompt tokens; uses `_get_prompt_logprobs_dict` in `vllm/v1/worker/gpu_model_runner.py`.
- `TopKTopPSampler` (`vllm/v1/sample/ops/topk_topp_sampler.py`) can return logits/logprobs *after* top-k/p/temperature when `logprobs_mode` is `processed_logits`/`processed_logprobs`.
- `SamplingParams` carries `top_k/top_p/temperature/logit_bias/allowed_token_ids/bad_words` etc., and `SamplingMetadata` is built in `vllm/v1/worker/gpu_input_batch.py`.
- Streaming/logprob plumbing: `LogprobsProcessor` in `vllm/v1/engine/logprobs.py` and `RequestOutput` in `vllm/outputs.py`.

## Gaps to close
1) Prefill-only requests (e.g., `max_tokens=0`) currently drop prompt logprobs because `_process_model_outputs` in `vllm/v1/core/sched/scheduler.py` only emits outputs when there are generated tokens/pooler data.  
2) Prompt logprobs are computed from raw logits without the sampling stack (temperature/top-k/p/logit processors), so they do not match the distribution we want to arithmetic-code.  
3) No compact, streaming-oriented container for "top-k after processing" probs; we currently materialize full logprob tensors then slice on CPU.  
4) OpenAI/Python surface does not expose a clear "compression mode" flag and top-k-after-processing output.

## Implementation plan (files and edits)

### 1) API surface (`SamplingParams`, request creation)
- File: `vllm/sampling_params.py`  
  - Add a boolean `compression_mode` (default False) and an int `compression_top_k` (default None -> fall back to `prompt_logprobs`/`logprobs`).  
  - Validation: if `compression_mode` is True, force `max_tokens=0` unless explicitly overridden; require `prompt_logprobs` to be set (or set it to `compression_top_k`). Disallow `n > 1` since we are not sampling.  
  - Document that `logprobs_mode` must be `processed_logits` or `processed_logprobs` to get renormalized top-k.
- File: `vllm/v1/engine/input_processor.py`  
  - When `compression_mode`, set `num_prompt_logprobs = compression_top_k or params.prompt_logprobs`, and auto-set `params.max_tokens = 0` if unset.  
  - Skip min_tokens enforcement and any decode-only validators for compression requests.
- File: `vllm/entrypoints/openai/serving_*` (`serving_completions.py` / `serving_responses.py`)  
  - Add request parsing for a new JSON flag (e.g., `"compression": true`) and `"compression_top_k"`; map to the new `SamplingParams` fields.  
  - Ensure logprobs_mode is settable via the OpenAI compatibility layer (propagate to `ModelConfig.logprobs_mode`).

### 2) Keep prompt-logprob outputs for prefill-only
- File: `vllm/v1/core/sched/scheduler.py` (`_process_model_outputs`)  
  - If `prompt_logprobs_dict` has entries and either `compression_mode` or `max_tokens==0`, emit an `EngineCoreOutput` even when `new_token_ids` is empty. This is required so prefill-only requests return data.  
  - Ensure `request.status` transitions to finished after the prefill chunk when `compression_mode` to avoid waiting for decode steps.
- File: `vllm/v1/engine/logprobs.py`  
  - Allow `LogprobsProcessor.pop_prompt_logprobs()` to be called even when no sampled tokens were produced (currently triggered only when outputs are emitted). No behavior change needed if scheduler emits an `EngineCoreOutput`, but add a comment/guard so compression requests work.

### 3) Compute *processed* top-k probs for teacher forcing
- File: `vllm/v1/worker/gpu_model_runner.py` (`_get_prompt_logprobs_dict`)  
  - Add a path gated by `compression_mode` (check `self.requests[req_id].sampling_params`) to run logits through the sampling stack:  
    - Build a lightweight `SamplingMetadata`-like struct for this req: temperature/top_k/top_p/allowed_token_ids/logit_bias/bad_words/logits processors.  
    - Reuse `Sampler.apply_logits_processors`, `Sampler.apply_temperature`, and `TopKTopPSampler.apply_top_k_top_p` to mutate logits in place.  
    - If `logprobs_mode` is `processed_logits`, compute `logits.log_softmax` after filtering; if `processed_logprobs`, let `TopKTopPSampler` return renormalized logprobs.  
    - Use `gather_logprobs` with `num_prompt_logprobs` to pull exactly k entries (sampled token + top-(k-1) others) from the processed logits.  
  - Keep the existing raw-logits path for non-compression requests.
- File: `vllm/v1/sample/sampler.py`  
  - Factor out a helper (e.g., `compute_processed_logprobs_for_targets(logits, sampling_metadata, target_ids)`) to share logic between decode sampling and compression prompt logprobs. This keeps temperature/logits-processor/top-k code in one place.
- File: `vllm/v1/sample/ops/topk_topp_sampler.py`  
  - No functional change; document that `logprobs_mode` controls whether returned logprobs are post-top-k/p and renormalized, which compression relies on.

### 4) Stream a compact payload
- File: `vllm/logprobs.py`  
  - Add a small container (e.g., `TopKProbs` with `token_ids: list[int]`, `logprobs: list[float]`, `ranks: list[int]`) that can be reused for both prompt and decode. Optionally wrap it in a `FlatTopKProbs` similar to `FlatLogprobs` to keep GC overhead low.  
  - Make `append_logprobs_for_next_position` accept an optional flag `already_processed=True` to skip re-ranking when the input is already truncated/renormalized.
- File: `vllm/outputs.py`  
  - Extend `RequestOutput` with an optional `compression` field (list of per-position `TopKProbs`), or reuse `prompt_logprobs` if we guarantee it is now "processed" when `compression_mode`. Document the contract in the docstring.  
  - Ensure `RequestOutput.add()` merges compression chunks when streaming with `RequestOutputKind.DELTA`.
- File: `vllm/v1/engine/output_processor.py`  
  - When `compression_mode`, populate `request_output.prompt_logprobs` (or the new `compression` field) even if `outputs` is empty; skip detokenization if not requested.
- File: `vllm/entrypoints/openai/serving_responses.py`  
  - For JSON responses, add a `compression.top_logprobs` array mirroring OpenAI's logprob shape (`token`, `logprob`, `bytes`, `rank`). For streaming, emit the same in the delta event so the arithmetic coder can consume incrementally.

### 5) Scheduler/accounting
- File: `vllm/v1/request.py` (and related accounting in scheduler)  
  - Add a flag on `Request` (e.g., `is_compression`) derived from `SamplingParams.compression_mode`.  
  - On enqueue, set `num_output_placeholders` so chunked prefill still allocates KV blocks but the decode phase is skipped.  
  - Ensure finished-state bookkeeping releases KV blocks immediately after the prefill chunk.

### 6) Python helper for dataset compression
- Add a small utility (e.g., `tools/compress_dataset.py`) that:  
  - Tokenizes the dataset, chunks into prompts (respecting `block_size`/`max_model_len`).  
  - Issues `SamplingParams(compression_mode=True, compression_top_k=100, top_k=100, temperature=1.0, logprobs_mode="processed_logprobs", max_tokens=0, detokenize=False, output_kind=RequestOutputKind.DELTA)` requests via `AsyncLLM`.  
  - Streams prompt logprobs, renormalizes on the client (if needed), and feeds them to the arithmetic coder.

## End-to-end flow after the change
1) Client builds a request with `compression_mode=True`, `compression_top_k=100`, `top_k=100`, `logprobs_mode="processed_logprobs"`, `max_tokens=0`.  
2) `InputProcessor` validates, converts `compression_top_k` -> `prompt_logprobs`, and tags the request.  
3) Scheduler batches prefill as usual.  
4) On GPU, `_get_prompt_logprobs_dict` runs logits through temperature/logit processors + top-k, computes renormalized logprobs, and writes only the top-k slice to CPU.  
5) Scheduler emits an `EngineCoreOutput` even with `new_token_ids=[]`; `LogprobsProcessor` drops the prompt logprob chunk into `RequestOutput`.  
6) OpenAI/Python layer streams per-position top-k logprobs; the client renormalizes if desired and passes them to the arithmetic coder.  
7) Request is marked finished immediately after prefill, freeing KV blocks.

## Testing/validation
- Unit: add coverage for `_get_prompt_logprobs_dict` to confirm processed vs raw paths and that returned probs sum to ~1 after exponentiation.  
- Scheduler: regression test that `compression_mode` requests finish with zero decoded tokens but non-empty prompt logprobs.  
- API: OpenAI JSON/streaming responses include the compression payload; backward compatibility when `compression_mode=False`.  
- Perf: benchmark throughput vs baseline prompt-logprob path on long prompts; ensure no unexpected CPU-GPU syncs (profile `torch.cuda.synchronize` counts).  
- Correctness: round-trip small corpora through the arithmetic coder/decoder using the emitted distributions.

## Open questions / choices
- Do we expose renormalized logprobs in the API or raw logprobs + a flag? Proposal: processed logprobs, plus a `distribution_type: "topk-renorm"` marker in the response.  
- Is `compression_mode` allowed to coexist with `logprobs` for sampled tokens? Simplest: forbid decode in compression mode (max_tokens==0).  
- Client-side renorm vs GPU renorm: with `logprobs_mode="processed_logprobs"`, renorm happens on GPU; keep a raw-logits fallback for debugging.  
- Storage format: for large corpora we may want an on-disk binary stream (token_id, topk_ids, logprobs) instead of JSON; the proposed helper can write that without changing the engine.
