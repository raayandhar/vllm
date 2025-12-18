# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test compression mode (teacher-forced top-k probability export).

This tests Parts 1 & 2 of the compression mode implementation:
- API surface (SamplingParams validation)
- Prefill-only requests (no generation, immediate finish)
- Prompt logprobs returned with empty outputs

Run with: pytest tests/v1/test_compression_mode.py -v
"""

import pytest

from vllm import SamplingParams


class TestCompressionModeValidation:
    """Test SamplingParams validation for compression mode."""

    def test_compression_mode_sets_max_tokens_zero(self):
        """Compression mode should force max_tokens=0."""
        params = SamplingParams(
            compression_mode=True,
            compression_top_k=100,
        )
        assert params.max_tokens == 0
        assert params.compression_mode is True
        assert params.prompt_logprobs == 100  # Set from compression_top_k

    def test_compression_mode_with_explicit_prompt_logprobs(self):
        """Can set prompt_logprobs directly instead of compression_top_k."""
        params = SamplingParams(
            compression_mode=True,
            prompt_logprobs=50,
        )
        assert params.max_tokens == 0
        assert params.prompt_logprobs == 50

    def test_compression_mode_requires_prompt_logprobs(self):
        """Compression mode requires either prompt_logprobs or compression_top_k."""
        with pytest.raises(ValueError, match="prompt_logprobs must be set"):
            SamplingParams(compression_mode=True)

    def test_compression_mode_n_must_be_one(self):
        """Compression mode doesn't support n > 1 (no sampling)."""
        with pytest.raises(ValueError, match="n must be 1"):
            SamplingParams(
                compression_mode=True,
                compression_top_k=100,
                n=2,
            )

    def test_compression_top_k_validation(self):
        """compression_top_k must be >= 1 or -1."""
        with pytest.raises(ValueError, match="compression_top_k must be at least 1"):
            SamplingParams(
                compression_mode=True,
                compression_top_k=0,
            )

    def test_compression_top_k_minus_one_valid(self):
        """compression_top_k=-1 means all vocab (valid)."""
        params = SamplingParams(
            compression_mode=True,
            compression_top_k=-1,
        )
        assert params.prompt_logprobs == -1


class TestCompressionModeDisabled:
    """Test that normal requests are unaffected."""

    def test_normal_request_unchanged(self):
        """Non-compression requests should work as before."""
        params = SamplingParams(
            max_tokens=100,
            temperature=0.7,
        )
        assert params.compression_mode is False
        assert params.max_tokens == 100

    def test_prompt_logprobs_without_compression(self):
        """prompt_logprobs should work without compression mode."""
        params = SamplingParams(
            max_tokens=100,
            prompt_logprobs=10,
        )
        assert params.compression_mode is False
        assert params.prompt_logprobs == 10


# Integration test (requires a running model)
# @pytest.mark.skip(reason="Requires model - run manually")
class TestCompressionModeIntegration:
    """Integration tests for compression mode with actual model."""

    def test_compression_mode_returns_prompt_logprobs(self):
        """Test that compression mode returns prompt logprobs."""
        from vllm import LLM

        llm = LLM(model="Qwen/Qwen2.5-3B-Instruct", enforce_eager=True)

        params = SamplingParams(
            compression_mode=True,
            compression_top_k=10,
        )

        outputs = llm.generate(["Hello, world!"], sampling_params=params)

        assert len(outputs) == 1
        output = outputs[0]

        # Should have prompt logprobs
        assert output.prompt_logprobs is not None
        assert len(output.prompt_logprobs) > 0

        # Should have no generated tokens (prefill only)
        assert len(output.outputs) == 1
        assert len(output.outputs[0].token_ids) == 0

        # Should be finished
        assert output.finished


if __name__ == "__main__":
    # Run unit tests
    pytest.main([__file__, "-v", "-k", "not Integration"])

