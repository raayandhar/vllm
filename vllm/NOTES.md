# ARITHMETIC CODING IN vLLM
Hand-written notes from Raayan.
## How to do arithmetic coding with language models
We are interested in the problem of *offline* compression using language models: the model is not updated, and our dataset we are interested in compressing is our "prompt". We do not do any generation, and the true data is the input at every step.

We have a "teacher-forcing" setup in the sense that we always condition on the true data prefix, and not on anything that the model might have "generated". In fact, it should not be doing any sampling / generation of any kind. 

## Review of compression and arithmetic coding
...

## Efficient compression pipeline in vLLM
### Key files
- sampling_params.py: