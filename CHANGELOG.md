## [0.0.3] - 2025-02-08

- DeepSeek Llama distilled model tested and works.
- DeepSeek </think> overriding example added `example-deepseek.py`
- logits processors now get the next token from the sampler. This makes it less efficient to generate but only when using logits processors and allows for interesting experiments (such as suppressing </think> a certain number of times with DeepSeek.)
- Breaking: logits processor functions are now passed THREE arguments

## [0.0.2] - 2025-02-08

- Much better docs
- Added a realtime (True/False) flag for real-time output vs returning a string at the end

## [0.0.1] - 2025-02-08

- The initial release as a package
