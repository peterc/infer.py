#!/usr/bin/env python3

#
# infer.py
#
# You can use this file as a library but for experimenting
# you can edit and run it directly!
# 
# $ python infer.py --prompt 'Tell me a joke.' 
#
# This runs Llama 3.2 1B by default as it's small and almost
# anyone can run it (~3GB free RAM needed).
#
# Will work with Llama-compatible models, including the small Mistral ones.
# mistralai/Mistral-7B-Instruct-v0.2 is a very good one
# with this script and surprisingly smart (but needs ~15GB free RAM).
#
# $ python infer.py --prompt 'Tell me a joke.' --model 'mistralai/Mistral-7B-Instruct-v0.2' --temp 0
#
# See the end of this file for the fun you can have with
# logits processors for interfering with the model's output!
#
# -----
# Original source copyright © 2023-2024 Apple Inc.
# as at https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm
#
# Amendments copyright © 2025 Peter Cooper
# MIT-licensed (original and amendments)

#   ▐█ [○_○] █▌  ▐█ [○_○] █▌  ▐█ [○_○] █▌  
# <=|█  \/\/  █|==|█  \/\/  █|==|█  \/\/  █|=>
#   |█[[]][][]█|  |█[[]][][]█|  |█[[]][][]█|
#   ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀

# LET THE TOUR BEGIN!
# ===================
#
# Boring housekeeping stuff first. Imports, some data classes,
# then fall straight into the math-heavy transformer part.
# The most 'interesting' stuff for newcomers is actually in the
# bottom half of the file, so feel free to read from the bottom up
# instead!

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# MLX is doing the heavy lifting on macOS rather than
# PyTorch or whatever you might be used to elsewhere.
import mlx.core as mx
import mlx.nn as nn

# Let's get the most boring class out of the way first.
# These model arguments come in from the model's config.json file
# and are used all over the place here
@dataclass
class ModelArgs:
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})
    
# Another boring class that bundles together a bunch of data
# that's useful for monitoring progress and output
@dataclass
class GenerationResponse:
    text: str
    token: int
    logprobs: mx.array
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None

# Now for the incredibly geeky 'heart' of all the math I still don't quite understand!
# Basically the transformer block is the core of the model's magic
# and uses the attention mechanism to do its thing.
#
# TransformerBlock /---- Attention
#                  \---- MLP

# The transformer block is essentially the workflow
# for bringing together the attention mechanism and the MLP
# and why it's quite simple.
class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None):
        x_norm = self.input_layernorm(x)
        attn_out = self.self_attn(x_norm, mask, cache)
        h = x + attn_out
        mlp_out = self.mlp(self.post_attention_layernorm(h))
        return h + mlp_out
    
# Attention figures out which parts of the input are important
# and their relationships to each other.
class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim ** -0.5
        bias = args.attention_bias
        # If you've heard of 'query', 'key' and 'value' in the context of attention
        # then q, k, and v here are literally those.
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(args.hidden_size, args.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=bias)
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.rope = nn.RoPE(self.head_dim, traditional=args.rope_traditional, base=args.rope_theta, scale=1.0)

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        q = self.rope(q, offset=cache.offset)
        k = self.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

# Once we've figured out what's important with 'Attention' (above),
# the MLP (multi-layer perceptron) processes that info
# and kinda ties it together. I still need to read more about this TBH
# as I can't quite grok it.
class MLP(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        dim, hidden_dim = args.hidden_size, args.intermediate_size
        bias = args.mlp_bias
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        up = self.up_proj(x)
        return self.down_proj(nn.silu(self.gate_proj(x)) * up)

# A simple detokenizer that keeps track of tokens and decodes
# them when needed. You could arguably do this on the fly.
class NaiveStreamingDetokenizer:
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self._tokenizer = tokenizer
        self.reset()

    def reset(self) -> None:
        self.offset = 0
        self.tokens = []
        self._text = ""
        self._current_tokens = []

    def add_token(self, token: int) -> None:
        self._current_tokens.append(token)
        self.tokens.append(token)

    def finalize(self) -> None:
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens.clear()

    @property
    def text(self) -> str:
        current = self._tokenizer.decode(self._current_tokens)
        if self._tokenizer.clean_up_tokenization_spaces and current.endswith(" "):
            current = current[:-1]
        if current.endswith("\n"):
            self._text += current
            self._current_tokens.clear()
            current = ""
        return self._text + current

    @property
    def last_segment(self) -> str:
        segment = self.text[self.offset:]
        self.offset = len(self.text)
        return segment
    
# A wrapper that keeps track of the tokenizer and detokenizer
# so 'token stuff' is all in a one stop shop, of sorts..
class TokenizerWrapper:
    def __init__(self, tokenizer, detokenizer_class=NaiveStreamingDetokenizer, eos_token_ids=None):
        super().__setattr__("_tokenizer", tokenizer)
        super().__setattr__("_detokenizer", detokenizer_class(tokenizer))
        super().__setattr__("_eos_token_ids", set(eos_token_ids) if eos_token_ids is not None else {tokenizer.eos_token_id})

    def add_eos_token(self, token):
        try:
            token_id = int(token)
        except ValueError:
            token_id = self._tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            raise ValueError(f"'{token}' is not a token for this tokenizer")
        self._eos_token_ids.add(token_id)

    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self._detokenizer
        if attr == "eos_token_ids":
            return self._eos_token_ids
        return getattr(self._tokenizer, attr)

    def __setattr__(self, attr, value):
        if attr in {"detokenizer", "eos_token_ids"}:
            raise AttributeError(f"Cannot set {attr}.")
        if attr.startswith("_"):
            super().__setattr__(attr, value)
        else:
            setattr(self._tokenizer, attr, value)

# The cache for the key and value pairs that are used in the attention mechanism
# so you don't need to recompute over and over and over and over...
# I find it a bit opaque, but it works.
class KVCache:
    def __init__(self, max_size=4096) -> None:
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size

    def _expand_cache(self, keys, values):
        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        new_size = max(self.keys.shape[2] * 2 if self.keys is not None else self.max_size, self.offset + keys.shape[2])
        
        pad_k = mx.zeros((B, n_kv_heads, new_size, k_head_dim), keys.dtype)
        pad_v = mx.zeros((B, n_kv_heads, new_size, v_head_dim), values.dtype)
        
        if self.keys is not None:
            pad_k[..., :self.offset, :] = self.keys
            pad_v[..., :self.offset, :] = self.values

        self.keys, self.values = pad_k, pad_v

    def update_and_fetch(self, keys, values):
        n_new = keys.shape[2]

        if self.keys is None or self.offset + n_new > self.keys.shape[2]:
            self._expand_cache(keys, values)

        self.keys[..., self.offset:self.offset + n_new, :] = keys
        self.values[..., self.offset:self.offset + n_new, :] = values
        self.offset += n_new

        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]

# The model itself is a wrapper around the underlying model
# which ends up stored in here..
class UnderlyingModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        # Note: This is where the transformer blocks are created and the
        # whole Transformer/MLP/Attention stuff from earlier gets involved
        self.layers = [TransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, mask=None, cache=None) -> mx.array:
        h = self.embed_tokens(inputs)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)
        return self.norm(h)

# An organizational class that represents a model, wraps the model
# loads the model, finds the model - you get the idea.
# The 'generate' method is the one that actually starts the magic trick.
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = UnderlyingModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    # When the model object is 'called' directly, it calls the underlying model instead
    # Makes code easier to read (allegedly!)
    def __call__(self, inputs, mask=None, cache=None):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    # Gets the prompt parsing and token generation going
    # and prints out the results as we go
    def generate(self, prompt, system_prompt = None, seed = None, temp = 0.0, **kwargs):
        # Set a random seed for the PRNG
        if seed:
            mx.random.seed(seed)

        if temp > 0.0:
            # If temperature is set, we use sampling.
            # All the logits are scaled inversely by the temperature
            # and then we sample from the resulting distribution
            # So higher temperature equals more randomness as the logit
            # values are scaled lower and thus the distribution is more spread out
            sampler = (lambda logits: mx.random.categorical(logits * (1 / temp)))
        else:
            # Temperature of zero means we always get the 'highest'/most likely token
            # or 'greedy sampling' as ML types seem to call it
            # Fun idea: Change to argmin to get the least likely tokens (and a pile of gibberish)
            sampler = (lambda logits: mx.argmax(logits, axis=-1))

        # Compile the model's chat template along with system prompt, if present
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

        for response in self.stream_generate(prompt_tokens, sampler = sampler, **kwargs):
            print(response.text, end="", flush=True)

        # Now we're done, we can print out stats about the run
        print("\n\n", file=sys.stderr)
        print(f"    Prompt: {response.prompt_tokens} tok, {response.prompt_tps:.3f} tok/s", file=sys.stderr)
        print(f"Generation: {response.generation_tokens} tok, {response.generation_tps:.3f} tok/s", file=sys.stderr)
        print(f"  Peak mem: {response.peak_memory:.1f}GB", file=sys.stderr)

    # So there's a chain of generators..
    # generate -> stream_generate -> generate_step
    # stream_generate tracks the timing and when to stop
    def stream_generate(self, prompt, **kwargs):
        tokenizer = self.tokenizer
        prompt = mx.array(prompt)
        detokenizer = tokenizer.detokenizer
        token_generator = self.generate_step(prompt, **kwargs)

        detokenizer.reset()
        tic = time.perf_counter()
        last_resp = None
        for n, (token, logprobs) in enumerate(token_generator):
            if n == 0:
                # On the first token GENERATED, we can calculate the TPS for the *prompt* processing
                prompt_time = time.perf_counter() - tic
                prompt_tps = prompt.size / prompt_time
                # We then reset the timer to calculate the TPS for the generation phase instead
                tic = time.perf_counter()
            finish_reason = None

            # If we get any of the model's 'end of sequence' token, we stop
            if token in tokenizer.eos_token_ids:
                finish_reason = "stop"
            else:
                detokenizer.add_token(token)
            last_resp = GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.metal.get_peak_memory() / 1e9,
                finish_reason=finish_reason,
            )
            yield last_resp
            if finish_reason:
                break
        else:
            if last_resp is not None:
                last_resp.finish_reason = "length"
        detokenizer.finalize()

    # This is where actual tokens get generated
    # Well.. after we process the prompt tokens, anyway
    def generate_step(self, prompt_tokens, max_tokens = 256,
                      sampler = None,
                      logits_processors = None,
                      prefill_step_size = 1024):
        # Some naming niceties for ease of reading and brevity, respectively
        model = self
        ts = prompt_tokens

        # Let prompt processing and token generation run on the same GPU stream
        generation_stream = mx.new_stream(mx.default_device())

        # We keep a store of tokens for logit processors to use
        token_store = []

        # We keep a cache for each layer of the model
        prompt_cache = [KVCache() for _ in range(len(model.layers))]

        def _step(tokens):
            # The 'core' of generating each new token!
            with mx.stream(generation_stream):
                # Run the underlying model to get the logits for the next token
                # (this ends up going via __call__ in Model through to UnderlyingModel)
                logits = model(tokens[None], cache=prompt_cache)
                logits = logits[:, -1, :]

                # This block is just convenience for the logit processors
                # which will want to know the tokens that have been generated
                if logits_processors:
                    nonlocal token_store
                    token_store.append(tokens)
                    for processor in logits_processors:
                        logits = processor(token_store, logits)

                # We use our sampler to get our winning token!
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                y_next = sampler(logprobs)
                return y_next, logprobs.squeeze(0)

        # Prompt processing phase where we process prompt tokens in chunks to update the cache.
        with mx.stream(generation_stream):
            while ts.size > prefill_step_size:
                model(ts[:prefill_step_size][None], cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                ts = ts[prefill_step_size:]
                mx.metal.clear_cache()
            
        # Let's get the ball rolling with generating our first token!
        ts, logprobs = _step(ts)

        # Our main generation loop, generating one token at a time
        for n in range(max_tokens):
            next_ts, next_logprobs = _step(ts)
            # We need to evaluate the cache to update the state
            mx.async_eval(next_ts, next_logprobs)

            # Pass the token and logprobs back up the chain..
            yield ts.item(), logprobs

            if n % 512 == 0:
                mx.metal.clear_cache()

            ts, logprobs = next_ts, next_logprobs

    @property
    def layers(self):
        return self.model.layers

    @staticmethod
    def get_model_path(path_or_huggingface):
        # Because we might get a real path OR a HuggingFace model name
        # we need to check if it's a real path or not
        # and then get HF to download the model if it's not
        model_path = Path(path_or_huggingface)
        if not model_path.exists():
            try:
                model_path = Path(snapshot_download(
                    path_or_huggingface,
                    allow_patterns=["*.json", "*.safetensors", "*.py", "tokenizer.model", "*.tiktoken", "*.txt"]
                ))
            except Exception:
                print(f"Model not found: {path_or_huggingface}.", file=sys.stderr)
                sys.exit(1)
        return model_path

    @classmethod
    def load_model(cls, model_path):
        # First we need to get the model path (see above)
        model_path = cls.get_model_path(model_path)

        # Now read in the model's config and get the model set up
        with (model_path / "config.json").open("r") as f:
            config = json.load(f)
        model = cls(ModelArgs.from_dict(config))

        # Load the model's weights - they're quite important
        weights = {}
        for wf in model_path.glob("model*.safetensors"):
            weights.update(mx.load(str(wf)))
        model.load_weights(list(weights.items()))

        # Load the tokenizer and detokenizer into a handy wrapper
        # and let it know what the models 'end of sequence' token is
        # otherwise we won't know when to stop generating text.
        model.tokenizer = TokenizerWrapper(
            AutoTokenizer.from_pretrained(model_path),
            NaiveStreamingDetokenizer,
            eos_token_ids=[config.get("eos_token_id")]
        )

        return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--prompt", "-p", default="Tell me a joke.")
    parser.add_argument("--max-tokens", "-m", type=int, default=1000)
    parser.add_argument("--system-prompt", "-s", default=None)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Just two steps to get things going from an external POV
    # Load the model and then run generate
    model = Model.load_model(args.model)
    
    # Kick the whole thing off
    # Note that kwargs are kicked all the way down from here to the depths of the
    # token generation process.. ideal for passing in logits processors, etc.
    model.generate(args.prompt, system_prompt=args.system_prompt, temp=args.temp, max_tokens=args.max_tokens, seed=args.seed)

    # Here's two fun examples of how to use a logits processor
    # ========================================================
    #
    # Here's how to force the model to start with a specific sequence of tokens!
    # You'd uncomment all of this and comment out the model.generate line above.
    #
    #token_stream = model.tokenizer.encode("I refuse to answer that because", False, False)
    #def make_the_model_start_with_something(tokens_so_far, logits):
    #    nonlocal token_stream
    #    if token_stream:
    #        next_token = token_stream.pop(0)
    #        logits[:, next_token] = 2000
    #    return logits
    #
    #model.generate(args.prompt, max_tokens=args.max_tokens, sampler=sampler, logits_processors=[make_the_model_start_with_something])

    # Here's a slightly more evil example of how to make the model randomly
    # be forced to say something every now and then..
    #
    # token_stream = []
    # import random
    # def make_the_model_weird(tokens_so_far, logits):
    #     nonlocal token_stream
    #     if len(token_stream) > 0:
    #         next_token = token_stream.pop(0)
    #         logits[:, next_token] = 2000
    #     else:
    #       if random.random() < 0.08:
    #           token_stream.extend(model.tokenizer.encode("What the??", False, False))

    #     return logits
    
    # model.generate(args.prompt, max_tokens=args.max_tokens, sampler=sampler, logits_processors=[make_the_model_weird])

if __name__ == "__main__":
    main()
