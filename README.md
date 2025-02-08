# inferMLX - Easy LLM inference on macOS with MLX
*(including `infer.py` - LLM inference in one annotated Python file!)*

[![PyPI](https://img.shields.io/pypi/v/infermlx.svg)](https://pypi.org/project/infermlx/)

An easy library for doing LLM inference of Llama-based models on macOS, thanks to Apple's MLX project. **The main feature is being able to manipulate the inference process** for either educational or entertainment reasons.

## Basic installation and usage

```bash
pip install infermlx
```

```python
import infermlx.infer as infermlx
model = infermlx.Model.load_model()
model.generate("Tell me a joke")
```

> A man walked into a library and asked the librarian,
> "Do you have any books on Pavlov's dogs and Schrödinger's cat?"

Alternatively, if you don't want it to output in real-time to stdout:

```python
output, metadata = model.generate("Tell me a joke", temp=1.8, max_tokens=50, realtime=False)
print(output)
print(metadata.generation_tps)
```

Look at the `example-*.py` scripts for more ideas. You can also pass in parameters to `generate`: `system_prompt`, `max_tokens`, `seed`, `realtime` and `logits_processors`.

(`infer.py`, in the `infermlx` folder, is a *single* Python program with the fewest parts you need to do inference of Llama-compatible models on macOS. On a modern Mac with Python and the dependencies installed, `python infer.py --prompt 'Tell me a joke.'` should result in a cringeworthy joke.)

## Running `infer.py` directly

If you download the repo and want to play with `infer.py`, do `pip install -r requirements.txt` and run `infer.py` like so:

```bash
python -m infermlx.infer --prompt 'Tell me a joke.' 
# OR if you end up in the infermlx directory
python infer.py --prompt 'Tell me a joke.' 
```

It defaults to `unsloth/Llama-3.2-1B-Instruct` as it's free to use, quick, and only needs 4GB of RAM for inference. `mistralai/Mistral-7B-Instruct-v0.2` is a better model by far but needs >16GB of RAM free.

`infer.py` can be run with several options:

- **`--model`**  
  **Default:** `"unsloth/Llama-3.2-1B-Instruct"`  
  **Description:** The model to load for inference. Can be a HuggingFace repo or local directory.

- **`--prompt`, `-p`**  
  **Default:** `"Tell me a joke."`  
  **Description:** Your prompt, unsurprisingly.

- **`--system-prompt`, `-s`**  
  **Default:** None  
  **Description:** An optional system prompt, e.g. *"Talk like a pirate."*

- **`--max-tokens`, `-m`**  
  **Type:** `int`  
  **Default:** `1000`  
  **Description:** The maximum number of tokens to generate.

- **`--temp`**  
  **Type:** `float`  
  **Default:** `0.0`  
  **Description:** Temperature for sampling. As always, values yield more deterministic outputs, while higher values introduce more randomness. This is explained in more detail in the code.

- **`--seed`**  
  **Type:** `int`  
  **Default:** `42`
  **Description:** A random seed for the PRNG. Everyone in LLM-land seems to use 42 as a default because Ilya Sutskever did it in a demo once or something.

## Logits processors (the fun bit)

One way to have fun with LLMs, even small ones, is to 'force' them to say things. For example, if your model always refuses to answer a specific prompt, what happens if you force it to start its response with 'Yes, I am happy to help'? It depends. But you can do that!

```python
import infermlx.infer as infermlx

model = infermlx.Model.load_model()

token_stream = model.tokenizer.encode("Here's one about pirates.", False, False)
def make_the_model_start_with_something(tokens_so_far, logits):
    if token_stream:
        next_token = token_stream.pop(0)
        logits[:, next_token] = 2000
    return logits

model.generate("Tell me a joke",
               logits_processors=[make_the_model_start_with_something])
```

You could also give the model a 'tic' of sorts:

```python
import random
import infermlx.infer as infermlx

model = infermlx.Model.load_model("mistralai/Mistral-7B-Instruct-v0.2")

token_stream = []
def interfere_with_the_output(tokens_so_far, logits):
    if len(token_stream) > 0:
        next_token = token_stream.pop(0)
        logits[:, next_token] = 2000
    else:
      if random.random() < 0.08:
          token_stream.extend(model.tokenizer.encode("crap!", False, False))
    return logits

model.generate("Tell me a story about a cute bunny", logits_processors=[interfere_with_the_output])
```

Which yields something akin to this:

> Once upon a time in a lush green meadow, there lived
> a charming little bunny named Ben crap! I mean, Bunny.
> Bunny was known for his crap! I mean, his beautiful,
> crap! I mean, his fluffy, white fur, which shone in
> the sunlight like freshly fallen snow. He had
> twinkling, blue eyes that spark crap! I mean, shone
> with curiosity crap! I mean, curiosity and wonder.

Let your imagination run wild! (You could also detect certain things it has said and then force its ongoing response from there.)

## Motivation

When I saw [Apple's work on LLM inference tooling](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) I had fun using it, but realized if I wanted to *understand* inference I'd need to get my head into it and reimplement things.

`infer.py` (and then the overall `infermlx` package) is the result. I've annotated the source to provide guidance. Much of the code has been changed from Apple's original source (and the structure is totally different) but check out [the mlx_lm project](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) to see where it began and for more features.

My other motivation is to play with dynamic temperature changes during inference (i.e during 'thinking') and with logit stuffing (i.e. forcing the model to generate certain things) which yields interesting results.

For now, I will let you play. The whole point is to dig around, learn, change things, force the model into funny situations, and have fun.

## TODO

* Provide an option to not automatically print streamed output
* Add support for LoRA adapters to add fine tuning fun to the mix

## Credits

* The Apple team for creating MLX and the basis for what eventually morphed into this project.
* Everyone making and training models.