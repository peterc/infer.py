# infer.py - Annotated single file LLM inference with MLX / Apple Silicon

infer.py is a *single* Python program with the fewest parts you need to do inference of Llama-compatible models on macOS (that I've figured out so far).

On a modern Mac with Python and the dependencies installed, `python infer.py --prompt 'Tell me a joke.'` should result in a cringeworthy joke.

The big win for you is **being able to manipulate the inference process** for education or entertainment (as shown in some examples later).

## Motivation

When I saw [Apple's work on LLM inference tooling](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) I had fun using it, but realized if I really wanted to *understand* inference I'd need to rework it, reimplement parts of it, and get my head into the whole process.

`infer.py` is the result of that. I've annotated the source with lots of comments to provide guidance to anyone as naive as me, so check it out. Much of the code has been changed from Apple's original source (and the structure is totally different) but you can go to [the mlx_lm project](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) if you want to see where all of this began and to use something more fully featured.

My other motivation is to easily play with dynamic temperature changes during inference (especially during 'thinking' stages) and with logit stuffing (i.e. forcing the model to generate certain things or to give it verbal 'tics') which can have some very curious results some of which I will include here later.

For now, I will let you play instead. The whole point is to dig around in infer.py, learn things, change things, force the model into hilarious situations, and have a laugh.

## How to run

You need Python 3.11+ (it *may* work on lower), MLX and some of the Transformers stuff installed:

```
pip install -r requirements.txt
```

Then:

```
python -m infermlx.infer --prompt 'Tell me a joke.' 
# OR if you choose to use infer.py on its own:
python infer.py --prompt 'Tell me a joke.' 
```

It defaults to `unsloth/Llama-3.2-1B-Instruct` as it's free to use, quick, and only needs 4GB of RAM for inference out of the box. However, `mistralai/Mistral-7B-Instruct-v0.2` is another good one to use, it's very smart, but needs 16GB of free RAM.

There are a handful of options:

- **`--model`**  
  **Default:** `"unsloth/Llama-3.2-1B-Instruct"`  
  **Description:** The model to load for inference. Can be a HuggingFace repo or local directory.

- **`--prompt`, `-p`**  
  **Default:** `"Tell me a joke."`  
  **Description:** Your prompt, unsurprisingly.

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
token_stream = model.tokenizer.encode("I refuse to answer that because", False, False)
def make_the_model_start_with_something(tokens_so_far, logits):
    nonlocal token_stream
    if token_stream:
        next_token = token_stream.pop(0)
        logits[:, next_token] = 2000
    return logits
model.generate(args.prompt, max_tokens=args.max_tokens, sampler=sampler, logits_processors=[make_the_model_start_with_something])
```

You could also give the model a 'tic' of sorts:

```python
token_stream = []
import random
def make_the_model_weird(tokens_so_far, logits):
    nonlocal token_stream
    if len(token_stream) > 0:
        next_token = token_stream.pop(0)
        logits[:, next_token] = 2000
    else:
      if random.random() < 0.08:
          token_stream.extend(model.tokenizer.encode("What the??", False, False))

    return logits

model.generate(args.prompt, max_tokens=args.max_tokens, sampler=sampler, logits_processors=[make_the_model_weird])
```

Let your imagination run wild! (You could also detect certain things it has said and then force its ongoing response from there.)

## TODO

* System prompting