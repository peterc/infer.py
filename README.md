# infer.py - Annotated single file LLM inference with MLX / Apple Silicon

infer.py is a *single* Python program with the fewest parts you need to do inference of Llama-compatible models on macOS (that I've figured out so far).

On a modern Mac with Python and the dependencies installed, `python infer.py --prompt 'Tell me a joke.'` should result in a cringeworthy joke.

## Motivation

When I saw [Apple's work on LLM inference tooling](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) I had fun using it, but realized if I really wanted to *understand* inference I'd need to rework it, reimplement parts of it, and get my head into the whole process.

`infer.py` is the result of that. I've annotated the source with lots of comments to provide guidance to anyone as naive as me, so check it out. Much of the code has been changed from Apple's original source (and the structure is totally different) but you can go to [the mlx_lm project](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) if you want to see where all of this began and to use something more fully featured.

My other motivation is to easily play with dynamic temperature changes during inference (especially during 'thinking' stages) and with logit stuffing (i.e. forcing the model to generate certain things or to give it verbal 'tics') which can have some very curious results some of which I will include here later.

For now, I will let you play instead.

## How to run

You need Python 3.11+ (it *may* work on lower), MLX and some of the Transformers stuff installed:

```
pip install -r requirements.txt
```

Then:

```
python infer.py --prompt 'Tell me a joke.' 
```

It defaults to `mistralai/Mistral-7B-Instruct-v0.2` which you may need to approve the terms of on HuggingFace, so if you'd rather not run into any such problems, you can try this instead:

```
python infer.py --prompt 'Tell me a joke.' --model 'unsloth/Llama-3.2-1B-Instruct'
```

There are a handful of options:

- **`--model`**  
  **Default:** `"mistralai/Mistral-7B-Instruct-v0.2"`  
  **Description:** The model to load for inference. Can be a HuggingFace repo or local directory.

- **`--prompt`, `-p`**  
  **Default:** `"Tell me a joke."`  
  **Description:** Your prompt, unsurprisingly.

- **`--max-tokens`, `-m`**  
  **Type:** `int`  
  **Default:** `100`  
  **Description:** The maximum number of tokens to generate.

- **`--temp`**  
  **Type:** `float`  
  **Default:** `0.0`  
  **Description:** Temperature for sampling. As always, values yield more deterministic outputs, while higher values introduce more randomness. This is explained in more detail in the code.

- **`--seed`**  
  **Type:** `int`  
  **Default:** `42`
  **Description:** A random seed for the PRNG. Everyone in LLM-land seems to use 42 as a default because Ilya Sutskever did it in a demo once or something.

## TODO

* System prompting
* Provide actual examples of fun logit processing trickery