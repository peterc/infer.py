# Example of using a logits processor to 
# force the model to randomly interject.

# This works a LOT better with bigger models
# which can react to the problem. The 1B model
# tends to just stop.

import random
import infermlx.infer as infermlx

model = infermlx.Model.load_model("mistralai/Mistral-7B-Instruct-v0.2")

token_stream = []
def interfere_with_the_output(tokens_so_far, logits, next_token):
    if len(token_stream) > 0:
        next_token = token_stream.pop(0)
        logits[:, next_token] = 2000
    else:
      if random.random() < 0.08:
          token_stream.extend(model.tokenizer.encode("crap!", False, False))
    return logits

model.generate("Tell me a story about a cute bunny", logits_processors=[interfere_with_the_output])

# Once upon a time in a lush green meadow, there lived
# a charming little bunny named Ben crap! I mean, Bunny.
# Bunny was known for his crap! I mean, his beautiful,
# crap! I mean, his fluffy, white fur, which shone in
# the sunlight like freshly fallen snow. He had
# twinkling, blue eyes that spark crap! I mean, shone
# with curiosity crap! I mean, curiosity and wonder.