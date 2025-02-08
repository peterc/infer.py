# Example of using a logits processor to 
# force the model to start with a certain set of tokens.

import infermlx.infer as infermlx

model = infermlx.Model.load_model()

token_stream = model.tokenizer.encode("Here's one about pirates.", False, False)
def make_the_model_start_with_something(tokens_so_far, logits, next_token):
    if token_stream:
        next_token = token_stream.pop(0)
        logits[:, next_token] = 2000
    return logits

model.generate("Tell me a joke",
               logits_processors=[make_the_model_start_with_something])