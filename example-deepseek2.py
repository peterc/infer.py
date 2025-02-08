import infermlx.infer as infermlx
model = infermlx.Model.load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# Controlling DeepSeek's thinking process to actually reach the
# right answer when usually it gets it wrong!
# I extend the thinking process to give it more time to think
# but then stop it from thinking after a certain point to
# avoid thinking loops common in the small models.

token_stream = []
times_to_extend_thinking = 3
is_it_thinking = True
def keep_deepseek_thinking(tokens, logits, next_token):
    global times_to_extend_thinking
    global is_it_thinking

    # </think> is a special token in this model – you can't
    # just tokenize </think> and expect it to work, alas.
    end_of_thinking = 128014

    # If we get 500 tokens in and it's still thinking, stop it doing so.
    if len(tokens) == 500 and is_it_thinking:
        # It's thought for long enough!
        print("\n\nStopping thinking!!\n")
        logits[:, end_of_thinking] = 2000
        return logits

    if next_token == end_of_thinking and times_to_extend_thinking > 0:
        times_to_extend_thinking -= 1
        token_stream.extend(model.tokenizer.encode("Wait, ", False, False))
    elif next_token == end_of_thinking and times_to_extend_thinking == 0:
        is_it_thinking = False

    if len(token_stream) > 0:
        next_token = token_stream.pop(0)
        logits[:, next_token] = 2000
        
    return logits

model.generate("How can I solve x = 2x + 20", system_prompt="Speak in plain ASCII text. Do not use any special formatting.", max_tokens = 2048, logits_processors=[keep_deepseek_thinking])
