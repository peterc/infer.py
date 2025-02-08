import infermlx.infer as infermlx
model = infermlx.Model.load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

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
    if len(tokens) == 2000 and is_it_thinking:
        # It's thought for long enough!
        logits[:, end_of_thinking] = 2000
        return logits

    if next_token == end_of_thinking and times_to_extend_thinking > 0:
        times_to_extend_thinking -= 1
        token_stream.extend(model.tokenizer.encode("Wait, I have a idea. ", False, False))
    elif next_token == end_of_thinking and times_to_extend_thinking == 0:
        is_it_thinking = False

    if len(token_stream) > 0:
        next_token = token_stream.pop(0)
        logits[:, next_token] = 2000
        
    return logits

model.generate("Generate an HTML page with fun colors, CSS, and the word 'infermlx' in the middle bouncing around in a funny way.", temp=0.5, system_prompt="You are a creative Web designer.", max_tokens = 4096, logits_processors=[keep_deepseek_thinking])
