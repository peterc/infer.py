import infermlx.infer as infermlx
model = infermlx.Model.load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

token_stream = []
times_to_extend_thinking = 2
def keep_deepseek_thinking(_, logits, next_token):
    global times_to_extend_thinking
    # </think> is a special token in this model – you can't
    # just tokenize </think> and expect it to work, alas.
    end_of_thinking = 128014

    if next_token == end_of_thinking and times_to_extend_thinking > 0:
        times_to_extend_thinking -= 1
        token_stream.extend(model.tokenizer.encode("Wait, let's think again. ", False, False))

    if len(token_stream) > 0:
        next_token = token_stream.pop(0)
        logits[:, next_token] = 2000
        
    return logits

#model.generate("What is the shortest way to tally an array's contents in Ruby?", max_tokens = 4096)
model.generate("What is your name?", max_tokens = 4096, logits_processors=[keep_deepseek_thinking])