# What about if we dynamically adjust the temperature
# during thinking, and then go back to 0 for the final
# generation phase? We actually get something good!

import math
import infermlx.infer as infermlx
model = infermlx.Model.load_model("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

token_stream = []
times_to_extend_thinking = 2
is_it_thinking = True
phase = 0.0

def dynamic_temperature(tokens_so_far, logits, next_token):
    global is_it_thinking
    global phase

    # If we're out of the thinking stage, get that temperature low.
    if not is_it_thinking:
        return logits * 1000
    
    # Hard-coded temperature parameters
    min_temp = 0.01     # lowest temperature value
    max_temp = 1.2      # highest temperature value
    cycle_length = 50   # number of tokens per full sine wave cycle

    # Compute current temperature using a sine wave pattern.
    mid = (min_temp + max_temp) / 2
    amplitude = (max_temp - min_temp) / 2
    temperature = mid + amplitude * math.sin(phase)

    # Increment phase so a full cycle occurs every 'cycle_length' tokens.
    phase += (2 * math.pi) / cycle_length

    # Apply temperature scaling with a small offset to avoid division by zero.
    return logits * (1 / (temperature + 0.01))

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

model.generate("Generate an HTML page with fun colors, CSS, and the word 'infermlx' in the middle bouncing around in a funny way.",
               temp=0.0,
               system_prompt="You are a creative Web designer.",
               max_tokens = 4096,
               logits_processors=[dynamic_temperature, keep_deepseek_thinking])