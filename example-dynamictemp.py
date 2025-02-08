# Example of using a logits processor to 
# change the temperature of generation on
# the fly as we go.

import infermlx.infer as infermlx
import math
import sys

model = infermlx.Model.load_model("mistralai/Mistral-7B-Instruct-v0.2")

phase = 0.0
token_stream = model.tokenizer.encode("OK, here you go: ", False, False)
def dynamic_temperature(tokens_so_far, logits, next_token):
    global phase
    if token_stream:
        next_token = token_stream.pop(0)
        logits[:, next_token] = 2000
        return logits

    # Hard-coded temperature parameters
    min_temp = 0.3      # lowest temperature value
    max_temp = 1.4      # highest temperature value
    cycle_length = 40   # number of tokens per full sine wave cycle

    # Compute current temperature using a sine wave pattern.
    mid = (min_temp + max_temp) / 2
    amplitude = (max_temp - min_temp) / 2
    temperature = mid + amplitude * math.sin(phase)

    # Increment phase so a full cycle occurs every 'cycle_length' tokens.
    phase += (2 * math.pi) / cycle_length

    # Apply temperature scaling with a small offset to avoid division by zero.
    return logits * (1 / (temperature + 0.01))

# Take the prompt from the command line or use a default
prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain calculus in layman's terms"

# It's essential to start with a temperature of 1.0 so we can scale against that
model.generate(prompt, max_tokens=1024, temp = 1.0, logits_processors=[dynamic_temperature])
