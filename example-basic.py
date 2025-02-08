# Simplest example of generating text with the default settings

import infermlx.infer as infermlx

model = infermlx.Model.load_model()
model.generate("Tell me a joke")