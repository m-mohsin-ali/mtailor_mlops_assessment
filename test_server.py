
import banana_dev as banana
import base64
from io import BytesIO


with open("test.jpg", "rb") as f:
    im_b64 = base64.b64encode(f.read())


model_inputs = {
  "prompt": f"{im_b64}",
}

api_key = "27138ec9-7d2a-4833-878d-9176d9200400"
model_key = "a54c9c7d-f6b3-47aa-bcaf-0dac6ca5fca2"

# Run the model
#out = banana.run(api_key, model_key, model_inputs)
print(model_inputs)