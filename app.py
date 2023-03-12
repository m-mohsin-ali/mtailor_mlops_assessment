from transformers import pipeline
import torch
from model import OnnxModel
from model import Prep

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    device = 0 if torch.cuda.is_available() else -1
    model = OnnxModel()


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    pre= Prep()
    result = model.inference(input_image=pre.preprocess(prompt))

    # Return the results as a dictionary
    return result
