import torch
import onnx
import onnxruntime
from pytorch_model import Classifier
from pytorch_model import BasicBlock


# Load PyTorch model

def convert_to_onnx(weights, model):
    pytorch_model = model
    pytorch_model.load_state_dict(torch.load(weights))
    pytorch_model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)
    onnx_model_path = f'{model.__class__.__name__ }.onnx'
    torch.onnx.export(pytorch_model, dummy_input, onnx_model_path)


if __name__ == "__main__":
    convert_to_onnx('pytorch_model_weights.pth', Classifier(BasicBlock, [2, 2, 2, 2]))
