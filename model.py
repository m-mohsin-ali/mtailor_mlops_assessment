import onnx
import onnxruntime
import numpy as np
from PIL import Image


class Prep:

    def preprocess(self, image_path):
        image = Image.open(image_path)
        # Convert to RGB format if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize to 224x244
        image = image.resize((224, 224), resample=Image.BILINEAR)
        # Convert to numpy array
        image = np.array(image)
        # Divide by 255
        image = image / 255.0
        # Normalize using mean and standard deviation
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        # Transpose to channel-first format
        # print(image.shape)
        image = np.transpose(image, (2, 0, 1))
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        # print(image.shape)
        return image.astype(np.float32)


class OnnxModel:
    onnx_model = onnx.load("Classifier.onnx")
    onnx_session = onnxruntime.InferenceSession("Classifier.onnx")

    def inference(self, input_image):
        onnx_input_name = self.onnx_session.get_inputs()[0].name
        onnx_output_name = self.onnx_session.get_outputs()[0].name
        onnx_output = self.onnx_session.run([onnx_output_name], {onnx_input_name: input_image})[0]
        return np.argmax(onnx_output[0])
