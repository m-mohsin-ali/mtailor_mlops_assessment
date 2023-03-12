import onnx
import onnxruntime
import numpy as np
from PIL import Image
import unittest


def preprocess(image_path):
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


def inference(onnx_model_path, input_image):
    onnx_model = onnx.load(onnx_model_path)
    onnx_session = onnxruntime.InferenceSession(onnx_model_path)
    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name
    onnx_output = onnx_session.run([onnx_output_name], {onnx_input_name: input_image})[0]
    return np.argmax(onnx_output[0])


class Testimages(unittest.TestCase):

    def test_image1(self):
        self.assertEqual(inference('Classifier.onnx', preprocess("n01440764_tench.jpeg")), 0, "Should be 0")

    def test_image2(self):
        self.assertEqual(inference('Classifier.onnx', preprocess("n01667114_mud_turtle.JPEG")), 35, "Should be 35")


if __name__ == '__main__':
    unittest.main()
