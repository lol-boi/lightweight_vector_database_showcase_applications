import onnxruntime as ort
from PIL import Image
import numpy as np
import os

class ImageEmbedder:
    def __init__(self, model_path='models/mobilenetv2-7.onnx'):
        # Check if the model file exists
        if not os.path.exists(model_path):
            # Fallback for running from script's directory
            script_dir = os.path.dirname(__file__)
            model_path_fallback = os.path.join(script_dir, model_path)
            if not os.path.exists(model_path_fallback):
                raise FileNotFoundError(f"ONNX model not found at {model_path} or {model_path_fallback}. Please ensure the model file is in the correct directory.")
            model_path = model_path_fallback

        # Load the ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Define image preprocessing parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _preprocess(self, image):
        # 1. Resize so smaller edge is 256, maintaining aspect ratio
        w, h = image.size
        if w < h:
            new_w = 256
            new_h = int(h * (256 / w))
        else:
            new_h = 256
            new_w = int(w * (256 / h))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 2. Center crop 224x224
        left = (new_w - 224) / 2
        top = (new_h - 224) / 2
        right = (new_w + 224) / 2
        bottom = (new_h + 224) / 2
        image = image.crop((left, top, right, bottom))

        # 3. Convert to numpy array, scale to [0, 1], and change to CHW format
        img_np = np.array(image).astype('float32') / 255.0
        img_np = img_np.transpose((2, 0, 1)) # HWC to CHW

        # 4. Normalize
        normalized_img = (img_np - self.mean[:, np.newaxis, np.newaxis]) / self.std[:, np.newaxis, np.newaxis]
        
        return normalized_img.astype('float32')

    def embed_image(self, image_path):
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess the image
        input_tensor = self._preprocess(image)
        
        # Add a batch dimension
        input_batch = np.expand_dims(input_tensor, axis=0)

        # Run inference
        output = self.session.run([self.output_name], {self.input_name: input_batch})[0]

        # Flatten the output
        embedding = output.flatten()
        return embedding

if __name__ == '__main__':
    # Example usage:
    # You'll need an image file for this to run.
    # For example, create a dummy image or use an existing one.
    try:
        img = Image.new('RGB', (250, 300), color = 'red')
        img.save('test_image.png')
        print("Created a dummy 'test_image.png'.")
    except Exception as e:
        print(f"Could not create dummy image: {e}")


    try:
        embedder = ImageEmbedder()
        
        print("Embedding 'test_image.png'...")
        embedding = embedder.embed_image('test_image.png')
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 10 elements of embedding: {embedding[:10]}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'mobilenetv2.onnx' is in the 'models' directory and you are running this script from the correct path.")
    except Exception as e:
        print(f"An error occurred: {e}")
