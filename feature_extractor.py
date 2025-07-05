import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def pad_image(image, target_size=(299, 299)):
    width, height = image.size
    target_width, target_height = target_size

    # Check if padding is needed
    pad_width = max(0, target_width - width)
    pad_height = max(0, target_height - height)
    
    # Resize if the image is larger than the target size
    if width > target_width or height > target_height:
        image = image.resize((target_width, target_height))

    # Apply padding (if any)
    new_image = Image.new("RGB", target_size, (0, 0, 0))  # Black padding
    new_image.paste(image, (pad_width // 2, pad_height // 2))

    return new_image

def extract_features(img_path):
    try:
        img = image.load_img(img_path)
        img = pad_image(img)
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        features = base_model.predict(img_data)
        return features.flatten()
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        return None
