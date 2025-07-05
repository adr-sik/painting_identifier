import os
import numpy as np
from multiprocessing import Pool
import feature_extractor as fe

def process_image(img_path):
    try:
        features = fe.extract_features(img_path)
        if features is not None:
            return features, img_path
        else:
            return None
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        return None

if __name__ == '__main__':
    # Directory containing images
    input_directory = ""
    print(f"Checking directory: {input_directory}")

    if os.path.exists(input_directory):
        print("Directory exists.")
    else:
        print("Directory does not exist or cannot be accessed.")        

    # Get list of image paths
    image_paths = []
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                image_paths.append(img_path)

    # Process images in parallel
    num_processes = os.cpu_count()  # Number of CPU cores
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_image, image_paths)

    # Separate results into features and details
    features_list = []
    details = []
    for result in results:
        if result is not None:
            features_list.append(result[0])
            img_path = result[1]
            parts = os.path.basename(img_path).split('_')
            author = parts[0]
            painting_parts = parts[1].split('.')
            painting = painting_parts[0]
            details.append((author, painting))

    # Convert lists to numpy arrays
    if features_list and details:
        features_array = np.array(features_list)
        details_array = np.array(details)

        # Save features and image paths to NumPy file
        np.save('features.npy', features_array)
        np.save('details.npy', details_array)
    else:
        print("No valid features extracted.")
