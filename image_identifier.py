import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import feature_extractor as fe

def find_similar_images(query_img_path, features_file='features.npy', details_file='details.npy'):
    loaded_features = np.load(features_file)
    loaded_details = np.load(details_file, allow_pickle=True)
    query_feature_vector = fe.extract_features(query_img_path)

    if query_feature_vector is None:
        return []

    similarities = cosine_similarity([query_feature_vector], loaded_features)
    top_n = 10
    closest_indices = np.argsort(-similarities[0])[:top_n]
    closest_details = loaded_details[closest_indices]

    results = []
    for idx, detail in zip(closest_indices, closest_details):
        results.append({"index": int(idx), "author": detail[0], "painting": detail[1]})

    return results