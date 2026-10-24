import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])


def extract_features_batch(image_paths):
    """Extract L2-normalized ResNet50 embeddings for a batch of image paths."""
    if not image_paths:
        return np.empty((0, 2048), dtype=np.float32)

    images = [
        image.img_to_array(image.load_img(path, target_size=(224, 224)))
        for path in image_paths
    ]
    batch = np.asarray(images, dtype=np.float32)
    batch = preprocess_input(batch)

    features = model(batch, training=False).numpy()
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("ResNet50 produced a zero-length image embedding.")
    return (features / norms).astype(np.float32)


def extract_features(img_path):
    return extract_features_batch([img_path])[0]