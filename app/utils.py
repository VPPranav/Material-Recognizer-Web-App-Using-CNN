
import cv2
import numpy as np

def preprocess_image(file, target_size=(96, 96)):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")

    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = img[start_y:start_y + min_dim, start_x:start_x + min_dim]

    resized = cv2.resize(cropped, target_size)
    normalized = resized.astype("float32") / 255.0

    return np.expand_dims(normalized, axis=0)
