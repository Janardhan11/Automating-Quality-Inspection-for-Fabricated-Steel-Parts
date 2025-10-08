import os, cv2
import numpy as np

IMG_SIZE = 256

def load_data(img_dir, mask_dir):
    X, y = [], []
    for fname in os.listdir(img_dir):
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.exists(mask_path): continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0
        mask = np.expand_dims(mask, axis=-1)

        X.append(img)
        y.append(mask)

    return np.array(X), np.array(y)
