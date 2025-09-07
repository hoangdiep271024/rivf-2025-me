import math
import numpy as np
import pandas as pd
from pathlib import Path
import os
import cv2

def load_image(path, image_size=(224, 224)):
    img = cv2.imread(path)
    if img is not None:
        resized_img = cv2.resize(img, image_size)
        return resized_img
    return None


def _compute_dynamic_image(frames):
    num_frames, h, w, depth = frames.shape

    # ARP coefficients: từ -(T-1) đến (T-1)
    coefficients = np.array([2 * (n + 1) - num_frames - 1 for n in range(num_frames)])

    # Áp dụng trọng số cho từng frame
    x1 = np.expand_dims(frames, axis=0)                     # (1, T, H, W, C)
    x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))    # (T, 1, 1, 1)
    result = x1 * x2                                        # Broadcasting theo trọng số
    return np.sum(result[0], axis=0).squeeze()              # (H, W, C)

def get_dynamic_image(frames, normalized=True):
    """
    frames: List hoặc mảng numpy với shape (T, H, W, C)
    """
    frames = np.array(frames)
    dynamic_image = _compute_dynamic_image(frames)
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image

if __name__ == "__main__":
    img_seq_dir_path = "/media/hmi/Transcend1/CASMEV2/sequences/Seq_5/"
    dest_path = "/media/hmi/Transcend1/CASMEV2/dynamic_images"   
    image_size = (224, 224)
    
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    
    # sort by extract number \d from image name (image_1.jpg -> 1)
    img_list = sorted(os.listdir(img_seq_dir_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # should resize image to (image_size, image_size)
    img_pat_List = [os.path.join(img_seq_dir_path, img_name) for img_name in img_list]
    images = [load_image(img_path) for img_path in img_pat_List if load_image(img_path) is not None]
    dynamic_image = get_dynamic_image(images)
    save_path = os.path.join(dest_path, img_seq_dir_path.split('/')[-2] + '.jpg')
    cv2.imwrite(save_path, dynamic_image)

