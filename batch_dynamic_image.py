import math
import numpy as np
import pandas as pd
from pathlib import Path
import os
import cv2
from tqdm import tqdm 


def load_image(path, image_size=(224, 224)):
    img = cv2.imread(path)
    if img is not None:
        resized_img = cv2.resize(img, image_size)
        return resized_img
    return None


def assign_coefficients(coefficients, apex_index):
    num_frames = len(coefficients)
    sorted_coefficients = np.sort(coefficients)[::-1]
    new_coefficients = np.zeros(num_frames)
    new_coefficients[apex_index] = sorted_coefficients[0]

    left = apex_index - 1
    right = apex_index + 1
    coeff_index = 1  

    while coeff_index < num_frames:
        if left >= 0 and coeff_index < num_frames:
            new_coefficients[left] = sorted_coefficients[coeff_index]
            left -= 1
            coeff_index += 1
        if right < num_frames and coeff_index < num_frames:
            new_coefficients[right] = sorted_coefficients[coeff_index]
            right += 1
            coeff_index += 1

    return new_coefficients


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
    batch_img_seq_dir_path = "/media/hmi/Transcend1/CASMEV2/sequences/"
    dest_path = "/media/hmi/Transcend1/CASMEV2/dynamic_images"   
    image_size = (112, 112)
    
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    seq_dir_list = Path(batch_img_seq_dir_path).glob('*')
    seq_dir_list = [str(p.name) for p in seq_dir_list if p.is_dir()]
    # sort by extract number \d from image name (image_1.jpg -> 1)
    for seq_dir in tqdm(seq_dir_list):
        print("Seq: ", seq_dir)
        img_seq_dir_path = os.path.join(batch_img_seq_dir_path, seq_dir)
        # use lambda func to extract number from image name
        # should resize image to (image_size, image_size)
        img_pat_list = Path(img_seq_dir_path).glob('*.jpg')
        img_pat_list = sorted(img_pat_list, key=lambda x: int(x.stem.split('_')[-1]))
        img_pat_list = [str(p) for p in img_pat_list]
        images = [load_image(img_path, image_size) for img_path in img_pat_list if load_image(img_path) is not None]
        dynamic_image = get_dynamic_image(images)
        save_path = os.path.join(dest_path, seq_dir + '.jpg')
        cv2.imwrite(save_path, dynamic_image)

