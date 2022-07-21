import numpy as np
import cv2 as cv
import torch
import imgaug.augmenters as iaa
import json

from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from core.opts import dataset_opts


def crop_angle(angle):
    angle = angle % 90
    if angle > 45:
        return 90 - angle
    else:
        return angle


class ImageDataset(Dataset):
    def __init__(self, root, orig_image, dataset_opts=dataset_opts, transform=None, training=True):
        self.root = Path(root)
        self.orig_image = orig_image
        self.transform = transform
        
        self.original_size = dataset_opts.original_size
        self.scale = dataset_opts.scale
        self.scaled_size = self.original_size // self.scale
        # self.scaled_size = orig_image.shape[0]
        # self.scale = self.original_size / self.scaled_size
        
        if training:
            self.cropped_size = 800
        else:
            self.cropped_size = 100
        
        self.aug_clouds = iaa.Clouds()
        self._process_json()
        
    def _process_json(self):
        json_dir = self.root / 'json'
        self.gt = []

        for p in json_dir.iterdir():
            if not p.name.endswith('.json'):
                continue

            with open(p, 'r') as f:
                obj = json.load(f)            
            
            img_name = f"{int(p.stem)}.png"
            
            self.gt.append((img_name, np.array(obj['left_top'] + obj['right_bottom'] + [np.deg2rad(obj['angle'])])))
        
    def __len__(self):
        return len(self.gt) + self.cropped_size
    
    def __getitem__(self, idx):
        if idx >= len(self.gt):
            image, labels = self.cropped_data()
        else:
            image_name, labels = self.gt[idx][0], self.gt[idx][1]
            image = Image.open(str(self.root / 'img' / image_name))

        if self.transform:
            image = self.transform(image)
        
        labels = self._process_labels(labels.copy())
        return image, torch.from_numpy(labels)
    
    def cropped_data(self):
        angle = np.random.randint(360)
        
        
        rad_angle = np.deg2rad(crop_angle(angle))
        s, c = np.sin(rad_angle), np.cos(rad_angle)
        size = int(round(1024 * (s + c)))
        s_less = size // 2

        x = np.random.randint(s_less + 1, self.original_size - s_less - 1)
        y = np.random.randint(s_less + 1, self.original_size - s_less - 1)
        
        labels, image = self._get_rotated((x, y), angle)
        
        if np.random.random() > 0.5:
            image = self.aug_clouds.augment_image(image)
            
        return image, labels 
        
    def _get_rotated(self, center, deg_angle):
        cc = np.array([center[0], center[1]])
        
        x_scaled, y_scaled = center[1] // self.scale, center[0] // self.scale
        rad_angle_ = np.deg2rad(crop_angle(deg_angle))
        crop_size_scaled = int(round(
            (np.cos(rad_angle_) + np.sin(rad_angle_)) * 1024 / self.scale
        ))
        half = crop_size_scaled // 2
        
        s1, s2 = max(x_scaled - half, 0), min(x_scaled + half, self.scaled_size)
        s3, s4 = max(y_scaled - half, 0), min(y_scaled + half, self.scaled_size)
        
        cropped_image = self.orig_image[s1:s2, s3:s4]
        M = cv.getRotationMatrix2D((half, half), deg_angle, 1.0)
        
        rotated = cv.warpAffine(cropped_image, M, (s2 - s1, s4 - s3))
        cX, cY = half, half
        half_s = 512 // self.scale
        
        # generated coordinates
        M_hat = M[:, :2].T
        lt = (cc + M_hat @ np.array([-512, -512]))  # left top
        rb = (cc + M_hat @ np.array([512, 512]))    # right bottom
        
        return np.array(lt.tolist() + rb.tolist() + [np.deg2rad(deg_angle)]), rotated[cX - half_s: cX + half_s, cY - half_s: cY + half_s]
    
    def _process_labels(self, labels):
        """
        labels: [ltx, lty, rbx, rby, angle in radians]
        """
        
        copy = labels.copy()
        
        labels[:4] = labels[:4] / self.original_size
        labels[4] = labels[4] / (2 * np.pi)
        # if labels.min() < 0. or labels.max() > 1.:
        #     print(copy)
        # assert labels.min() >= 0. and labels.max() <= 1., f'incorrect labels, not in [0, 1] {labels}'
        
        return labels
    
    
class PredictImageDataset(Dataset):
    def __init__(self, root, dataset_opts=dataset_opts, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.scale = dataset_opts.scale
        
        self.data = [(p.stem, p) for p in self.root.iterdir() if p.name.endswith('.png')]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        stem, path = self.data[idx]
        image = Image.open(path).resize((1024 // self.scale, 1024 // self.scale))
        image = self.transform(image)
        return stem, image
