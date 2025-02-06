# Dino feature computation

#!wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip
#!unzip tandt_db.zip
#!git clone https://huggingface.co/IsaacLabe/data_hyperNerf
#!git clone https://huggingface.co/IsaacLabe/data_hyperNerf_2
#!unzip data_hyperNerf/interp_chickchicken.zip
#!unzip /content/4D-gaussian-splatting-sementic-New/data_hyperNerf_2/interp_chickchicken.zip
#!unzip /content/4D-gaussian-splatting-sementic-New/data_hyperNerf/misc_split-cookie.zip
import os
import sys
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from torchvision import transforms
import copy
import torch
import torch.nn.functional as F
# import open_clip
# from open_clip import tokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import argparse
from alive_progress import alive_bar

# large model
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
# dinov2_vits14.to('cuda')
# print("OK")

dinov2_vits14 = None

class Dino_V2():
    def __init__(self, image_name, model_DINO_net, image=None):
        self.image_name = image_name
        if image is not None:
            self.image = image
        else:
            self.image = Image.open(str(self.image_name)).convert('RGB')
        (self.H, self.W, self.d) = np.shape(self.image)
        self.feature_dim = 384
        self.model_DINO_net = model_DINO_net
        self.patch_size = model_DINO_net.patch_size
        self.image_tensor = self.process_image()
        self.patch_h = self.H // self.patch_size
        self.patch_w = self.W // self.patch_size

    def closest_mult_shape(self, n_H, n_W):
        closest_multiple_H = n_H * round(self.H // n_H)
        closest_multiple_W = n_W * round(self.W // n_W)
        return (closest_multiple_H, closest_multiple_W)

    def process_image(self):
        transform1 = transforms.Compose([
            transforms.Resize((self.H, self.W)),
            transforms.CenterCrop(self.closest_mult_shape(
                self.patch_size, self.patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.2)
        ])
        image_tensor = transform1(self.image).to('cuda').unsqueeze(0)
        return image_tensor

    def extract_feature(self):
        with torch.no_grad():
            # Extract per-pixel DINO features (1, 384, H // patch_size, W // patch_size)
            image_feature_dino = self.model_DINO_net.forward_features(self.image_tensor)[
                'x_norm_patchtokens']
            image_feature_dino = image_feature_dino[0].reshape(
                self.patch_h, self.patch_w, self.feature_dim)
        return image_feature_dino




def extract_features_blender(parent_path, imgs_folder, dest_path="features"):
    # create dest folder if not exists
    print("Creating dest folder if not exists")
    dest_path = os.path.join(parent_path, dest_path)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    images_path = os.path.join(parent_path, imgs_folder)

    # filter out non-image files
    images_list = [str(img_path) for _, img_path in enumerate(os.listdir(images_path)) if 
                   not ("tiff" in img_path or "normal" in img_path or "alpha" in img_path or "disp" in img_path or "feat" in img_path)]

    with alive_bar(len(images_list)) as bar:
        for i, img_path in enumerate(images_list):
            # try:
            if (not ('png' in img_path or 'jpg' in img_path or 'jpeg' in img_path or 
                    'PNG' in img_path or 'JPG' in img_path or 'JPEG' in img_path)) or \
                        'normal' in img_path or 'dept' in img_path:
                continue
            img = Image.open(os.path.join(parent_path, imgs_folder, img_path)).convert('RGB')
            # except: continue
            width, height = img.size
            print('original img shape: ', img.size)
            scale_factor = int(dest_path.split('_')[-1])
            new_width, new_height = width//scale_factor, height//scale_factor
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print('new img shape: ', img.size)
            dino_class = Dino_V2(os.path.join(images_path, img_path), dinov2_vits14, image=img)
            image_feature = dino_class.extract_feature()
            img_name = img_path.split(".")[0]
            torch.save(image_feature, os.path.join(dest_path, f"{img_name}_feat.pt"))
            bar()

def extract_features(parent_path, imgs_folder, dest_path="features"):
    # create dest folder if not exists
    print("Creating dest folder if not exists")
    dest_path = os.path.join(parent_path, dest_path)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    images_path = os.path.join(parent_path, imgs_folder)
    dest_path = os.path.join(parent_path, dest_path)

    # filter out non-image files
    images_list = [str(img_path) for _, img_path in enumerate(os.listdir(images_path)) if 
                   not ("tiff" in img_path or "normal" in img_path or "disp" in img_path or "feat" in img_path)]

    with alive_bar(len(images_list)) as bar:
        for i, img_path in enumerate(images_list):                    
            dino_class = Dino_V2(os.path.join(images_path ,img_path), dinov2_vits14, image=None)
            image_feature = dino_class.extract_feature()
            img_name = img_path.split(".")[0]
            torch.save(image_feature, os.path.join(dest_path, f"{img_name}_feat.pt"))
            bar()

def resize_images(parent_path, imgs_folder, dest_path="resized", size=(196, 196)):
    # create dest folder if not exists
    dest_path = os.path.join(parent_path, dest_path)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    images_path = os.path.join(parent_path, imgs_folder)
    
    for i, img_path in enumerate(os.listdir(images_path)):
        if "tiff" in img_path or "normal" in img_path or "disp" in img_path or "feat" in img_path:
            continue

        img = cv2.imread(os.path.join(images_path, img_path))
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        
        cv2.imwrite(os.path.join(dest_path, img_path), img)
        print(i)

def resize_png_images(parent_path, imgs_folder, dest_path="resized", size=(196, 196)):
    # create dest folder if not exists
    dest_path = os.path.join(parent_path, dest_path)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # resize images and keep alpha channel
    images_path = os.path.join(parent_path, imgs_folder)
    for i, img_path in enumerate(os.listdir(images_path)):
        if "tiff" in img_path or "normal" in img_path or "disp" in img_path or "feat" in img_path:
            continue

        img = Image.open(os.path.join(images_path, img_path))
        img = img.resize(size, Image.LANCZOS)
        img.save(os.path.join(dest_path, img_path))
        print(i)

def test():
    img_path = ''
    dino_class = Dino_V2(img_path, dinov2_vits14, image=None)
    image_feature = dino_class.extract_feature()
    print(np.shape(image_feature))
    (H, W, d) = np.shape(image_feature)
    image_feature = image_feature.reshape(H*W, d)

    pca = PCA(n_components=3)
    pca.fit(image_feature.detach().cpu())
    
    # pca_features = pca.transform(image_feature.detach().cpu()) # ndarray
    pca_features = image_feature.detach().cpu() @ pca.components_.T  # tensor

    # pca_features = (pca_features - pca_features.min(axis=0)) / (pca_features.max(axis=0) - pca_features.min(axis=0))
    pca_features = (pca_features - pca_features.min(dim=0)
                    [0]) / (pca_features.max(dim=0)[0] - pca_features.min(dim=0)[0])

    plt.figure()
    plt.imshow(pca_features.reshape(H, W, 3))
    plt.savefig(fname='pca_features.png')


if __name__ == "__main__":
    """
    example for run:
    python ./scripts/extract_dino_features.py --parent_path ./data/refnerf/sedan --scale_factor 4
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_path', type=str, required=True, help='parent path of the scene directory')
    parser.add_argument('--scale_factor', type=int, default=4, help='scale factor for the images')
    parser.add_argument('--blender', type=bool, default=False, help='boolean indicating if blender or colmap format')

    args = parser.parse_args()    
    
    print("Loading dinov2...")
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    dinov2_vits14.to('cuda')
    print("Dinov2 loaded successfully!")

    parent_path = args.parent_path

    if args.blender:
        dirs = ['train', 'val', 'test']
        for dir_name in dirs:
            imgs_folder = f"{dir_name}"
            if dir_name not in os.listdir(parent_path):
                continue
            dest_path = f"features_{dir_name}_{args.scale_factor}"
            print("Start extracting features...")
            extract_features_blender(parent_path, imgs_folder, dest_path=dest_path)
            print(f"Features extraction finished for {dir_name}!")
    else:
        suffix = "" if args.scale_factor==1 else f"_{args.scale_factor}"
        imgs_folder = f"images{suffix}" #TODO: change here blender
        dest_path = f"features{suffix}"

        print("Start extracting features...")
        extract_features(parent_path, imgs_folder, dest_path=dest_path)
        print("Features extraction finished!")