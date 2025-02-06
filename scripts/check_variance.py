import torch
import numpy as np
from torchvision.io import read_image
import os

name_coord_map = {
'_DSC1454': [332, 1201],
'_DSC1459': [276, 1274],
'_DSC1461': [350, 1360],
'_DSC1462': [370, 1250],
'_DSC1464': [360, 990],
'_DSC1480': [324, 1280],
'_DSC1486': [313, 1250],
'_DSC1488': [316, 1227],
'_DSC1490': [270, 1070],
}

img_suffix = 'jpg'
base_path = './data/refnerf/sedan'
downscale = 2
feat_path = os.path.join(base_path, f'features_{downscale}')
img_path = os.path.join(base_path, f'images_{downscale}')

id1 = (list(name_coord_map.keys())[0])
img_test = read_image(os.path.join(img_path, f'{id1}.{img_suffix}'))
img_size = img_test.shape[-2:]# y, x
dino = torch.load(os.path.join(feat_path, f'{id1}_feat.pt'))
dino_size = dino.shape[:-1]# y, x

scale_y = dino_size[0] / img_size[0]
scale_x = dino_size[1] / img_size[1]

print(img_size, dino_size)
dino_feats = []

for name, coord in name_coord_map.items():
    dino = torch.load(os.path.join(feat_path, f'{name}_feat.pt'))
    y, x = coord
    y = int(y * scale_y)
    x = int(x * scale_x)
    dino_feats.append(dino[y, x])

dino_feats = torch.stack(dino_feats)

dino_mean = torch.mean(dino_feats, dim=0)
dino_std = torch.var(dino_feats, dim=0)

print(f"min: {torch.min(dino_feats)}, max: {torch.max(dino_feats)}")
print(dino_feats.shape,dino_feats.mean(), dino_feats.var())
print(dino_mean.shape, dino_mean.mean(), dino_mean.var())
print(dino_std.shape, dino_std.mean(), dino_std.median(),dino_std.var())


