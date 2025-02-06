import math


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF


import os
from PIL import Image
import imageio
import numpy as np
from torch.utils.data import Dataset
from nerfacc import ContractionType
import pytorch_lightning as pl
from utils.misc import get_rank


from pathlib import Path
import open3d as o3d
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json

def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center

def normalize_poses(poses, pts, up_est_method, center_est_method, pts3d_normal=None):
    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[...,3].mean(0)
    elif center_est_method == 'lookat':
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[...,3]
        cams_dir = poses[:,:3,:3] @ torch.as_tensor([0.,0.,-1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1,0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1,0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1,0)], dim=-1) * t[:,None,:] + torch.stack([cams_ori, cams_ori.roll(1,0)], dim=-1)).mean((0,2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = poses[...,3].mean(0)
    else:
        raise NotImplementedError(f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(pts.numpy(), thresh=0.01) # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(plane_eq) # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1) # plane normal as up direction
        signed_distance = (torch.cat([pts, torch.ones_like(pts[...,0:1])], dim=-1) * plane_eq).sum(-1)
        if signed_distance.mean() < 0:
            z = -z # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[...,3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f'Unknown up estimation method: {up_est_method}')

    # new axis
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == 'point':
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, torch.as_tensor([[0.,0.,0.]]).T], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]

        # translation and scaling
        poses_min, poses_max = poses_norm[...,3].min(0)[0], poses_norm[...,3].max(0)[0]
        pts_fg = pts[(poses_min[0] < pts[:,0]) & (pts[:,0] < poses_max[0]) & (poses_min[1] < pts[:,1]) & (pts[:,1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat([poses_norm, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses_norm.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        # apply the rotation to the point cloud normal
        if pts3d_normal is not None:
            pts3d_normal = (R @ pts3d_normal.T).T
        pts = pts / scale
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3] # (N_images, 4, 4)

        # scaling
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale

        # apply the transformation to the point cloud
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        # apply the rotation to the point cloud normal
        if pts3d_normal is not None:
            pts3d_normal = (R @ pts3d_normal.T).T
        pts = pts / scale

    return poses_norm, pts, pts3d_normal
def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0., 0., 0.], dtype=cameras.dtype, device=cameras.device)
    mean_d = (cameras - center[None, :]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:, 2].mean()
    r = (mean_d ** 2 - mean_h ** 2).sqrt()
    up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:, None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)

    return all_c2w


def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x

def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat

def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def open_file(pth, mode='r'):
    return open(pth, mode=mode)


class BlenderDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}

    def setup(self, config, split):
        print('in setup blender, split: ', split)
        # Initialize attributes
        self.config = config
        self.split = split
        self.rank = get_rank()
        self.scale_factor = 1. / self.config.img_downscale #TODO: temp
        # self.alpha_color = "white"
        self.alpha_color = None
        if self.alpha_color is not None:
            self.alpha_color_tensor = get_color(self.alpha_color)
        else:
            self.alpha_color_tensor = None

        self.data_dir = Path(config['root_dir'])

        if not BlenderDatasetBase.initialized or split == 'test': #we'll initialize again for test phase with test dir
            # load cameras:
            outputs = self._generate_dataparser_outputs(split)
            metadata = outputs.metadata
            H = int(metadata['height'])
            W = int(metadata['width'])

            if 'img_wh' in self.config:
                w, h = self.config.img_wh
                assert round(W / w * h) == H
            elif 'img_downscale' in self.config:
                w, h = int(W / self.config.img_downscale + 0.5), int(H / self.config.img_downscale + 0.5)
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")

            img_wh = (w, h)
            factor = w / W

            all_c2w = outputs.cameras.camera_to_worlds
            fx, fy = outputs.cameras.fx[0]*factor, outputs.cameras.fy[0]*factor
            cx, cy = outputs.cameras.cx[0]*factor, outputs.cameras.cy[0]*factor

            directions = get_ray_directions(w, h, fx, fy, cx, cy)

            filenames = outputs.image_filenames

            has_mask = False
            all_images, all_fg_masks = [], []
            all_fg_indexs, all_bg_indexs = [], []
            all_features = []
            for i, filename in enumerate(filenames):
                # if self.split in ['train', 'val']:
                img = Image.open(filename)
                img = img.resize(img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1,2,0)
                if img.shape[-1] == 4: # contains alpha channel
                    has_mask = True
                    mask = img[..., -1]  # .bool()
                    img = img[...,:-1]
                elif os.path.exists(os.path.join(self.config.root_dir,
                                            self.split,
                                            '.'.join(filename.name.split('.')[:-1])+"_alpha."+filename.name.split('.')[-1])):
                    mask_path = os.path.join(self.config.root_dir, self.split, '.'.join(filename.name.split('.')[:-1])+"_alpha."+filename.name.split('.')[-1])
                    mask = Image.open(mask_path)
                    mask = mask.resize(img_wh, Image.BICUBIC)
                    mask = TF.to_tensor(mask)[0]
                    has_mask = True
                else:
                    mask = torch.ones_like(img[..., 0], device=img.device)
                if self.config.load_features:
                    feat = torch.load(os.path.join(self.config.root_dir,
                                                   f'features_{self.split}_{self.config.img_downscale}',
                                                   filename.name.split('.')[0] + '_feat.pt')).to('cpu')
                    all_features.append(feat)
                fg_index = torch.stack(torch.nonzero(mask.bool(), as_tuple=True), dim=0)
                bg_index = torch.stack(torch.nonzero(~mask.bool(), as_tuple=True), dim=0)
                fg_index = torch.cat([torch.full((1, fg_index.shape[1]), i), fg_index], dim=0)
                bg_index = torch.cat([torch.full((1, bg_index.shape[1]), i), bg_index], dim=0)
                all_fg_indexs.append(fg_index.permute(1, 0))
                all_bg_indexs.append(bg_index.permute(1, 0))
                all_fg_masks.append(mask) # (h, w)
                all_images.append(img)

            apply_mask = has_mask and self.config.apply_mask
            print("apply_mask: ", apply_mask)
            pts3d = metadata['pts3d']
            pts3d_normal = metadata['pts3d_normal']
            pts3d_confidence = metadata['pts3d_confidence']

            pts3d = torch.from_numpy(pts3d).float()
            pts3d_normal = torch.from_numpy(pts3d_normal).float()
            pts3d_confidence = torch.from_numpy(pts3d_confidence).float()

            # When there's no prior point cloud we can't use other estimation methods, if you have a ply then it's not neccesary (not tested)
            assert self.config.up_est_method=="camera"
            assert self.config.center_est_method=="lookat"
            # all_c2w, _, _ = normalize_poses(all_c2w, pts3d, up_est_method=self.config.up_est_method,
            #                                                center_est_method=self.config.center_est_method, pts3d_normal=pts3d_normal)

            BlenderDatasetBase.properties = {
                'w': w,
                'h': h,
                'img_wh': img_wh,
                'factor': factor,
                'has_mask': has_mask,
                'apply_mask': apply_mask,
                'directions': directions,
                'pts3d': pts3d,
                'pts3d_confidence': pts3d_confidence,
                'pts3d_normal': pts3d_normal,
                'all_c2w': all_c2w,
                'all_images': all_images,
                'all_fg_masks': all_fg_masks,
                'all_fg_indexs': all_fg_indexs,
                'all_bg_indexs': all_bg_indexs,
                'all_features': all_features
            }

            BlenderDatasetBase.initialized = True

        for k, v in BlenderDatasetBase.properties.items():
            setattr(self, k, v)

        self.all_c2w = self.all_c2w.float()
        self.all_images, self.all_fg_masks = torch.stack(self.all_images, dim=0).float(), torch.stack(self.all_fg_masks, dim=0).float()
        self.all_features = torch.stack(self.all_features, dim=0) if len(self.all_features) > 0 else None
        self.all_points = self.pts3d.float()
        self.pts3d_normal = self.pts3d_normal.float()
        self.all_points_confidence = self.pts3d_confidence.float()
        self.all_fg_indexs = torch.cat(self.all_fg_indexs, dim=0)
        self.all_bg_indexs = torch.cat(self.all_bg_indexs, dim=0)
        self.all_points_ = contract_to_unisphere(self.all_points, 1.0, ContractionType.AABB)  # points normalized to (0, 1)

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.data_dir / f"transforms_{split}.json")
        # meta = load_from_json(f"{self.data_dir}/transforms_{split}.json")
        image_filenames = []
        poses = []
        for frame in meta["frames"]:
            fname = self.data_dir / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        # camera_to_world[..., 3] *= self.scale_factor
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        metadata = {}
        # if self.config.ply_path is not None:
        metadata.update(self._load_3D_points(self.data_dir / self.config.ply_path))
        metadata.update({
            "height": image_height,
            "width": image_width,
        })

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata=metadata,
        )
        return dataparser_outputs

    def _load_3D_points(self, ply_file_path: Path):
        print('in _load_3D_points, path is: ', ply_file_path)
        if os.path.exists(ply_file_path):
            pcd = o3d.io.read_point_cloud(str(ply_file_path))

            pts3d = np.asarray(pcd.points, dtype=np.float32) * self.scale_factor
            pts3d_normal = np.asarray(pcd.normals, dtype=np.float32)
            # pts3d_rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)

        else:
            pts3d = np.array([], dtype=np.float32)
            pts3d_normal = np.array([], dtype=np.float32)


        out = {
            "pts3d": pts3d,
            "pts3d_normal": pts3d_normal,
            "pts3d_confidence": np.ones([pts3d.shape[0]])
        }

        return out



    def query_radius_occ(self, query_points, radius=0.01):

        num_query = query_points.shape[0]

        # Compute minimum distances
        min_dist, _ = torch.cdist(query_points, self.all_points_).min(dim=1)

        # Create occupancy masks based on min dist
        occ_mask = (min_dist < radius)

        return occ_mask


class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}
import datasets

@datasets.register('blender')
class BlenderDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = BlenderDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDataset(self.config, 'train')

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset,
            num_workers=os.cpu_count(),
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)

