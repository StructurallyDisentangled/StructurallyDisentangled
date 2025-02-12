import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy


@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays
        self.sample_foreground_ratio = self.config.dataset.get('sample_foreground_ratio', 1.0)

    def forward(self, batch):
        out = self.model(batch['rays'], clip_scales=batch.get('clip_scale', None))
        if self.config.get('model_2', None) is not None:
            out_2 = self.model_2(batch['rays'], clip_scales=batch.get('clip_scale', None))
            return (out, out_2)
        return (out,)
    
    def predict_step(self, batch, batch_idx):
        out = self(batch)

        if self.config.system.get('selected_model', 0) == 1:
            out = out[1]
        else:
            out = out[0]

        W, H = self.dataset.img_wh

        # self.save_image_grid(f"it{self.global_step}-predict/rgb/rgb-normal-{batch['index'][0].item()}.png", [
        #     {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},])
        self.save_image_grid(f"it{self.global_step}-predict/rgb_full/rgb-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},])
        
        self.save_image_grid(f"it{self.global_step}-predict/indep/indep-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': out['comp_indep_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            ]) 
        
        self.save_image_grid(f"it{self.global_step}-predict/dep/dep-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': out['comp_ref_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},            
            ]) 
        # rgb, normals
        # self.save_image_grid(f"it{self.global_step}-predict/basics/rgb-normal-{batch['index'][0].item()}.png", [
        #     {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        #     ])
        # # diff, cam, diff+cam ,ref
        # self.save_image_grid(f"it{self.global_step}-predict/dep-indep/dep-indep-{batch['index'][0].item()}.png", [
        #     {'type': 'rgb', 'img': out['comp_indep_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     # {'type': 'rgb', 'img': out['comp_cam_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     # {'type': 'rgb', 'img': ((out['comp_cam_color'] +out['comp_indep_color'])/2).view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},            
        #     {'type': 'rgb', 'img': out['comp_ref_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},            
        #     # {'type': 'grayscale', 'img': out['comp_ref_weights'].view(H, W), 'kwargs': {}}
        #     ])        

        # features
        
        # feat_path = self.get_save_path(f"it{self.global_step}-predict/features/total-{batch['index'][0].item()}.pt")
        # torch.save(out['features'].view(H, W, self.config.system.semantic_feature_dim), feat_path)
        
        # feat_path = self.get_save_path(f"it{self.global_step}-predict/dep_features/dep-{batch['index'][0].item()}.pt")
        # torch.save(out['dep_features'].view(H, W, self.config.system.semantic_feature_dim), feat_path)
        
        # feat_path = self.get_save_path(f"it{self.global_step}-predict/indep_features/indep-{batch['index'][0].item()}.pt")
        # torch.save(out['indep_features'].view(H, W, self.config.system.semantic_feature_dim), feat_path)
        
    def predict_epoch_end(self, out):
        print("Predict epoch end")
     
    def preprocess_data(self, batch, stage):
        batch_features = None
        clip_batch_features = None
        # torch.cuda.empty_cache()
        if 'index' in batch: # validation / testing
            index = batch['index'].cpu()
        else:
            if self.config.model.batch_image_sampling:
                if self.sample_foreground_ratio < 1:
                    fg_ray_index = torch.randint(0, len(self.dataset.all_fg_indexs), size=(int(self.train_num_rays * 0.8),))
                    bg_ray_index = torch.randint(0, len(self.dataset.all_bg_indexs), size=(self.train_num_rays - int(self.train_num_rays * 0.8),))
                    
                    fg_ray_index = self.dataset.all_fg_indexs[fg_ray_index]
                    bg_ray_index = self.dataset.all_bg_indexs[bg_ray_index]
                    ray_index = torch.cat([fg_ray_index, bg_ray_index], dim=0)
                    index, y, x = ray_index[:, 0], ray_index[:, 1], ray_index[:, 2]
                else:
                    index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,))
                    x = torch.randint(
                        0, self.dataset.w, size=(self.train_num_rays,)
                    )
                    y = torch.randint(
                        0, self.dataset.h, size=(self.train_num_rays,)
                    )
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,))
                x = torch.randint(
                        0, self.dataset.w, size=(self.train_num_rays,)
                    )
                y = torch.randint(
                    0, self.dataset.h, size=(self.train_num_rays,)
                )
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            
            # sample the same number of points as the ray
            if len(self.dataset.all_points) >= 3:  # for the case we load a blender dataset (no prior pointcloud)
                pts_index = torch.randint(0, len(self.dataset.all_points), size=(self.train_num_rays,))
                pts = self.dataset.all_points[pts_index]
                pts_weights = self.dataset.all_points_confidence[pts_index]
                if self.dataset.pts3d_normal is not None:
                    pts_normal = self.dataset.pts3d_normal[pts_index]
                else:
                    pts = torch.tensor([])
            else:
                pts = torch.tensor([])
                pts_weights = torch.tensor([])
                pts_normal = torch.tensor([])
                
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            if self.config.system.train_features:
                w, h = self.dataset.img_wh
                
                img_scale = (
                    self.dataset.all_features.shape[-2] / w, # x
                    self.dataset.all_features.shape[-3] / h # y
                )

                x_ind = (x * img_scale[0]).long()
                y_ind = (y * img_scale[1]).long()

                feat_indices = index, y_ind, x_ind
                
                batch_features = self.dataset.all_features[feat_indices].view(-1, self.dataset.all_features.shape[-1]).to(self.rank)

                if self.config.system.get('lerf', False):                    
                    clip_batch_features, clip_scale = self.dataset.clip_interpolator(torch.stack((index, x_ind, y_ind)).permute(-1,0))
                    clip_scale = clip_scale * self.dataset.img_h / self.dataset.fy

            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            pts = torch.tensor([])
            pts_weights = torch.tensor([])
            pts_normal = torch.tensor([])
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32)
                if self.config.get('model_2', None) is not None:
                    self.model_2.background_color = torch.ones((3,), dtype=torch.float32)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32)
                if self.config.get('model_2', None) is not None:
                    self.model_2.background_color = torch.rand((3,), dtype=torch.float32)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32)
            if self.config.get('model_2', None) is not None:
                self.model_2.background_color = torch.ones((3,), dtype=torch.float32)
        
        self.model.background_color = self.model.background_color.to(self.device)
        if self.config.get('model_2', None) is not None:
            self.model_2.background_color = self.model_2.background_color.to(self.device)
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        if self.config.system.train_features:
            if batch_features is not None:
                batch.update({
                    'features': batch_features
                })
            if clip_batch_features is not None:
                batch.update({
                    'clip_features': clip_batch_features,
                    'clip_scale': clip_scale
                })
            

        batch.update({
            'rays': rays.to(self.device),
            'rgb': rgb.to(self.device),
            'fg_mask': fg_mask.to(self.device),
            'pts': pts.to(self.device),
            'pts_normal': pts_normal.to(self.device),
            'pts_weights': pts_weights.to(self.device)
        })      
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        if self.config.system.get('selected_model', 0) == 1:
            out = out[1]
        else:
            out = out[0]

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        ## normals loss
        if self.C(self.config.system.loss.get('lambda_normal_prediction', 0)) > 0 and (out.get('pred_normals', None) is not None):
            n = out['sdf_normals']            
            loss_normal_pred = torch.mean(
            (out['weights'] * (1.0 - torch.sum(n * out['pred_normals'], dim=-1))).sum(dim=-1))

            loss += loss_normal_pred * self.C(self.config.system.loss.lambda_normal_prediction)
            del n, loss_normal_pred
            
        if self.C(self.config.system.loss.get('lambda_normal_orientation', 0)) > 0:
            n = out['sdf_normals']            
            zero = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            v=-batch['rays'][...,-3:]
            n_dot_v = (n * v[..., None, :]).sum(dim=-1)
            loss_normal_orient = torch.mean((out['weights'] * torch.minimum(zero, n_dot_v)**2).sum(dim=-1))

            loss += loss_normal_orient * self.C(self.config.system.loss.lambda_normal_orientation)
            del n, v, n_dot_v, loss_normal_orient


        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        # features loss
        # change lambda in the config file
        if self.config.system.train_features and self.config.system.get('add_features_after', 0) <= self.global_step:
            if self.C(self.config.system.loss.get('lambda_features',0)) > 0:
                # loss_features = F.mse_loss(out['features'], batch['features'])             
                loss_features = F.mse_loss(out['features'][out['rays_valid'][...,0]], batch['features'][out['rays_valid'][...,0]])
                self.log('train/loss_features', loss_features)
                loss += loss_features * self.C(self.config.system.loss.lambda_features)
            if self.C(self.config.system.loss.get('lambda_clip_features',0)) > 0:
                loss_clip_features = F.huber_loss(
                    out['clip_features'][out['rays_valid'][...,0]], 
                    batch['clip_features'][out['rays_valid'][...,0]],
                    delta=1.25, reduction="none")
                loss_clip_features = loss_clip_features.sum(dim=-1).nanmean()
                self.log('train/loss_clip_features', loss_clip_features)
                loss += loss_clip_features * self.C(self.config.system.loss.lambda_clip_features)

        if self.config.system.get('selected_model', 0) == 1:
            losses_model_reg = self.model_2.regularizations(out)
        else:
            losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        # sdf loss proposed in Geo-Neus and normal loss proposed in regsdf
        if self.C(self.config.system.loss.lambda_sdf_l1) > 0:
            pts = batch['pts']
            pts_normal = batch['pts_normal']
            pts_weights = batch['pts_weights']
            pts2sdf, pts2sdf_grad = self.model.geometry(pts, with_grad=True, with_feature=False)
            loss_sdf = F.l1_loss(pts2sdf, torch.zeros_like(pts2sdf)) * pts_weights
            loss_sdf = loss_sdf.mean(dim=0)
            
            normal_gt = torch.nn.functional.normalize(pts_normal, p=2, dim=-1)
            normal_pred = torch.nn.functional.normalize(pts2sdf_grad, p=2, dim=-1)
            loss_normal = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
            self.log('train/loss_sdf_l1', loss_sdf)
            self.log('train/loss_normal_cos', loss_normal)
            loss += loss_sdf * self.C(self.config.system.loss.lambda_sdf_l1)
            loss += loss_normal * self.C(self.config.system.loss.get('lambda_normal', self.config.system.loss.lambda_sdf_l1))

        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        # torch.cuda.empty_cache()
        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        if self.config.system.get('selected_model', 0) == 1:
            out = out[1]
        else:
            out = out[0]

        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"validation/it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] +[
            {'type': 'rgb', 'img': out['comp_indep_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            # {'type': 'rgb', 'img': out['comp_cam_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            # {'type': 'rgb', 'img': ((out['comp_cam_color'] +out['comp_indep_color'])/2).view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            
            {'type': 'rgb', 'img': out['comp_ref_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},            
            # {'type': 'grayscale', 'img': out['comp_ref_weights'].view(H, W), 'kwargs': {}},

        ]+ ([
            # {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            # {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            # {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])

        # self.save_image_grid(f"validation/sep-it{self.global_step}-{batch['index'][0].item()}.png", [
        #     {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     {'type': 'rgb', 'img': out['comp_indep_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     # {'type': 'rgb', 'img': out['comp_cam_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     # {'type': 'rgb', 'img': out['comp_diffuse_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     {'type': 'rgb', 'img': out['comp_ref_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     # {'type': 'grayscale', 'img': out['comp_ref_weights'].view(H, W), 'kwargs': {}},
        #     # {'type': 'grayscale', 'img': out['comp_diffuse_weight'].view(H, W), 'kwargs': {}},
        # ])
        del out
        torch.cuda.empty_cache()
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    def test_step(self, batch, batch_idx):
        
        if not self.config.system.force_export_all:
            if self.config.system.get('export_frames', None):
                if batch['index'][0] not in self.config.system.export_frames:
                    return {
                        'psnr': 0,
                        'index': batch['index']
                    }

            
            # print("here! test_step")
            if (batch['index'][0] % self.config.system.get('skip_every_n_frames', 1) != 0) or \
                (batch['index'][0] < self.config.system.get('start_frame', 0)) or\
                (batch['index'][0] in self.config.system.get('skip_frames', [])):
                return {
                    'psnr': 0,
                    'index': batch['index']
                } 
            
        ## iterate over the batch 
        ### create smaller batches with 'rays' and 'clip_scale' if exists

        out = {
            'comp_rgb_full': [],
            'comp_indep_color': [],
            'comp_ref_color': [],
            'comp_normal': [],
            'features': [],
            'dep_features': [],
            'indep_features': [],
            'clip_features_indep': [],
            'clip_features_dep': [],
            'clip_features': [],
        }

        rays_count = len(batch['rays'])
        curr_idx = 0
        num_rays = 512 # 512

        while curr_idx < rays_count:
            # print(f"Processing {curr_idx}/{rays_count}")
            next_idx = min(curr_idx + num_rays, rays_count)
            sub_batch = {
                'rays': batch['rays'][curr_idx:next_idx],
                'clip_scale': batch.get('clip_scale', [None]*rays_count)[curr_idx:next_idx],
            }
            
            curr_idx = next_idx

            out_sub = self(sub_batch)
            if self.config.system.get('selected_model', 0) == 1:
                out_sub = out_sub[1]
            else:
                out_sub = out_sub[0]

            for k, v in out.items():
                if k not in out_sub:
                    continue
                out[k].append(out_sub[k].to('cpu'))

        for k, v in out.items():
            if len(v) == 0:
                continue
            out[k] = torch.cat(v, dim=0)

        # out = self(batch)
        # if self.config.system.get('selected_model', 0) == 1:
        #     out = out[1]
        # else:
        #     out = out[0]

        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh        

        
        self.save_image_grid(f"rgb/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ])

        self.save_image_grid(f"indep/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': out['comp_indep_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ])

        # self.save_image_grid(f"cam/{batch['index'][0].item()}.png", [
        #     {'type': 'rgb', 'img': out['comp_cam_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        # ])

        self.save_image_grid(f"dep/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': out['comp_ref_color'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ])

        self.save_image_grid(f"normal/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
            

        # features
        if "features" in out and \
                ('export_features' in (self.config.system) and self.config.system.export_features):
            feat_path = self.get_save_path(f"it{self.global_step}-feat/total_features/total-{batch['index'][0].item()}.pt")
            torch.save(out['features'].view(H, W, self.config.system.semantic_feature_dim), feat_path)
            
            feat_path = self.get_save_path(f"it{self.global_step}-feat/dep_features/dep-{batch['index'][0].item()}.pt")
            torch.save(out['dep_features'].view(H, W, self.config.system.semantic_feature_dim), feat_path)
            
            feat_path = self.get_save_path(f"it{self.global_step}-feat/indep_features/indep-{batch['index'][0].item()}.pt")
            torch.save(out['indep_features'].view(H, W, self.config.system.semantic_feature_dim), feat_path)
            
            if self.config.system.get('lerf', False):
                feat_path = self.get_save_path(f"it{self.global_step}-feat-clip/indep_features/indep-{batch['index'][0].item()}.pt")
                torch.save(out['clip_features_indep'].view(H, W, 512), feat_path)

                feat_path = self.get_save_path(f"it{self.global_step}-feat-clip/dep_features/dep-{batch['index'][0].item()}.pt")
                torch.save(out['clip_features_dep'].view(H, W, 512), feat_path)

                feat_path = self.get_save_path(f"it{self.global_step}-feat-clip/total_features/total-{batch['index'][0].item()}.pt")
                torch.save(out['clip_features'].view(H, W, 512), feat_path)

        torch.cuda.empty_cache()
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            for test_name in ["rgb", "indep", "dep", "normal"]:
                self.save_img_sequence(
                    test_name,
                    test_name,
                    '(\d+)\.png',
                    save_format='mp4',
                    fps=30
                )
            
            self.export()
   
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )        

    def render(self, datamodule):
        
        self.model.eval()
        
        self.dataset = datamodule.test_dataloader().dataset
        imgs_count = len(self.dataset)
        for i in range(imgs_count):
            batch = self.dataset[i]
            self.preprocess_data(batch, 'test')

            # TODO: split batch to avoid OOM
            # call self to get the output
            # patch all output to final images
            # save the final images
            
                
   
