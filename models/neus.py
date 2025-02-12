import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models.base import BaseModel
from models.utils import chunk_batch, linear_to_srgb
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching, render_weight_from_density, render_weight_from_alpha, accumulate_along_rays
from nerfacc.intersection import ray_aabb_intersect


class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)


@models.register('neus')
class NeuSModel(BaseModel):
    def setup(self):
        print(f"self.config.learned_background = {self.config.learned_background}")
        self.geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.texture = models.make(self.config.texture.name, self.config.texture)
        self.geometry.contraction_type = ContractionType.AABB

        if self.config.learned_background:
            self.geometry_bg = models.make(self.config.geometry_bg.name, self.config.geometry_bg)
            self.texture_bg = models.make(self.config.texture_bg.name, self.config.texture_bg)
            self.geometry_bg.contraction_type = ContractionType.UN_BOUNDED_SPHERE
            self.near_plane_bg, self.far_plane_bg = 0.1, 1e3
            self.cone_angle_bg = 10**(math.log10(self.far_plane_bg) / self.config.num_samples_per_ray_bg) - 1.
            self.render_step_size_bg = 0.01            

        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )
            if self.config.learned_background:
                self.occupancy_grid_bg = OccupancyGrid(
                    roi_aabb=self.scene_aabb,
                    resolution=256,
                    contraction_type=ContractionType.UN_BOUNDED_SPHERE
                )
        self.randomized = self.config.randomized
        self.background_color = None
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
    
    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.texture, epoch, global_step)
        if self.config.learned_background:
            update_module_step(self.geometry_bg, epoch, global_step)
            update_module_step(self.texture_bg, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[...,None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[...,None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            return alpha
        
        def occ_eval_fn_bg(x):
            density, _ = self.geometry_bg(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size_bg) based on taylor series
            return density[...,None] * self.render_step_size_bg
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.01))
            if self.config.learned_background:
                self.occupancy_grid_bg.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn_bg, occ_thre=self.config.get('grid_prune_occ_thre_bg', 0.01))

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(sdf.shape[0], 1)

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).view(-1).clip(0.0, 1.0)
        return alpha

    def forward_bg_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry_bg(positions)
            return density[...,None]            

        _, t_max = ray_aabb_intersect(rays_o, rays_d, self.scene_aabb)
        # if the ray intersects with the bounding box, start from the farther intersection point
        # otherwise start from self.far_plane_bg
        # note that in nerfacc t_max is set to 1e10 if there is no intersection
        near_plane = torch.where(t_max > 1e9, self.near_plane_bg, t_max)
        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=None,
                grid=self.occupancy_grid_bg if self.config.grid_prune else None,
                sigma_fn=sigma_fn,
                near_plane=near_plane, far_plane=self.far_plane_bg,
                render_step_size=self.render_step_size_bg,
                stratified=self.randomized,
                cone_angle=self.cone_angle_bg,
                alpha_thre=0.0
            )       
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints  
        intervals = t_ends - t_starts

        density, feature = self.geometry_bg(positions) 
        rgb = self.texture_bg(feature, t_dirs)

        weights = render_weight_from_density(t_starts, t_ends, density[...,None], ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_rgb = comp_rgb + self.background_color * (1.0 - opacity)       

        out = {
            'comp_rgb': comp_rgb,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': intervals.view(-1),
                'ray_indices': ray_indices.view(-1)
            })

        return out

    def forward_(self, rays, clip_scales=None):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=None,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        if clip_scales is not None:
            with torch.no_grad():
                clip_scales = clip_scales[ray_indices] * dists.norm(dim=-1, keepdim=True)
        elif (not self.training) and self.config.get('lerf', False):
            clip_scales = torch.ones_like(dists, device=dists.device) 
        
        geo_out = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True)
        if len(geo_out) == 4:
            sdf, sdf_grad, feature, sdf_laplace = geo_out
            pred_normals = None
        else: # contain pred_normals
            sdf, sdf_grad, pred_normals, feature, sdf_laplace = geo_out
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]
        # if self.config.only_features:
        #     normal = normal.detach()
            # alpha = alpha.detach()
            # sdf_laplace = sdf_laplace.detach()

        tex_out = self.texture(feature, t_dirs, normal, clip_scales=clip_scales, with_split_color=True)
        if self.config.get('lerf', False):
            tex_out, lerf_out = tex_out[:-2], tex_out[-2:]
        # split outputs
        is_refnerf_like = len(tex_out) > 2
        learn_sem_features = len(tex_out) == 2 or len(tex_out) > 3        
        rgb = tex_out[0]
        indep_color = tex_out[1] if is_refnerf_like else torch.zeros_like(rgb)
        ref_color = tex_out[2] if is_refnerf_like else torch.zeros_like(rgb)
        total_features = None

        if learn_sem_features:
            if (not is_refnerf_like): # dff like
                total_features = tex_out[1]
                indep_features = tex_out[1]
                ref_features = torch.zeros_like(total_features)
            else:                
                if len(tex_out) == 4:
                    total_features = tex_out[3]
                    indep_features = tex_out[3]
                    ref_features = torch.zeros_like(total_features)                    
                else:
                    indep_features = tex_out[3]
                    ref_features = tex_out[4] 
                    total_features = indep_features + ref_features

        if total_features is not None:
            ntex_out =  rgb, indep_color, ref_color, total_features, indep_features, ref_features
            
            if 'edit_config' in self.config:
                if self.config.edit_config.edit_type == 'color':
                    rgb, indep_color, ref_color, total_features, indep_features, ref_features = self.segment_color(ntex_out, self.config.edit_config)
                if self.config.edit_config.edit_type == 'remove':
                    rgb, indep_color, ref_color, total_features, indep_features, ref_features = self.segment_remove(ntex_out, self.config.edit_config)
                if self.config.edit_config.edit_type == 'roughness':
                    # segment_rougness(self, old_tex_out, edit, feature, t_dirs, normal, clip_scales=None)
                    rgb, indep_color, ref_color, total_features, indep_features, ref_features = self.segment_rougness(ntex_out, self.config.edit_config, feature, t_dirs, normal, clip_scales=None)

        # if total_features is not None and 'edit_config' in self.config:
        #         for edit in self.config.edit_config.edits:
        #             ntex_out =  rgb, indep_color, ref_color, total_features, indep_features, ref_features
        #             if edit.edit_type == 'color':
        #                 rgb, indep_color, ref_color, total_features, indep_features, ref_features = self.segment_color(ntex_out, self.config.edit_config)

        indep_color = torch.clip(linear_to_srgb(indep_color), 0.0, 1.0)
        ref_color = torch.clip(linear_to_srgb(ref_color), 0.0, 1.0)

        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)

        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)
        comp_indep_color = accumulate_along_rays(weights, ray_indices, values=indep_color, n_rays=n_rays)
        comp_ref_color = accumulate_along_rays(weights, ray_indices, values=ref_color, n_rays=n_rays)
        
        rays_fg = opacity > 0.1
        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)
        comp_normal[~rays_fg[:, 0]] = 0


        out = {
            'comp_rgb': comp_rgb,
            'comp_indep_color': comp_indep_color,
            'comp_ref_color': comp_ref_color,
                        
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),            
        }
        

        if learn_sem_features:
            comp_feature = accumulate_along_rays(weights.detach(), ray_indices, values=total_features, n_rays=n_rays)
            
            comp_dep_features = accumulate_along_rays(weights, ray_indices, values=ref_features, n_rays=n_rays)            
            comp_indep_features = accumulate_along_rays(weights, ray_indices, values=indep_features, n_rays=n_rays)            
            
            out.update({
                'features': comp_feature, # total_sem_feat
                'dep_features': comp_dep_features, # ref_sem_feat
                'indep_features': comp_indep_features, # indep_sem_feat                
            })
          
        if self.config.get('lerf', False):
            lerf_comb = lerf_out[0] + lerf_out[1]
            lerf_comb = lerf_comb / lerf_comb.norm(dim=-1, keepdim=True)
            lerf_indep = lerf_out[0] / lerf_out[0].norm(dim=-1, keepdim=True)
            lerf_dep = lerf_out[1] / lerf_out[1].norm(dim=-1, keepdim=True)
            
            comp_lerf_indep = accumulate_along_rays(weights, ray_indices, values=lerf_indep, n_rays=n_rays)
            comp_lerf_dep = accumulate_along_rays(weights, ray_indices, values=lerf_dep, n_rays=n_rays)
            comp_lerf_total = accumulate_along_rays(weights.detach(), ray_indices, values=lerf_comb, n_rays=n_rays)

            comp_lerf_indep = F.normalize(comp_lerf_indep, p=2, dim=-1)
            comp_lerf_dep = F.normalize(comp_lerf_dep, p=2, dim=-1)
            comp_lerf_total = F.normalize(comp_lerf_total, p=2, dim=-1)

            out.update({
                'clip_features_indep': comp_lerf_indep,
                'clip_features_dep': comp_lerf_dep,
                'clip_features': comp_lerf_total
            })

        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'sdf_normals': normal,
                'pred_normals': pred_normals,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1)                
            })
            out.update({
                'sdf_laplace_samples': sdf_laplace
            })

        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                'comp_rgb': self.background_color[None,:].expand(*comp_rgb.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }

        if self.config.only_features:
            for k,v in out.items():
                if 'features' not in k and v is not None:
                    out[k] = v.detach()
            for k,v in out_bg.items():
                if 'features' not in k and v is not None:
                    out_bg[k] = v.detach()

        out_full = {
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }

    
    def get_segmentation_mask(self, indep_sem_feat, ref_sem_feat, total_sem_feat):
        
        mask = None
        sample_factor = self.config.edit_config.get('sample_factor', 1)
        for segment in self.config.edit_config.segments:
            
            
            sem_feat = ref_sem_feat if segment.seg_type == 'dep' \
                    else indep_sem_feat if segment.seg_type == 'indep' \
                        else total_sem_feat
            
            feat_path = segment.feat_path + f"/{segment.seg_type}_features/{segment.seg_type}-{segment.feat_img_numb}.pt"
            seg_features_img = torch.load(feat_path).to(sem_feat.device) # [H, W, 384]
                                                        
            pix_indices = segment.pix_indices
            indices = torch.tensor(pix_indices, device=sem_feat.device)  
            # print(indices.shape)
            torch.int
            threshold = segment.threshold # 0.75

            seg_features = seg_features_img[torch.round(indices[:,0] / sample_factor).to(torch.int), torch.round(indices[:,1] / sample_factor).to(torch.int)] # [p, 384]
            # print("============",seg_features.shape, "============")
            # print("============",sem_feat.shape, "============")
            seg_features = F.normalize(seg_features, p=2, dim=-1) # [p, 384]
            normed_features = F.normalize(sem_feat, p=2, dim=-1) 

            # max over the second dimension
            # print("==============",normed_features.shape, seg_features.shape,"==============")
            features_similarity, _ = torch.max(torch.matmul(normed_features, seg_features.T), dim=1) # [n]

            
            if mask is None:
                mask = features_similarity >= threshold
            elif segment.get('mask_op', 'or') == 'or':
                mask = mask | (features_similarity >= threshold)
            elif segment.get('mask_op', 'or') == 'exclude':
                mask = mask & (features_similarity < threshold)
            else: # and
                mask = mask & (features_similarity >= threshold)          
        
        return mask

    def segment_color(self, tex_out, edit):
        rgb, indep_color, ref_color, total_features, indep_features, ref_features = tex_out

        
        mask = self.get_segmentation_mask(indep_features, ref_features, total_features)
        
        target_color = torch.tensor(edit.get("color", [0.0, 1.0, 0.0]), device=rgb.device)

        if "total" in edit.on_layers:
            rgb = rgb.detach()
            rgb[mask] = target_color
        else:            
            if "indep" in edit.on_layers:
                indep_color = indep_color.detach()
                indep_color[mask] = target_color
            if "dep" in edit.on_layers:
                ref_color = ref_color.detach()
                ref_color[mask] = target_color
        
            if self.config.texture.name == 'refnerf-color':
                rgb = torch.clip(linear_to_srgb(ref_color + indep_color), 0.0, 1.0)
            elif self.config.texture.name == "refnerf-color-no-tone-map":
                rgb = torch.sigmoid(ref_color + indep_color)
            else: # regular
                rgb = indep_color
            
        return rgb, indep_color, ref_color, total_features, indep_features, ref_features

    def segment_remove(self, tex_out, edit):
        rgb, indep_color, ref_color, total_features, indep_features, ref_features = tex_out

        
        mask = self.get_segmentation_mask(indep_features, ref_features, total_features)
        
        target_color = torch.tensor(edit.get("color", [0.0, 0.0, 0.0]), device=rgb.device)

        if "total" in edit.on_layers:
            rgb = rgb.detach()
            rgb[mask] = target_color
        else:            
            if "indep" in edit.on_layers:
                indep_color = indep_color.detach()
                indep_color[mask] = target_color
            if "dep" in edit.on_layers:
                ref_color = ref_color.detach()
                ref_color[mask] = target_color
        
            if self.config.texture.name == 'refnerf-color':
                rgb = torch.clip(linear_to_srgb(ref_color + indep_color), 0.0, 1.0)
            else: # regular
                rgb = indep_color
            
        return rgb, indep_color, ref_color, total_features, indep_features, ref_features 

    def segment_extract1(self, tex_out, edit):
        lerf = self.config.get('lerf', False)
        if not lerf:
            rgb, cam_color, indep_color, ref_color, diffuse_color, ref_weights, indep_weight, total_sem_feat, cam_sem_feat, ref_sem_feat, indep_sem_feat = tex_out
        else:
            rgb, cam_color, indep_color, ref_color, diffuse_color, ref_weights, indep_weight, total_sem_feat, cam_sem_feat, ref_sem_feat, indep_sem_feat, \
                total_clip_feat, cam_clip_feat, ref_clip_feat, indep_clip_feat = tex_out
        mask = self.get_segmentation_mask(cam_sem_feat, ref_sem_feat, indep_sem_feat, total_sem_feat, ref_weights, indep_weight)
        
        rgb = rgb.detach()
        rgb[~mask] = torch.tensor([0.0, 0.0, 0.0], device=rgb.device)
        cam_color = cam_color.detach()
        cam_color[~mask] = torch.tensor([0.0, 0.0, 0.0], device=cam_color.device)        
        indep_color = indep_color.detach()
        indep_color[~mask] = torch.tensor([0.0, 0.0, 0.0], device=indep_color.device)
        ref_color = ref_color.detach()
        ref_color[~mask] = torch.tensor([0.0, 0.0, 0.0], device=ref_color.device)
        
        if not lerf:
            return rgb, cam_color, indep_color, ref_color, diffuse_color, ref_weights, indep_weight, total_sem_feat, cam_sem_feat, ref_sem_feat, indep_sem_feat
        else:
            return rgb, cam_color, indep_color, ref_color, diffuse_color, ref_weights, indep_weight, total_sem_feat, cam_sem_feat, ref_sem_feat, indep_sem_feat, \
                total_clip_feat, cam_clip_feat, ref_clip_feat, indep_clip_feat
      
    def segment_remove1(self, tex_out, edit):
        lerf = self.config.get('lerf', False)
        if not lerf:
            rgb, cam_color, indep_color, ref_color, diffuse_color, ref_weights, indep_weight, total_sem_feat, cam_sem_feat, ref_sem_feat, indep_sem_feat = tex_out
        else:
            rgb, cam_color, indep_color, ref_color, diffuse_color, ref_weights, indep_weight, total_sem_feat, cam_sem_feat, ref_sem_feat, indep_sem_feat, \
                total_clip_feat, cam_clip_feat, ref_clip_feat, indep_clip_feat = tex_out
        mask = self.get_segmentation_mask(cam_sem_feat, ref_sem_feat, indep_sem_feat, total_sem_feat, ref_weights, indep_weight)
        pass_energy = edit.pass_energy
        
        new_ref_weights = ref_weights.clone()
        new_indep_weight = indep_weight.clone()

        indep_part = indep_weight * indep_color
        cam_part = (1-indep_weight) * cam_color

        diffuse_part_weights = (1-ref_weights) 
        if "dep" in edit.on_layers: # ref
            new_ref_weights[mask] = 0
            if pass_energy > 0:
                diffuse_part_weights = 1-ref_weights 
                diffuse_part_weights[mask] += (ref_weights*pass_energy)[mask]

        if "indep" in edit.on_layers: # indep
            new_indep_weight[mask] = 0
            indep_part = new_indep_weight * indep_color
            if pass_energy > 0:
                cam_part_weights = 1-indep_weight
                cam_part_weights[mask] += (indep_weight*pass_energy)[mask]
                cam_part = cam_part_weights * cam_color                
        elif "cam" in edit.on_layers: # cam
            new_indep_weight[mask] = 1
            cam_part = (1-new_indep_weight) * cam_color
            if pass_energy > 0:
                indep_part_weights = indep_weight
                indep_part_weights[mask] += ((1-indep_weight)*pass_energy)[mask]
                indep_part = indep_part_weights * indep_color
        
        if "cam" and "indep" in edit.on_layers: # cam-indep
            new_ref_weights[mask] = ref_weights[mask] + (1-ref_weights[mask]) * pass_energy
        
        # if pass_energy > 0:
        rgb = new_ref_weights * ref_color + diffuse_part_weights * (indep_part + cam_part)
        
        if not lerf:
            return rgb, cam_color, indep_color, ref_color, diffuse_color, ref_weights, indep_weight, total_sem_feat, cam_sem_feat, ref_sem_feat, indep_sem_feat
        else:
            return rgb, cam_color, indep_color, ref_color, diffuse_color, ref_weights, indep_weight, total_sem_feat, cam_sem_feat, ref_sem_feat, indep_sem_feat, \
                total_clip_feat, cam_clip_feat, ref_clip_feat, indep_clip_feat

    def segment_rougness(self, old_tex_out, edit, feature, t_dirs, normal, clip_scales=None):
        rgb, indep_color, ref_color, total_features, indep_features, ref_features = old_tex_out

        
        mask = self.get_segmentation_mask(indep_features, ref_features, total_features)
        
        roughness_multiplier = edit.edit_gamma
        tex_out = self.texture(feature, t_dirs, normal, clip_scales=clip_scales, with_split_color=True, roughness_multiplier=roughness_multiplier)
        _, _, n_ref_color, _, _ = tex_out

        ref_color[mask] = n_ref_color[mask]

        rgb = torch.clip(linear_to_srgb(ref_color + indep_color), 0.0, 1.0)

        return rgb, indep_color, ref_color, total_features, indep_features, ref_features         
 
    def forward(self, rays, clip_scales=None):
        if self.training:
            out = self.forward_(rays, clip_scales=clip_scales)
        else:
            out = chunk_batch(self.forward_, self.config.ray_chunk, True, rays)
        return {
            **out,
            'inv_s': self.variance.inv_s
        }

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, features = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            base_color = torch.sigmoid(features[..., 1:4])
            mesh['v_rgb'] = base_color.cpu()
            mesh['v_norm'] = normal.cpu()
        return mesh

@models.register('sh-neus')
class SphericalHarmonicNeuSModel(NeuSModel):
    def forward_(self, rays):
        n_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=None,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0
            )
        
        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs * midpoints
        dists = t_ends - t_starts

        sdf, sdf_grad, feature, sdf_laplace, auxiliary_feature = self.geometry(positions, with_grad=True, with_feature=True, with_laplace=True, with_auxiliary_feature=True)
        normal = F.normalize(sdf_grad, p=2, dim=-1)
        alpha = self.get_alpha(sdf, normal, t_dirs, dists)[...,None]
        rgb, sh_coeff = self.texture(feature, t_dirs, normal)
        auxiliary_sh_coeff = self.texture.get_sh_coeff(auxiliary_feature, normal)

        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, ray_indices, values=rgb, n_rays=n_rays)

        comp_normal = accumulate_along_rays(weights, ray_indices, values=normal, n_rays=n_rays)
        comp_normal = F.normalize(comp_normal, p=2, dim=-1)
        
        out = {
            'comp_rgb': comp_rgb,
            'comp_normal': comp_normal,
            'opacity': opacity,
            'depth': depth,
            'rays_valid': opacity > 0,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device)
        }

        if self.training:
            out.update({
                'sdf_samples': sdf,
                'sdf_grad_samples': sdf_grad,
                'weights': weights.view(-1),
                'points': midpoints.view(-1),
                'intervals': dists.view(-1),
                'ray_indices': ray_indices.view(-1),
                'sh_coeff': sh_coeff,
                'auxiliary_sh_coeff': auxiliary_sh_coeff   
            })
            out.update({
                'sdf_laplace_samples': sdf_laplace
            })

        if self.config.learned_background:
            out_bg = self.forward_bg_(rays)
        else:
            out_bg = {
                'comp_rgb': self.background_color[None,:].expand(*comp_rgb.shape),
                'num_samples': torch.zeros_like(out['num_samples']),
                'rays_valid': torch.zeros_like(out['rays_valid'])
            }

        out_full = {
            'comp_rgb': out['comp_rgb'] + out_bg['comp_rgb'] * (1.0 - out['opacity']),
            'num_samples': out['num_samples'] + out_bg['num_samples'],
            'rays_valid': out['rays_valid'] | out_bg['rays_valid']
        }

        return {
            **out,
            **{k + '_bg': v for k, v in out_bg.items()},
            **{k + '_full': v for k, v in out_full.items()}
        }
        
    def regularizations(self, out):
        losses = {}
        losses['sh_mse'] = F.mse_loss(out['sh_coeff'], out['auxiliary_sh_coeff'])
        losses.update(self.geometry.regularizations(out))
        losses.update(self.texture.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_vertex_color:
            _, sdf_grad, feature = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True)
            normal = F.normalize(sdf_grad, p=2, dim=-1)
            sh_coeff = self.texture.get_sh_coeff(feature, normal) # set the viewing directions to the normal to get "albedo"
            base_color = sh_coeff[..., :3] * 0.28209479177387814 + 0.5
            mesh['v_rgb'] = base_color.cpu()
        return mesh
