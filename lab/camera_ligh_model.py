import torch
import pytorch3d
import torch.nn as nn
import numpy as np

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    softmax_rgb_blend
)

import matplotlib.pyplot as plt

from pytorch3d.transforms import Rotate, Translate



class Estimator(nn.Module):

    def __init__(self, meshes, renderer, image_ref) -> None:
        super().__init__()

        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer

        image_ref_normal = torch.from_numpy(image_ref,)
        image_ref = torch.from_numpy((image_ref[..., :3].max(-1) >= 0.2).astype(np.float32))
        
        self.register_buffer('image_ref_normal', image_ref_normal)
        self.register_buffer('image_ref', image_ref)

        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([0.0,  0.0, +2.5], dtype=np.float32)).to(meshes.device))

    def forward(self):
        R = look_at_rotation(self.camera_position[None, :], device=self.device)  # (1, 3, 3)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]   # (1, 3)

        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        loss = torch.sum((image[..., 3] - self.image_ref) ** 2)

        return loss, image

class Lightless_Shader(torch.nn.Module):

    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"
            raise ValueError(msg)
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = softmax_rgb_blend(texels, fragments, blend_params)

        return images