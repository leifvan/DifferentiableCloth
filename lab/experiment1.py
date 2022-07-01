import torch
import arcsim
import sys
import json

import matplotlib.pyplot as plt
import torch.nn as nn

from pytorch3d.io import load_obj, save_obj, IO, load_objs_as_meshes
from pytorch3d.renderer import TexturesUV, Textures
from pytorch3d.structures import Meshes

import numpy as np

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    SoftSilhouetteShader,
    Textures,
	look_at_rotation
)

sys.path.append('../pysim')

def reset_sim():
	arcsim.init_physics('curconf.json','',False)
	mat = sim.cloths[0].materials[0]
	density = mat.densityori
	stretch = mat.stretchingori
	bend = mat.bendingori
	return density, stretch, bend

def run_sim():
	arcsim.sim_step()


def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)


def to_mesh(vertices, faces):
	pass


with open("./real/S1/sim_conf.json", 'r') as file:
    config = json.load(file)

matfile = config['cloths'][0]['materials'][0]['data']

with open(matfile,'r') as f:
	matconfig = json.load(f)

save_config(matconfig, 'matconfig.json')
config['cloths'][0]['materials'][0]['data'] = 'matconfig.json'
save_config(config, 'curconf.json')


sim=arcsim.get_sim()

density, stretch, bend = reset_sim()
#run_sim()

print(density, stretch, bend)

verts_tensor = torch.tensor([])
for node in sim.cloths[0].mesh.nodes:
	verts_tensor = torch.cat((verts_tensor, node.x))



faces_tensor = torch.tensor([], dtype=torch.int)

for face in sim.cloths[0].mesh.faces:
	faces_tensor = torch.cat((faces_tensor, torch.tensor(
		[face.vertices[0].index, face.vertices[1].index, face.vertices[2].index])))
	print(face.vertices[0].index, face.vertices[1].index, face.vertices[2].index)


verts_tensor = verts_tensor.view(-1,3).to(dtype=torch.float32)
faces_tensor = faces_tensor.view(-1,3)

verts, faces, aux = load_obj("real/S1/templates/template_mesh_final_textured.obj")




verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
tex_maps = aux.texture_images

# tex_maps is a dictionary of {material name: texture image}.
# Take the first image:
texture_image = list(tex_maps.values())[0]
texture_image = texture_image[None, ...]  # (1, H, W, 3)

# Create a textures object

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

verts_rgb_color = torch.ones([1, len(sim.cloths[0].mesh.nodes), 3])
tex = Textures(verts_rgb=verts_rgb_color)
pytorch_mesh = Meshes(verts=[verts_tensor], faces=[faces_tensor], textures=tex)
pytorch_mesh = pytorch_mesh.to("cuda:0")

IO().save_mesh(pytorch_mesh, "test_mesh_1.obj")

camera_position = nn.Parameter(
            torch.from_numpy(np.array([0.0,  0.0, +2.5], dtype=np.float32)).to(device))

R = look_at_rotation(camera_position[None, :], up=((0,1,0),), device=device)
T = -torch.bmm(R.transpose(1, 2), camera_position[None, :, None])[:, :, 0]   # (1, 3)
cameras = FoVPerspectiveCameras(device="cuda:0", R=R, T=T)

raster_settings = RasterizationSettings(
image_size=512, 
blur_radius=0.0, 
faces_per_pixel=1, 
)
lights = PointLights(device="cuda:0", location=[[0.0, 0.0, -3.0]])

renderer = MeshRenderer(
rasterizer=MeshRasterizer(
	cameras=cameras, 
	raster_settings=raster_settings
),
shader=SoftPhongShader(
	device="cuda:0", 
	cameras=cameras,
	lights=lights
)
)
images = renderer(pytorch_mesh)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].detach().cpu().numpy())
plt.axis("off");
plt.show()