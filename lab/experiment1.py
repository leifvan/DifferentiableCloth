import torch
import arcsim
import sys
import json

from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes

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
run_sim()

verts_tensor = torch.tensor([])
for node in sim.cloths[0].mesh.nodes:
	verts_tensor = torch.cat((verts_tensor, node.x))



faces_tensor = torch.tensor([], dtype=torch.int)

for face in sim.cloths[0].mesh.faces:
	faces_tensor = torch.cat((faces_tensor, torch.tensor(
		[face.vertices[0].index, face.vertices[1].index, face.vertices[2].index])))


verts_tensor = verts_tensor.view(-1,3)
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
tex = TexturesUV(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

print(tex)

meshes = Meshes(verts=[verts_tensor], faces=[faces_tensor], textures=tex)