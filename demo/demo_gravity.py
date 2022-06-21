import torch
import arcsim
import gc
import time
import json
import sys

from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
from pytorch3d.ops import sample_points_from_meshes

from mpl_toolkits.mplot3d import Axes3D

device = ("cuda" if torch.cuda.is_available() else "cpu")

sys.path.append('../pysim')

with open('conf/gravity.json','r') as f:
	config = json.load(f)
matfile = config['cloths'][0]['materials'][0]['data']
with open(matfile,'r') as f:
	matconfig = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(matconfig, 'curmat.json')
config['cloths'][0]['materials'][0]['data'] = 'curmat.json'
save_config(config, 'curconf.json')


torch.set_num_threads(8)
sim=arcsim.get_sim()

def reset_sim():
	arcsim.init_physics('curconf.json','',False)
	g = sim.gravity
	g.requires_grad = True
	return g

def run_sim():
	for step in range(49):
		arcsim.sim_step()

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes([mesh], 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)     
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

def get_loss():
	vec = torch.tensor([0,0,-1],dtype=torch.float64)
	ans = torch.zeros([],dtype=torch.float64)
	cnt = 0
	for node in sim.cloths[0].mesh.nodes:
		cnt += 1
		ans = ans + torch.dot(node.x,vec)

	verts_tensor = torch.tensor([])
	for node in sim.cloths[0].mesh.nodes:
		verts_tensor = torch.cat((verts_tensor, node.x))
	


	faces_tensor = torch.tensor([], dtype=torch.int)

	for face in sim.cloths[0].mesh.faces:
		faces_tensor = torch.cat((faces_tensor, torch.tensor(
			[face.vertices[0].index, face.vertices[1].index, face.vertices[2].index])))


	verts_tensor = verts_tensor.view(-1,3)
	faces_tensor = faces_tensor.view(-1,3)

	print(verts_tensor)
	print(faces_tensor)

	pytorch_mesh = Meshes(verts=[verts_tensor.view(-1, 3)], faces=[faces_tensor.view(-1, 3)])

	print(pytorch_mesh[0].isempty())
	plot_pointcloud(pytorch_mesh)

	return ans / cnt

	
	


lr = 10

with open('log.txt','w',buffering=1) as f:
	tot_step = 1
	for cur_step in range(tot_step):
		g = reset_sim()
		st = time.time()
		run_sim()
		en0 = time.time()
		loss = get_loss()
		print('step={}'.format(cur_step))
		print('forward time={}'.format(en0-st))
		loss.backward()
		en1 = time.time()
		print('backward time={}'.format(en1-en0))
		f.write('step {}: g={} loss={} grad={}\n'.format(cur_step, g.data, loss.data, g.grad.data))
		print('step {}: g={} loss={} grad={}\n'.format(cur_step, g.data, loss.data, g.grad.data))
		g.data -= lr * g.grad
		config['gravity'] = g.detach().numpy().tolist()
		save_config(config, 'curconf.json')


