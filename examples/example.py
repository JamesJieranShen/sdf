#%%
from sdf import *

# f = sphere(1) & box(1.5)
f = sphere(1)
# c = cylinder(0.5)
# f -= c.orient(X) | c.orient(Y) | c.orient(Z)

# f.save('out.stl')
verts, faces = generate(f)
#%%
print(verts[:, 0])

#%%
# Plot
%matplotlib widget

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_mesh(verts, faces, fig=None):
    if fig is None:
        fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles = faces, edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.0, shade=False)
    # plt.show()

def plot_vertices(verts):
    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    ax = plt.axes(projection='3d')
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs))) 
    ax.plot3D(xs, ys, zs)

plot_vertices(verts)
plt.show()




# %% test face vector direction
import matplotlib.pyplot as plt
import numpy as np

A, B, C = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
normals = np.cross((B-A), (C-A))
normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

center = (A + B + C) / 3
center /= np.linalg.norm(center, axis=-1, keepdims=True)
# for a sphere, the center of the trig points in the same direction as the
# normal
product = np.sum(center * normals, axis=-1)
plt.figure()
plt.hist(product, bins=100, range=(-2, 2))
plt.show()
# %%
verts.shape