#!/usr/bin/python

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

#np.random.seed(17)
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# Objective of this exercise to visualize the linear maps
# By definition f: V -> W is a linear map if it satisfies
# f(x+y) = f(x) + f(y) {Additivity}
# f(alph* x) = alpha*f(x)  {Homogeneity of degree 1}
# I would like to see what it means visually

# Also if V and W have finite basis vectors
# f = Ax is always a linear map
# The condition of finite basis vectors is not much apparent :(
#A= np.random.random(9).reshape((3,3))
A=np.ones((3,3))


fig=plt.figure()
ax=fig.add_subplot(3,2,1, projection='3d')
ax.set_title('Matrix A')
ax.scatter(A[:,0],A[:,1],A[:,2], c='Blue', s=25)


x=np.random.random(300).reshape((3,100))
U, s, V = np.linalg.svd(A, full_matrices=True)

print "#### Singular Value Decomposition ####"
print U
print s
print V

eigVal, eigVec = np.linalg.eig(A)
print "#### Eigen Vectors ####"
print eigVec
print eigVal

for v in eigVec.T:
    a = Arrow3D([0, v[0]],
                [0, v[1]],
                [0, v[2]],
                mutation_scale=20, lw=3, arrowstyle="-|>", color="r")

#ax.add_artist(a)

s=np.diag(np.tile(s.reshape(A.shape[0],1),(1, A.shape[1])))
s=np.diag(s)
#Create x,y
u=np.arange(-30,30, 0.5)
v=np.arange(-30,30, 0.5)

#f=np.dot(A,x)

# Visualizing the points in space
# Before transformation 
#plt3d.scatter(x[0,:], x[1,:], x[2,:], c='red')
# After transformation 
#plt3d.scatter(f[0,:], f[1,:], f[2,:], c='green')

#plt.show()

#Not very helpful huh :(

#Plot surfaces and see how they transform

#Create a random normal vector
normal = np.random.random(3)
#Intercept
intercept=random.uniform(0, 4)

#Create xy mesh
xx,yy = np.meshgrid(u,v, sparse=True)

#evaluate z
z = (intercept - np.multiply(xx,normal[0]) - np.multiply(yy,normal[1])) / normal[2]

#plt3d=plt.figure().gca(projection='3d')
#fig=plt.figure(figsize=plt.figaspect(3.2))
plt3d=fig.add_subplot(3,2,2, projection='3d')
plt3d.set_title('Original Plane')
plt3d.plot_surface(xx,yy,z, color='blue')

# Apply the transformation
txx= np.tile(xx, yy.shape)
tyy= np.tile(yy, xx.shape)
p=np.vstack((txx,tyy,z)).reshape((3,z.shape[0],z.shape[1]))



f=np.tensordot(A,p, axes=([-1],[0]))
plt3d=fig.add_subplot(3,2,3, projection='3d')
plt3d.set_title('Plane Transformed by Matrix A')
plt3d.plot_surface(f[0,0,:].reshape(xx.shape),f[1,:,0].reshape(yy.shape),f[2,:,:].reshape(z.shape), color='green')


f=np.tensordot(V,p, axes=([-1],[0]))
plt3d=fig.add_subplot(3,2,4, projection='3d')
plt3d.set_title('Plane Transformed by Matrix V')
plt3d.plot_surface(f[0,0,:].reshape(xx.shape),f[1,:,0].reshape(yy.shape),f[2,:,:].reshape(z.shape), color='red')

f=np.tensordot(s,f, axes=([-1],[0]))
plt3d=fig.add_subplot(3,2,5, projection='3d')
plt3d.set_title('Plane Transformed by Matrix sV')
plt3d.plot_surface(f[0,0,:].reshape(xx.shape),f[1,:,0].reshape(yy.shape),f[2,:,:].reshape(z.shape), color='yellow')

f=np.tensordot(U,f, axes=([-1],[0]))+10
plt3d=fig.add_subplot(3,2,6, projection='3d')
plt3d.set_title('Plane Transformed by Matrix UsV, moved by 10 units')
plt3d.plot_surface(f[0,0,:].reshape(xx.shape),f[1,:,0].reshape(yy.shape),f[2,:,:].reshape(z.shape), color='orange')


#fig.tight_layout()
plt.show()



