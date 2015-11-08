#!/usr/bin/python

##########################################################
# Abstract: 
# 1. Visualize the Gaussian distribution and Variance aspects in 3
# dimensions
# 2. Contour plots for single and mixture of Gaussians
# 3. Perform a gradient descent on single gaussian dataset and plot the
# trajectory of gradient descent
##########################################################
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as sp
#from mpl_toolkits.mplot3d import proj3D


class gaussianExperiments(object):
    def __init__(self, dataSet=None):
        m=300
        d=2
        if dataSet is None:
            self.data=np.random.random(m*d).reshape((d,m))
    # Compute the sufficient statistic for the gaussian
    # From first principle
    # i.e. compute mean
    # zero center the data on the mean
    # compute variance
    def fitGaussianPro(self):
        self.mean=np.sum(self.data, axis=1, keepdims=True)/self.data.shape[1]
        meanCenteredData = self.data - self.mean
        self.cov=np.dot(meanCenteredData, meanCenteredData.T) / self.data.shape[1]

    def checkGaussianParam(self):
        assert self.mean.all() == np.mean(self.data, axis=1, keepdims=True).all()
        assert self.cov.all() == np.cov(self.data).all()

    def plotData(self):
        # Scatter plot of the data
        fig=plt.figure()
        ax=fig.add_subplot(5,3,1)
        ax.set_title('Scatter Plot of Random Data')
        ax.scatter(self.data[0,:], self.data[1,:])

        #Plotting Gaussian Curve
        ax=fig.add_subplot(5,3,4, projection='3d')
        x, y = np.mgrid[-4:4:0.1, -4:4:0.1]
        pos = np.empty(x.shape + (2,))
        pos[:,:,0] = x
        pos[:,:,1] = y
        print self.cov
        self.cov=np.array([[ 0.5,0],[0,0.5]])
        ax.set_title('Gaussian: COV: [[ 0.5,0],[0,0.5]]')
        self.mean=[ 0.0,0.0]
        g=sp.multivariate_normal(self.mean, self.cov)
        z=g.pdf(pos)
        ax.plot_surface(x,y,z)
        ax=fig.add_subplot(5,3,5)
        ax.set_title('Gaussian Contour')
        ax.contour(x,y,z)
        ax=fig.add_subplot(5,3,6)
        print self.data.shape, self.cov.shape
        tr=np.dot(self.cov,self.data)
        ax.set_title('Random data transformed with Cov')
        ax.scatter(tr[0,:], tr[1,:])




        ax=fig.add_subplot(5,3,7, projection='3d')
        self.cov=[[ 0.5,0],[0,0.01]]
        ax.set_title('Gaussian: COV: [[ 0.5,0],[0,0.01]]')
        g=sp.multivariate_normal(self.mean, self.cov)
        z=g.pdf(pos)
        ax.plot_surface(x,y,z)
        ax.plot_surface(x,y,z)
        ax=fig.add_subplot(5,3,8)
        ax.set_title('Gaussian Contour')
        ax.contour(x,y,z)
        ax=fig.add_subplot(5,3,9)
        tr=np.dot(self.cov,self.data)
        ax.set_title('Random data transformed with Cov')
        ax.scatter(tr[0,:], tr[1,:])

        ax=fig.add_subplot(5,3,10, projection='3d')
        self.cov=[[ 0.5,0.1],[0.1,0.5]]
        ax.set_title('Gaussian: COV: [[ 0.5,0.1],[0.1,0.5]]')
        g=sp.multivariate_normal(self.mean, self.cov)
        z=g.pdf(pos)
        ax.plot_surface(x,y,z)
        ax.plot_surface(x,y,z)
        ax=fig.add_subplot(5,3,11)
        ax.set_title('Gaussian Contour')
        ax.contour(x,y,z)
        ax=fig.add_subplot(5,3,12)
        tr=np.dot(self.cov,self.data)
        ax.set_title('Random data transformed with Cov')
        ax.scatter(tr[0,:], tr[1,:])

        ax=fig.add_subplot(5,3,13, projection='3d')
        self.cov=[[ 0.5,-0.1],[-0.1,0.5]]
        ax.set_title('Gaussian: COV: [[ 0.5,-0.1],[-0.1,0.5]]')
        g=sp.multivariate_normal(self.mean, self.cov)
        z=g.pdf(pos)
        ax.plot_surface(x,y,z)
        ax.plot_surface(x,y,z)
        ax=fig.add_subplot(5,3,14)
        ax.set_title('Gaussian Contour')
        ax.contour(x,y,z)
        tr=np.dot(self.cov,self.data)
        ax=fig.add_subplot(5,3,15)
        ax.set_title('Random data transformed with Cov')
        ax.scatter(tr[0,:], tr[1,:])
        eigVal, eigVec = np.linalg.eig(self.cov)
        print eigVec

        fig.tight_layout()
        plt.show()
        






    



mg=gaussianExperiments()
mg.fitGaussianPro()
mg.checkGaussianParam()
mg.plotData()
