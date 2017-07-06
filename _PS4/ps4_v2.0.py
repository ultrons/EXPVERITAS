#!/usr/bin/python

import re
import numpy as np
import matplotlib.pyplot as plt
import pylab
import matplotlib.cm as cm
import copy
import random
import pprint
from matplotlib.backends.backend_pdf import PdfPages



from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n", "--noisyimage", dest="noisy_image", help="inputImage to Restore")
parser.add_option("-o", "--originalimage", dest="orig_image", help="original_image")
parser.add_option("-g", "--graphOUT", dest="graphOUT", help="OutputGraph File")

(options, args) = parser.parse_args()



class gibbs_sampler(object):

    def __init__(self, noisy_image_file, original_image_file):
        # Constants
        self.eta=1
        self.beta=1
        self.seed=7
        self.B=10
        self.S=10
        random.seed(self.seed)

        # Reading in the noisy_image_file
        data = np.loadtxt(noisy_image_file, comments='#')
        [nrows, ncolumn]=[int(data[-1][0]+1), int(data[-1][1]+1)]
        self.Y=data[:,2].reshape(nrows,ncolumn)
        # Reading in the original_image_file 
        data = np.loadtxt(original_image_file, comments='#')
        [nrows, ncolumn]=[int(data[-1][0]+1), int(data[-1][1]+1)]
        self.origX=data[:,2].reshape(nrows,ncolumn)



        # Intializing X image
    def initX(self,value):
        self.X=copy.deepcopy(value)
        self.p=np.zeros(self.X.shape)
        self.empericalPosterior=np.zeros(self.X.shape, dtype=float)
        self.ref=np.ones(self.X.shape)
        [self.x, self.y] = self.X.shape
        self.energy = []

        #plt.imshow(self.Y, cmap=cm.Greys_r)
        #plt.show()
    def computeMBindices(self):
        self.MBID={}
        for i in range(self.x):
            if i not in self.MBID:
                self.MBID[i]={}
            for j in range(self.y):
                if j not in self.MBID[i]:
                    self.MBID[i][j]=[]
                if i-1 >= 0 :
                    self.MBID[i][j].append([i-1,j])
                if j-1 >= 0 :
                    self.MBID[i][j].append([i,j-1])
                if i < self.x-1 :
                    self.MBID[i][j].append([i+1,j])
                if j < self.y-1 :
                    self.MBID[i][j].append([i,j+1])


    #def MB(self, i, j):
    #    mbValue=sum([self.X[index[0]][index[1]] for index in self.MBID[i][j]])
    #    return mbValue
    def MB(self, i, j):
        neighbours=0
        [x, y] = self.X.shape
        # Boundary cases
        if i-1 >= 0 :
            neighbours+=self.X[i-1][j]
        if j-1 >= 0 :
            neighbours+=self.X[i][j-1]
        if i < x-1 :
            neighbours+=self.X[i+1][j]
        if j < y-1 :
            neighbours+=self.X[i][j+1]

        return neighbours


    def samplenUpdatePixel(self, i, j):
        # Compute posterior
        z = self.eta * self.Y[i][j]  + self.beta*self.MB(i,j)
        self.p[i][j] = (1 /(1+np.exp(-2*z)))
        # Sample from posterior
        x_ij = self.sampleValue(i, j, self.p[i][j])
        # update pixel with the sampled value
        self.X[i][j]= x_ij

    def burnInItr(self):
        [x,y] = self.X.shape
        for j in range(0,y):
            for k in range(0,x):
                self.samplenUpdatePixel(k,j)
        self.energy.append(self.estimateEnergyMatrix())
        print self.energy[-1]


    def sampleValue(self,i, j, p):
        if (random.uniform(0,1) <= p):
            return 1.0
        else:
            return -1.0


    def burnIn(self):
        [x,y] = self.X.shape
        for i in range(0,self.B):
            print "Burnin Iterarion", i, "Energy:",
            self.burnInItr()

    def samplePartial(self, x1,x2, y1,y2):
        numOnes=[]
        self.empericalPosterior=np.zeros(self.X.shape, dtype=float)[x1:x2, y1:y2]
        # empericalPosterior will keep track of number of ones
        # Observed in each of the sampling step for each of the pixel
        for i in range(0, self.S):
            print "Sampling Iterarion:", i, "Energy:",
            #self.samplingItr()
            self.burnInItr()
            grid=np.array((self.X == self.ref), dtype=float)[x1:x2, y1:y2]
            self.empericalPosterior +=grid
            numOnes.append(np.sum(grid))
        return numOnes


    def sample(self):
        self.empericalPosterior=np.zeros(self.X.shape, dtype=float)
        # empericalPosterior will keep track of number of ones
        # Observed in each of the sampling step for each of the pixel
        for i in range(0, self.S):
            print "Sampling Iterarion", i, "Energy:",
            # We call burnIN iteration here difference wrt burnin phase is that
            # We record the number of ones sampled in each step
            #self.samplingItr()
            self.burnInItr()
            self.empericalPosterior += np.array((self.X == self.ref), dtype=float)
            # Normalization of emperical posterior is deferred to the
            # estimateImage Step
            # So that we can extract  one count for the given pixel range 
            # As required in the last part of the problem

    def estimateImage(self):

        self.empericalPosterior *= (1.0/self.S)
        [x, y] = self.X.shape
        threshold= 0.5*self.ref
        self.X=(np.array((self.empericalPosterior > threshold), dtype=int)) -  (np.array((self.empericalPosterior <= threshold), dtype=int))
        # Printing out the figure of merit (fraction of pixel different in the
        # orginal and estimated image
        print "Different Pixel Fraction: ",
        print np.sum(np.array((self.X != self.origX), dtype=float)) / x / y

    def samplingItr(self):
        [x,y] = self.X.shape
        random_draw=np.random.uniform(0,1, self.X.shape)
        self.X=(np.array((self.p >= random_draw), dtype=int)) - (np.array((self.p < random_draw), dtype=int))


    def estimateEnergy(self):
        [x, y] = self.X.shape
        beta_component=0
        eta_component=0
        for i in range(0,x):
            for j in range(0,y):
                beta_component-=(self.X[i][j]*sum(self.MB(i,j)))
                eta_component-=(self.X[i][j]*self.Y[i][j])

        #Since we are looping over pixels 
        #Beta component is double counted
        #Hence the factor of 0.5 is introduced in the expression
        return (self.beta*0.5*beta_component) + (self.eta*eta_component)

    def estimateEnergyMatrix(self):
        # Matrix rolled by one unit in up and down direction (zero padded)
        # Then multiplied with the orginal matrix and then summed would count neighbouring
        # product exactly onces
        # Since it is a matrix operation it is extermely fast in comparison to
        # the usual loop implementation (estimateEnery Sub-Routine)
        x_shifted=np.roll(self.X, 1, axis=1)
        x_shifted[:,0]=0

        y_shifted=np.roll(self.X, 1, axis=0)
        y_shifted[0,:]=0

        beta_component=np.sum(np.multiply(x_shifted, self.X) + np.multiply(y_shifted, self.X))
        eta_component=np.sum(np.multiply(self.X, self.Y)) 
        return -(self.beta*beta_component) - (self.eta*eta_component)

    def plotImages(self, fig):
        fig.add_subplot(221)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
        plt.imshow(self.X, cmap=cm.Greys_r)
        plt.title('Restored Image through Gibbs Sampling')
        #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.add_subplot(222)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
        plt.imshow(self.Y, cmap=cm.Greys_r)
        plt.title('Noisy Image')
        #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.add_subplot(223)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
        plt.imshow(self.origX, cmap=cm.Greys_r)
        plt.title('Original Image')
        #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    def runSampler(self, fig):
        print "###############################################"
        print "Starting Burnin  Phase..."
        self.burnIn()
        print "Finished Burnin  Phase..."
        print "###############################################"
        print "Starting Sampling Phase..."
        #self.sample()
        self.samplePartial(0,self.x,0,self.y)
        print "Finished Sampling  Phase..."
        print "###############################################"
        self.estimateImage()
        self.plotImages(fig)
        print "###############################################"
        return self.energy



class gobbs_sampler(object):

    def __init__(self, noisy_image_file, original_image_file):
        self.nItr=30
        # Constants
        # Reading in the noisy_image_file
        data = np.loadtxt(noisy_image_file, comments='#')
        [nrows, ncolumn]=[int(data[-1][0]+1), int(data[-1][1]+1)]
        self.Y=data[:,2].reshape(nrows,ncolumn)
        # Reading in the original_image_file 
        data = np.loadtxt(original_image_file, comments='#')
        [nrows, ncolumn]=[int(data[-1][0]+1), int(data[-1][1]+1)]
        self.origX=data[:,2].reshape(nrows,ncolumn)



        # Intializing X image
    def initX(self,value):
        self.X=copy.deepcopy(value)
        self.p=np.zeros(self.X.shape)
        self.empericalPosterior=np.zeros(self.X.shape, dtype=float)
        self.ref=np.zeros(self.X.shape)

    def stochasticVoteUpdate(self):
        [x, y] = self.X.shape
        for j in range(0,y):
            for k in range(0,x):
                if (sum(self.MB(k,j)) + self.Y[k][j] > 0):
                    self.X[k][j]=1
                else:
                    self.X[k][j]=-1
                    
    def MB(self, i, j):
        neighbours=[]
        [x, y] = self.X.shape
        # Boundary cases
        if i-1 >= 0 :
            neighbours.append(self.X[i-1][j])
        if j-1 >= 0 :
            neighbours.append(self.X[i][j-1])
        if i < x-1 :
            neighbours.append(self.X[i+1][j])
        if j < y-1 :
            neighbours.append(self.X[i][j+1])

        return neighbours




    def votesParallelUpdate(self):
        # Code Resused from Energy Computations Sub-Routine
        # We Sum up the shifted matrices
        # If the resultant is negative
        # This means more neigbours favor -1
        # Else more neighbours favor +1
        # We update self.X accordingly

        [x, y] = self.X.shape
        x_shifted_right=np.roll(self.X, 1, axis=1)
        x_shifted_right[:,0]=0
        x_shifted_down=np.roll(self.X, 1, axis=0)
        x_shifted_down[0,:]=0

        x_shifted_left=np.roll(self.X, 1, axis=1)
        x_shifted_left[:,y-1]=0
        x_shifted_up=np.roll(self.X, 1, axis=0)
        x_shifted_up[x-1,:]=0

        voteMatrix=x_shifted_right+ x_shifted_left + x_shifted_up + x_shifted_down + self.Y
        self.X=(np.array((voteMatrix > 0 ), dtype=int)) - (np.array((voteMatrix < 0 ), dtype=int))




    def plotImages(self, fig):
        fig.add_subplot(224)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
        plt.imshow(self.X, cmap=cm.Greys_r)
        plt.title('Naive Voting Based Estimation')
    
    def estimateImage(self):
        [x, y] = self.X.shape
        for i in range(0,self.nItr):
            print "Voting Iteration: ", i
            self.stochasticVoteUpdate()
        print "Different Pixel Fraction: ",
        print np.sum(np.array((self.X != self.origX), dtype=float)) / x / y

    def runGOB(self,fig):
        print "###############################################"
        print "Starting Voting Based Restoration ..."
        self.estimateImage()
        print "Finished Voting Based Restoration ..."
        self.plotImages(fig)
        print "###############################################"











noisy_image=options.noisy_image
orig_image=options.orig_image
pp=PdfPages(options.graphOUT)

gb=gibbs_sampler(noisy_image, orig_image)
gob=gobbs_sampler(noisy_image, orig_image)
#gb.computeMBindices()
#print gb.MB(235,359)
energyData=[]
oneCount=[]

fig = pylab.figure()
gb.initX(gb.Y)
energyData.append(gb.runSampler(fig))
gb.energy=[]
# Done Only for single initialization
print "Running Partial Sampling in the Window: (125,143), (162,174)"
oneCount.append(gb.samplePartial(125,162,143,174))
print "Data Collection for Histogram completed."
gob.initX(gb.Y)
gob.runGOB(fig)
#plt.show()
plt.savefig(pp, format='pdf')


fig = pylab.figure()
gb.initX(gb.Y*-1)
energyData.append(gb.runSampler(fig))
gob.initX(gb.Y*-1)
gob.runGOB(fig)
#plt.show()
plt.savefig(pp, format='pdf')



fig = pylab.figure()
x=np.random.uniform(0,1, gb.Y.shape)
ref=0.5*(np.ones(gb.Y.shape))
gb.initX((np.array((x >= ref), dtype=int)) - (np.array((x < ref), dtype=int)))
energyData.append(gb.runSampler(fig))

gob.initX(gb.Y*-1)
gob.runGOB(fig)
#plt.show()
plt.savefig(pp, format='pdf')




fig = pylab.figure()
plt.plot(energyData[0], label="init X=  Y")
plt.plot(energyData[1], label="init X= -Y")
plt.plot(energyData[2], label="init X=  random +/-1")
plt.xlabel("Gibbs Sampling Iterations")
plt.ylabel("Energy")
plt.title("Energy Across Iterations")
plt.legend()
#plt.show()
plt.savefig(pp, format='pdf')


fig = pylab.figure()
plt.hist(oneCount[0], label=noisy_image)
plt.xlabel("Number of Ones Counted")
plt.ylabel("Frequency")
plt.title("One Count Histogram")
plt.legend()
#plt.show()
plt.savefig(pp, format='pdf')

pp.close()






