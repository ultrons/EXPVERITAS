#!/usr/bin/python
import re
import copy
from math import log
import pprint
import scipy.io
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#------------------------------------
# Constants / Parameters
#------------------------------------


from optparse import OptionParser
parser = OptionParser()
parser.add_option("-m", "--matrix", dest="matrix_file", help="Input file containing LDPC G and H Matrices and possibly the image data")
parser.add_option("-e", "--epsilon", dest="epsilon", help="Channel error probability")
parser.add_option("-s", "--max_seeds", dest="max_seeds", help="Number of randome= Seeds to Use")

(options, args) = parser.parse_args()

#------------------------------------
# Constants / Parameters
#------------------------------------

# testVector denotes the data to be transmitted
max_iterations=50
num_scenarios=options.max_seeds
ldpc_matfile=options.matrix_file
epsilon=float(options.epsilon)

bitprob_file = 'BitProbabilities_'+ str(num_scenarios) + str(epsilon) + '.pdf'
hammingDistancePlot = 'HammingDistance_'+ str(num_scenarios) +  str(epsilon) + '.pdf'

class factorGraph (object):
    def __init__(self, matrix_file):
        global bitprob_file
        # Load the matrices
        matrix_data=scipy.io.loadmat(matrix_file)
        self.H_matrix = matrix_data['H']
        self.G_matrix = matrix_data['G']
        self.image_data=0
        if 'logo' in matrix_data:
            self.logo=matrix_data['logo']
            [self.x, self.y] = self.logo.shape
            self.image_data=1
            self.testVector=np.mat(np.reshape(self.logo, (self.x*self.y, 1)))
            

        self.init_message = 0.5
        # Assuming it is 1/2 code
        self.dSize=self.H_matrix.shape[0]

        self.pp=PdfPages(bitprob_file)

        # 0: Storage Containers
        # Constraint Factor dictionary is (constraint node) = > list of variable
        # nodes that it is connected to 
        self.constrFactor={}
        # Variable Node dictionary is (variable node) = > list of constraint
        # nodes that it is connected to 
        self.variableNode={}

        # phi denotes the initial poetential of variable cliques
        self.phi={}
        # xsi denotes the initial poetential of constraint nodes
        self.xsi={}

        self.mu={}
        self.V={}
        self.xorMaskRecord={}
        
        # 1: Define Factor Dictionary
        # Refering only to the constraint factors here
        # keys   :=> Contraint IDs (string)
        # Values :=> Array of indices (integer) that this constraint is holding
        constr_id=0
        for i in range(0,self.H_matrix.shape[0]):
            self.constrFactor[i]=[]
            for j in range(0, (self.H_matrix[i]).shape[0]):
                if self.H_matrix[i][j] == 1:
                    self.constrFactor[i].append(j)
                    #if (j not in self.variableNode):
                    #    self.variableNode[j]=[]
                    #self.variableNode[j].append(i) 
                    # Initialize V values with constant values
                    if j not in self.V: 
                        self.V[j]={}
                    if i not in self.V[j]:    
                        self.V[j][i]={}
                    self.V[j][i][0]= self.init_message
                    self.V[j][i][1]= self.init_message
        self.computeMu()
        #print "Initial Message Configuration from Constraint Nodes to variables"
        #pprint.pprint(self.mu)
        self.init_snapshot=copy.deepcopy(self.V)
        print "Factor Graph read from H Matrix in ", matrix_file, ":"
        print ""
        pprint.pprint(self.constrFactor)
        print ""





    def reset_messages(self):
        self.V=copy.deepcopy(self.init_snapshot)
        self.mu={}
        self.computeMu()


        # Compute mu values
    def computeMu(self):
        for constr in self.constrFactor:
            #dynamic programming scope
            w=len(self.constrFactor[constr])
            if (w not in self.xorMaskRecord):
                self.xorMaskRecord[w] = self.computeXORMask(w)
            xorMask=self.xorMaskRecord[w]
            #print xorMask
            if constr not in self.mu:
                self.mu[constr]={}
            for j in range(0, w):
                vNode=self.constrFactor[constr][j]
                if(vNode not in self.mu[constr]):
                    self.mu[constr][vNode]={}
                    self.mu[constr][vNode][0]=0
                    self.mu[constr][vNode][1]=0
                for mask in xorMask:
                    message=1
                    for vindex in range(0, len(mask)):
                        ovNode=self.constrFactor[constr][vindex]
                        symbol=mask[vindex]
                        if (ovNode != vNode): 
                            message*=self.V[ovNode][constr][symbol]
                        else:
                            vNodeSymbol=symbol
                    self.mu[constr][vNode][vNodeSymbol]+=message
                # Normalization to prevent over/underflow 
                s=sum(self.mu[constr][vNode].values())
                self.mu[constr][vNode][0]=self.mu[constr][vNode][0]/s
                self.mu[constr][vNode][1]=self.mu[constr][vNode][1]/s
        #pprint.pprint(self.mu)

    def parallelMessageUpdate (self):
        self.prevMu=copy.deepcopy(self.mu)
        self.prevV=copy.deepcopy(self.V)
        self.mu={}
        self.V={}
        self.updateV()
        self.computeMu()
        #for i in self.constrFactor:
        #    for j in self.constrFactor[i]:
        #        print "mu:", self.mu[i][j], "V:", self.V[j][i]

    def updateV(self):
        for vNode in self.prevV:
            if( vNode not in self.V):
                self.V[vNode]={}
            for constr in self.prevV[vNode]:
                if constr not in self.V[vNode]:
                    self.V[vNode][constr]={}
                self.V[vNode][constr][1]=self.phi[vNode][1]
                self.V[vNode][constr][0]=self.phi[vNode][0]
                oConstr=copy.deepcopy(self.prevV[vNode].keys())
                oConstr.remove(constr)
                for oc in oConstr:
                    self.V[vNode][constr][1]*=self.prevMu[oc][vNode][1]
                    self.V[vNode][constr][0]*=self.prevMu[oc][vNode][0]
                # Normalization to prevent over/underflow 
                s=sum(self.V[vNode][constr].values())
                self.V[vNode][constr][1]= self.V[vNode][constr][1] / s
                self.V[vNode][constr][0]= self.V[vNode][constr][0] / s

    def mapDecode(self):
        data=[]
        pr_1=[ ]
        for vNode in range(0, len(self.V.keys())):
            pr={}
            pr[0]=1
            pr[1]=1
            pr[0]=self.phi[vNode][0]
            pr[1]=self.phi[vNode][1]
            for constr in self.V[vNode]:
                pr[0]*=self.mu[constr][vNode][0]
                pr[1]*=self.mu[constr][vNode][1]
            # Normalize for better viewing
            s=sum(pr)
            pr[0]= pr[0]/s
            pr[1]= pr[1]/s
            pr_1.append(pr[1])
            if (pr[0] > pr [1] ):
                data.append(0)
            else:
                data.append(1)
        return [np.mat(np.array(data)), pr_1]
        #print data

    # Routine to compute transmit codeword
    # Inputs the binary string
    # Outputs the binary string followed by the parity bits
    # (systematic codes)
    # Plain multiplication with G
    # written as routine to allow calling it multiple times
    def computeCodeWord(self, inputData):
        # Compute the code word
        #print ":)"
        return self.G_matrix*self.testVector

    def computeRecievedCodeWord (self, transmittedCodeWord, epsilon, time_seed):
        random.seed(time_seed)
        #tr=copy.deepcopy(transmittedCodeWord)
        errorWord=np.mat(np.zeros(transmittedCodeWord.shape), dtype=int)
        upperLimit=int(1/epsilon)
        for i in range(0, transmittedCodeWord.shape[0]):
            errorWord[i] = np.mat(random.randint(0,upperLimit)==0)
        return np.bitwise_xor(errorWord, transmittedCodeWord)

    def computeXORMask(self, length):
        xorMask=[]
        for i in range(0,2**length):
            #if(np.bitwise_xor.reduce(np.array(list(np.binary_repr(i,width=length )), dtype=int))== 0):
            if(sum(np.array(list(np.binary_repr(i,width=length )),
                            dtype=int).tolist()) %2 == 0):
                xorMask.append(np.array(list(np.binary_repr(i, width=length)), dtype=int))
        return xorMask


    def computeUnaryPotentials(self,r, epsilon):
        for i in range(0, r.shape[0]): 
            self.phi[i]={}
            if(np.int(r[i]) == 0): 
                self.phi[i][0] = 1 - epsilon
                self.phi[i][1] = epsilon
            else:
                self.phi[i][0] = epsilon
                self.phi[i][1] = 1 - epsilon


    def checkLBP(self,max_iterations, expectedOutput, garbledData,timeseed):
        print "Checking for test Data:"
        pprint.pprint(expectedOutput.T)
        print ".."
        print "Data after noise was:"
        pprint.pprint(garbledData.T)
        print ".."
        one_prob_all_itr=[]
        image_capture_itr=[]
        hamming_distance=[]
        itr_to_print=[ 0, 1, 2, 3, 5, 10, 20, 30]
        # Loopy belief propagation
        for i in range(0, max_iterations): 
            self.parallelMessageUpdate() 
            [predictedData, prob_one] = self.mapDecode()
            one_prob_all_itr.append(prob_one)
            hamming_distance.append(np.sum(np.bitwise_xor(expectedOutput,
                                                          predictedData.T)))

            if i in itr_to_print:
                image_capture_itr.append(predictedData[:, 0:self.x*self.y])
                print predictedData[:, 0:self.x*self.y].shape

            # Convergence Condition
            #if (hamming_distance == 0):
            #    print "################################"
            #    print "Correct decoding achieved in ", i+1, " Iterations!!"
            #    break
        print "Recovered data after decoding :"
        pprint.pprint(predictedData)
        
        # If the decoding can not predict the data 
        if (hamming_distance[-1] != 0):
            print "Failed to give correct prediction in", max_iterations, " Iterations!!"
            print "Final Hamming Distance ", hamming_distance[-1]
        #print ".."
        #print "Testing completed!!!"
        if self.image_data == 1:
            self.plot_image(image_capture_itr)
        else: 
            self.plot_probabilities(one_prob_all_itr, timeseed)
        return hamming_distance




    def plot_probabilities(self, prob_array, time_seed):
        fig=plt.figure()
        fig.suptitle("Observed Probabilities of Each CodeWord Bits, , Time Seed="+ str(time_seed), fontsize=15)
        p_array=np.array(prob_array)
        num_bits=p_array.shape[1]
        num_itr=p_array.shape[0]
        for i in range(0, num_bits):
            plt.plot(np.arange(1, num_itr+1, 1.0), p_array[:,i:i+1], label="bit"+str(i))
        plt.axis([1,num_itr, 0, np.max(p_array)*1.2])
        #plt.xticks(np.arange(1, num_itr+1, 1.0))
        plt.ylabel("P(x=1)")
        plt.xlabel("Loopy Belief Iterations")
        plt.savefig(self.pp, format= 'pdf')

    def plot_image(self, image_capture_itr):
        plt.imshow(self.logo)
        for i in image_capture_itr:
            image=np.reshape(i, (self.x, self.y))
            plt.imshow(image)
            plt.show()
        
        
###################################################

# A. Read in the Matrix File and create FactorGraph
fg=factorGraph(ldpc_matfile)
# B. i/ Define an all zero test vector
if fg.image_data == 0:
    testVector=np.mat(np.zeros(fg.dSize), dtype=int).reshape(fg.dSize,1)
else:
    testVector=fg.testVector

# B. ii/ Compute the codeword corresponding the test vector
t=fg.computeCodeWord(testVector)
max_hd=0
hd_array=[]
seed_list=np.arange(1, int(num_scenarios)+1, 1)
for i in seed_list:
    time_seed=i
    # B. iii/ Compute the Recieved codeword after error injection
    r=fg.computeRecievedCodeWord(t, epsilon, time_seed)
    # B. iv/ Compute Unary potentials
    fg.computeUnaryPotentials(r, epsilon)
    # B. v/ Loopy Belief propagations
    hd_array.append(fg.checkLBP(max_iterations, t, r, time_seed))
    fg.reset_messages()
fg.pp.close()

pp=PdfPages(hammingDistancePlot)
fig_h=plt.figure()
fig_h.suptitle("Hamming Distance Convergence for Different Random Seeds ", fontsize=15)
for i in range(len(hd_array)):
    hd=hd_array[i]
    plt.plot(np.arange(1, max_iterations+1, 1.0), hd,
             label="TimeSeed"+str(seed_list[i]))
    if(max(hd) > max_hd):
        max_hd=max(hd)

plt.axis([1,max_iterations, 0, max_hd])
plt.ylabel("Hamming Distance")
plt.xlabel("Loopy Belief Iterations")
# Shrink current axis by 20%
ax=plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

#Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          fontsize='x-small', ncol=1)
        



plt.savefig(pp, format= 'pdf')
pp.close()


plt.show()



    
