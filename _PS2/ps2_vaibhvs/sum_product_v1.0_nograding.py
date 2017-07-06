#!/usr/bin/python
import re
import copy
from math import log
# Following implementation is an attempt to code Algorithm 9.1 from
# "Probabilistic Graphical Models and Techniques - Koller and Friedman

# Method used here is creation of sum product expression based on 9.1 
# and then substituting the observations in the final expression
# Followed by simplifying based on CPDs provided in the problem
# Dynamic programing methods are used to not make computation of substituted
# expressions only once, and then using the modified look up table to speed up
# the computation

class bayesianNet (object):
    def __init__(self,BN, observationRecord):
        # Constants 
        self.var_val=['A', 'C', 'G', 'T']
        self.leaf_nodes={
                           'human': 'r1',
                           'baboon': 'r2',
                           'marmoset': 'r4',
                           'rat': 'r6',
                           'mouse': 'r7',
                           'rabbit': 'r9',
                           'cow': 'r12',
                           'dog': 'r13',
                           'elephant': 'r16',
                           'platypus': 'r18'
        }

        self.cpd={
            # Parent child conditional probability table
            'A A':0.831,
            'A C':0.032,
            'A G':0.085,
            'A T':0.052,
            'C A':0.046,
            'C C':0.816,
            'C G':0.028,
            'C T':0.110,
            'G A':0.122,
            'G C':0.028,
            'G G':0.808,
            'G T':0.042,
            'T A':0.053,
            'T C':0.076,
            'T G':0.029,
            'T T':0.842,
            # Priors for root Node
            'X A':0.295,
            'X T':0.295,
            'X C':0.205,
            'X G':0.205
        }
                   
        self.PHI = {}
        self.ancestors = {}
        count = 1
        # BN is expected to be represented as ORDERED pair
        # N1, N2 means there is an edge from N1 -> N2
        file_pointer = open(BN, 'r')
        for aline in file_pointer:
            [tail, head] = aline.rstrip('\n').split(",")
            self.PHI[tail + ' ' + head] = {}
            self.PHI[tail + ' ' + head][count] = []
            self.PHI[tail + ' ' + head][count].append(tail + ' ' + head)
            if tail not in self.leaf_nodes.values() and tail != 'X' :
                self.ancestors[tail]=1
        file_pointer.close()

        # Save the Orignial Tree for restoration/loops
        self.PHI_buffer = copy.deepcopy(self.PHI)

        self.observationTable={}
        file_pointer=open(observationRecord, 'r')
        for aline in file_pointer:
            if re.match(">", aline):
                species=aline.split()[1]
                count=0
            else:
                char_list=list(aline.rstrip('\n'))
                for i in range(0,len(char_list)):
                    if count not in self.observationTable.keys():
                        self.observationTable[count]={}
                    self.observationTable[count][species]=char_list[i]
                    count+=1
        file_pointer.close()

                
    def sumProductEleminate(self,nodeList):
       for anode in nodeList:
           self.sumProductEleminateVar(anode)

    def sumProductEleminateVar(self,var):
        phi_prime={}
        for akey in self.PHI.keys():
            if re.search(var, akey):
                phi_prime[akey] = {}
                phi_prime[akey]= copy.deepcopy(self.PHI[akey])
                del self.PHI[akey]

        #print phi_prime
        self.marginalize(phi_prime,var)

    def marginalize(self,phi_prime,var):
        tau_key=re.sub(var, '', ' '.join(phi_prime.keys()))
        tau_key=re.sub("^ *", '', tau_key)
        tau_key=re.sub(" *$", '', tau_key)


        self.PHI[tau_key]= {}
        # Computation of XSI
        # This is where we multiply the factors present in phi_prime 
        self.XSI={}
        for afactor in phi_prime.keys():
            self.multiplyFactors(phi_prime[afactor])

        count=1
        for val in self.var_val:
            for akey in self.XSI.keys():
                self.PHI[tau_key][count]=[]
                self.PHI[tau_key][count] = [a.replace(var, val) for a in
                                            self.XSI[akey]]
                #for aexpr in self.XSI[akey]:
                #    self.PHI[tau_key][count].append(re.sub(var, val, aexpr))
                count+=1
                
        #print self.PHI[tau_key]


    def estimate_probability(self, observationSet):
        total_probability=0
        self.cpd_aux={}

        for akey in self.PHI.keys():
            for aexpr in self.PHI[akey]:
                pr_term=1.0
                for i in self.PHI[akey][aexpr]:
                    try: 
                        pr_term=pr_term*self.cpd[i]
                    except:
                        try:
                            pr_term=pr_term*self.cpd_aux[i]
                        except:    
                            p_c   = i
                            [p, c]= i.split(' ')
                            try:
                                p_c=i.replace(c, observationSet[c])
                            except:
                                pass
                            try:
                                p_c=i.replace(p, observationSet[p])
                            except:
                                pass
                            self.cpd_aux[i]=self.cpd[p_c]
                            pr_term=pr_term*self.cpd[p_c]    
                total_probability+=pr_term
        
        print total_probability
        return total_probability

    def multiplyFactors(self,factor):
        if (len(self.XSI.keys()) == 0):
            self.XSI = copy.deepcopy(factor)
        else:
            buffer = copy.deepcopy(self.XSI)
            self.XSI={}
            count=1
            for aexpr in factor.keys():
                for oexpr in buffer.keys():
                    self.XSI[count]=buffer[oexpr] + factor[aexpr]
                    count+=1

    def updateObservations(self,observation):
        print observation
        for afactor in self.PHI.keys():
            for aexpr in self.PHI[afactor][1]:
                _key=aexpr
                for akey in observation.keys():
                    _key = re.sub(r'\b%s\b' % akey, observation[akey], _key)
                self.PHI[afactor][1]=[_key]
    def restoreBN(self):
        self.PHI=copy.deepcopy(self.PHI_buffer)
    def saveBN(self):
        self.PHI_buffer=copy.deepcopy(self.PHI)
            
#my_bn=bayesianNet('./bn.csv', './multicolumn.fa')
my_bn=bayesianNet('./bn.csv', './column.fa')
my_bn.sumProductEleminate(my_bn.ancestors.keys())
#my_bn.sumProductEleminate(['r19', 'r8', 'r10', 'r5', 'r14', 'r11', 'r17', 'r15', 'r3'])
prob_results=[]
for aset in my_bn.observationTable.keys():
    observationSet={}
    for aspecies in my_bn.observationTable[aset].keys():
        observationSet[my_bn.leaf_nodes[aspecies]]=my_bn.observationTable[aset][aspecies]
    print observationSet 
    #my_bn.updateObservations(observationSet)
    prob_results.append(my_bn.estimate_probability(observationSet))
    #my_bn.restoreBN()
log_likelihood=0
for i in prob_results:
    log_likelihood+=log(i)

print log_likelihood

