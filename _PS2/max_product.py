#!/usr/bin/python
import re
import copy
from math import log
import pprint

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
        self.maxp={
            # Parent child conditional probability table
            'A' : [0.831, 'A'],
            'C' : [0.816, 'C'],
            'G' : [0.808, 'G'],
            'T' : [0.842, 'T'],
            # Priors for root Node
            'X r19':[0.295, 'A']
        }

                   
        self.PHI = {}
        self.MAP={}
        self.XSI={}
        self.ancestors = {}
        count = 1
        # BN is expected to be represented as ORDERED pair
        # N1, N2 means there is an edge from N1 -> N2
        file_pointer = open(BN, 'r')
        for aline in file_pointer:
            [tail, head] = aline.rstrip('\n').split(",")
            self.PHI[tail + ' ' + head] = []
            self.PHI[tail + ' ' + head].append(tail + ' ' + head)
            if tail not in self.leaf_nodes.values() and tail != 'X' :
                self.ancestors[tail]=1
        file_pointer.close()

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

                
    def maxProduct(self,nodeList):
       self.nodeList=nodeList
       for anode in nodeList:
           self.maxProductVar(anode)
       #pprint.pprint(my_bn.XSI)
       self.traceBackMAP()

    def maxProductVar(self,var):
        phi_prime={}
        for akey in self.PHI.keys():
            if re.search(var, akey):
                phi_prime[akey] = {}
                phi_prime[akey]= copy.deepcopy(self.PHI[akey])
                del self.PHI[akey]

        tau_key=re.sub(var, '', ' '.join(phi_prime.keys()))
        tau_key=re.sub("^ *", '', tau_key)
        tau_key=re.sub(" *$", '', tau_key)
        tau_key=re.sub("  *", ' ', tau_key)


        self.PHI[tau_key]= {}
        # Computation of XSI
        # This is where we multiply the factors present in phi_prime 
        self.XSI[var]={}
        self.XSI[var][tau_key]=[]
        for afactor in phi_prime.keys():
            self.XSI[var][tau_key]=self.XSI[var][tau_key]+phi_prime[afactor]

        count=1
        self.PHI[tau_key] = []
        self.PHI[tau_key] = [var]
    def traceBackMAP(self):
        #pprint.pprint(self.XSI)
        self.nodeList.reverse()
        #pprint.pprint(self.observationSet)
        for anode in self.nodeList:
            [p, self.observationSet[anode]] =  self.estimateMAP(anode,self.observationSet)
            print anode, p, self.observationSet[anode]
            #pprint.pprint(self.MAP)
            #pprint.pprint(self.observationSet)


    def estimateMAP (self, var, observationSet):
        lmap=-1000000
        local_obs=copy.deepcopy(observationSet)
        p_array=[]
        for aval in self.var_val:
            local_obs[var]=aval
            total_probability=0
            [elmNodes] = self.XSI[var].keys()
            updatedFactors=self.updateFactors(self.XSI[var][elmNodes], local_obs)
            #map_key=self.updateKey(elmNodes, local_obs)
            #print "HITX", map_key, var
            #try:
            #    total_probability+=self.MAP[var][map_key][0]
            #except:
            if var not in self.MAP:
               self.MAP[var]={}
            for afactor in updatedFactors:
                f_l=afactor.split(' ')
                if len(f_l) == 1:
                    total_probability+=self.estimateMAP(afactor,local_obs)[0]
                else:
                    total_probability+=log(self.cpd[afactor])
            #self.MAP[var][map_key]=[total_probability, aval]
            #print "HIT1", var, aval, lmap, map_key
            if total_probability > lmap: 
                [lmap, map_symbol] = [total_probability, aval]
        return [lmap, map_symbol]
    
    def updateKey(self,key_string,local_obs):
        for i in self.observationSet:
            key_string=re.sub(r'\b%s\b' % i,self.observationSet[i], key_string)
        for i in local_obs:
            key_string=re.sub(r'\b%s\b' % i, local_obs[i], key_string)
        return key_string    

    def updateFactors(self, factor_list, local_obs):
        result=[]
        for afactor in factor_list:
            for i in self.observationSet:
                afactor=re.sub(r'\b%s\b' % i,self.observationSet[i], afactor)
            for i in local_obs:
                afactor=re.sub(r'\b%s\b' % i, local_obs[i], afactor)
            result.append(afactor)
        return result
            
                                
                            
                        

    def updateObservations(self,observation):
       # print observation
        for afactor in self.PHI.keys():
            for aexpr in self.PHI[afactor]:
                _key=aexpr
                for akey in observation.keys():
                    _key = re.sub(r'\b%s\b' % akey, observation[akey], _key)
                self.PHI[afactor]=[_key]
    def restoreBN(self):
        self.PHI=copy.deepcopy(self.PHI_buffer)
    def saveBN(self):
        self.PHI_buffer=copy.deepcopy(self.PHI)
    def updateObservationSet(self):
        for aset in self.observationTable.keys():
            self.observationSet={}
            for aspecies in self.observationTable[aset].keys():
                self.observationSet[self.leaf_nodes[aspecies]]=self.observationTable[aset][aspecies]
            
my_bn=bayesianNet('./bn.csv', './column.fa')
var_list=['r3', 'r8', 'r14', 'r5', 'r10', 'r11', 'r15', 'r17', 'r19']
#var_list.reverse()
#my_bn.maxProduct(my_bn.ancestors.keys())
#pprint.pprint(my_bn.PHI)
my_bn.updateObservationSet()
print my_bn.observationSet
my_bn.updateObservations(my_bn.observationSet)
my_bn.maxProduct(var_list)
