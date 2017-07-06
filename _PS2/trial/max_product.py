#!/usr/bin/python
import re
import copy
from math import log

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

        self.cpd_aux={}
        self.cpd={
            # Parent child conditional probability table
            'AA':0.831,
            'AC':0.032,
            'AG':0.085,
            'AT':0.052,
            'CA':0.046,
            'CC':0.816,
            'CG':0.028,
            'CT':0.110,
            'GA':0.122,
            'GC':0.028,
            'GG':0.808,
            'GT':0.042,
            'TA':0.053,
            'TC':0.076,
            'TG':0.029,
            'TT':0.842,
            # Priors for root Node
            'XA':0.295,
            'XT':0.295,
            'XC':0.205,
            'XG':0.205
        }
        
        #purpose of this disctionar is to return the max probable conditional
        #and under what value of the other variable it is going to be maximum
        #Could have been generated using a function over self.cpd, but easy
        #enough to hard code
        self.max_pr = {
            'A':[0.831, 'A'],
            'C':[0.816, 'C'],
            'G':[0.808, 'G'],
            'T':[0.842, 'T']
        }
                   
        self.PHI = {}
        self.XSI={}
        self.ancestors = {}
        count = 1
        # BN is expected to be represented as ORDERED pair
        # N1, N2 means there is an edge from N1 -> N2
        file_pointer = open(BN, 'r')
        for aline in file_pointer:
            [tail, head] = aline.rstrip('\n').split(",")
            self.PHI[tail + '' + head] = {}
            self.PHI[tail + '' + head][count] = []
            self.PHI[tail + '' + head][count].append(tail + '' + head)
            if tail not in self.leaf_nodes.values() and tail != 'X' :
                # Creating the collection of ancestors
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

                
    def maxProductEleminateVar(self,nodeList):
       for anode in nodeList:
           # Instead of returning two values PHI and XSI, this function would
           # update the correspoding datastructure visible across this class
           self.maxProductEleminateVarVar(anode)

    def maxProductEleminateVarVar(self,var):
        phi_prime={}
        for akey in self.PHI.keys():
            if re.search(var, akey):
                phi_prime[akey] = {}
                phi_prime[akey]= copy.deepcopy(self.PHI[akey])
                del self.PHI[akey]

        tau_key=re.sub(var, '', ''.join(phi_prime.keys()))
        self.xsi_key=":".join([var,tau_key])


        self.XSI[self.xsi_key]={}
        self.PHI[self.xsi_key]= {}
        # Computation of XSI
        # This is where we multiply the factors present in phi_prime 

        for afactor in phi_prime.keys():
            self.multiplyFactors(phi_prime[afactor])

        count=1
        for akey in self.XSI[self.xsi_key].keys():
            self.PHI[self.xsi_key][akey] = self.XSI[self.xsi_key][akey]
            #for aexpr in self.XSI[akey]:
            #    self.PHI[tau_key][count].append(re.sub(var, val, aexpr))
            count+=1
                
        #print self.PHI[tau_key]


    def estimate_probability(self, observationSet):
        total_probability=0

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
                            #if p in observationSet.keys(): 
                            #    p_c=p_c.replace(p, observationSet[p])
                            #if c in observationSet.keys(): 
                            #    p_c=p_c.replace(c, observationSet[c])
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
        if (len(self.XSI[self.xsi_key].keys()) == 0):
            self.XSI[self.xsi_key] = copy.deepcopy(factor)
        else:
            buffer = copy.deepcopy(self.XSI[self.xsi_key])
            self.XSI[self.xsi_key]={}
            count=1
            for aexpr in factor.keys():
                for oexpr in buffer.keys():
                    self.XSI[self.xsi_key][count]=buffer[oexpr]
                    self.XSI[self.xsi_key][count].append(factor[aexpr])
                    count+=1

    def updateObservations(self,observation):
        for afactor in self.PHI.keys():
            for aexpr in self.PHI[afactor][1]:
                _key=aexpr
                for akey in observation.keys():
                    _key = re.sub(r'%s\b' % akey, observation[akey], _key)
                self.PHI[afactor][1]=[_key]
    def restoreBN(self):
        self.PHI=copy.deepcopy(self.PHI_buffer)
    def saveBN(self):
        self.PHI_buffer=copy.deepcopy(self.PHI)
    def traceBack(self, var_list):
        self.MAP={}
        var_list.reverse()
        maxXSI={}
        for akey in self.XSI.keys():
            max_key=akey.replace(":.*","")
            maxXSI[max_key]=akey
        for var in var_list:
            self.MAP_Buffer={}
            self.MAP[var]=self.estimateMAP(var,maxXSI[var], self.XSI[maxXSI[var]])

        return self.MAP
    def estimateMAP(self,var, xsi_key, xsi_expr):
        #echo =1
        for val in var_val:
            replace_var(xsi_expr, val)

        next_xsi_key=xsi_key.replace("^[^:]:,'')
        #next_xsi_key=xsi_key.replace("^[^:]:.*\(r[^:]*\)",\2)
        if re.match(".*:", xsi_key):



        
        
            

            
#my_bn=bayesianNet('./bn.csv', './multicolumn.fa')
my_bn=bayesianNet('./bn.csv', './column.fa')
#my_bn.maxProductEleminateVar(my_bn.ancestors.keys())
var_list=['r3', 'r8', 'r14', 'r5', 'r10', 'r11', 'r15', 'r17', 'r19']

for aset in my_bn.observationTable.keys():
    observationSet={}
    for aspecies in my_bn.observationTable[aset].keys():
        observationSet[my_bn.leaf_nodes[aspecies]]=my_bn.observationTable[aset][aspecies]




print ""
my_bn.updateObservations(observationSet)
print my_bn.PHI
my_bn.maxProductEleminateVar(var_list)
print my_bn.XSI
exit()
my_bn.traceBack(var_list)
prob_results=[]
prob_results.append(my_bn.estimate_probability(observationSet))
    #my_bn.restoreBN()
log_likelihood=0
for i in prob_results:
    log_likelihood+=log(i)

print log_likelihood

