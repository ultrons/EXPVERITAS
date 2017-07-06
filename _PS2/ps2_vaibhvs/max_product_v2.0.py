#!/usr/bin/python
import re
import copy
from math import log
import pprint
import operator

# Following implementation is an attempt to apply maximization based elemination
# concept to clique based information structure
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
        self.observationSet= {'r4': 'C', 'r16': 'G', 'r6': 'G', 'r7': 'C', 'r12': 'A', 'r1': 'C', 'r2': 'T', 'r9': 'A', 'r18': 'A', 'r13': 'A'}
                
        self.cpd={
            # Parent child conditional probability table
            'A' : {'A':0.831,
                   'C':0.032,
                   'G':0.085,
                   'T':0.052},
            'C' : {'A':0.046,
                   'C':0.816,
                   'G':0.028,
                   'T':0.110},
            'G' : {'A':0.122,
                   'C':0.028,
                   'G':0.808,
                   'T':0.042},
            'T' : {'A':0.053,
                   'C':0.076,
                   'G':0.029,
                   'T':0.842},
            'X' : {'A':0.295,
                   'T':0.295,
                   'C':0.205,
                   'G':0.205}
        }
                   
        self.PHI = {}
        self.ancestors = {}
        self.messageDB = {}
        self.children={}
        count = 1
        # BN is expected to be represented as ORDERED pair
        # N1, N2 means there is an edge from N1 -> N2
        file_pointer = open(BN, 'r')
        for aline in file_pointer:
            [tail, head] = aline.rstrip('\n').split(",")
            self.PHI[head] = [tail, head]
            if tail not in self.children:
                self.children[tail] = []
            self.children[tail].append(head)
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

                
    def collectPass(self,node):
        if node in self.messageDB:
            return self.messageDB[node][self.PHI[node][0]]
        else:
            # Check if children have communicated the message
            message={}
            self.messageDB[node]={}
            parent=self.PHI[node][0]
            self.messageDB[node][parent]={}
            for addressee in self.var_val:
                message[addressee]=0
                if node in self.observationSet:
                    message[addressee]=log(self.cpd[addressee][self.observationSet[node]])
                else:    
                    # Marginalizing current node
                    pr={}
                    for sender in self.var_val:
                        # Check if nodes have children    
                        pr[sender]=0
                        if node in self.children:
                            for child in self.children[node]:
                                if child in self.messageDB:
                                    pr[sender]+=self.messageDB[child][node][sender]
                                else:
                                    pr[sender]+=self.collectPass(child)[sender]
                        pr[sender]+=log(self.cpd[addressee][sender])
                    message[addressee]+=max(pr.values())
                self.messageDB[node][parent][addressee]=message[addressee]
        return message
    
    # This routine goes to each of ancestor nodes and compute the message to the
    # each of it's child
    def distributePass(self,node):
        if node in self.children:
            for child in self.children[node]:
                if child in self.messageDB[node]:
                    #return self.messageDB[node][child]
                    continue
                else:
                    message={}
                    self.messageDB[node][child]={}
                    message[child]={}
                    # Each node has parent and two chidren 
                    # Except leaf nodes and root node
                    # Leafs are excluded because of the if check at the begining
                    # So the only possible exception is root
                    # Since every neighbour except the target is sending message
                    # We create and sender's (schindle's list ;))
                    senders_list=copy.deepcopy(self.children[node])
                    senders_list.remove(child)
                    for target_val in self.var_val:
                        message[child][target_val]=0
                        p={}
                        for port in self.var_val:
                            p[port]=0    
                            if (self.PHI[node][0] != 'X') :
                                if node in self.messageDB[self.PHI[node][0]]:
                                    p[port]+=self.messageDB[self.PHI[node][0]][node][port]
                                else: 
                                    p[port]+=self.distributePass(self.PHI[node][0])[node][port] 
                                p[port]+=log(self.cpd[port][target_val])

                        message[child][target_val]=max(p.values())
                        for asender in senders_list: 
                            message[child][target_val]+=self.messageDB[asender][node][target_val]
                        if self.PHI[node][0] == 'X':
                            message[child][target_val]+=log(self.cpd['X'][target_val])

                        self.messageDB[node][child][target_val]=message[child][target_val]
        return message

    def computePosterior(self):
        self.posterior={}
        for anode in self.ancestors:
            self.posterior[anode]={}
            for target_val in self.var_val:
                self.posterior[anode][target_val]=0   
                #Since a clique in current example has two variable in it's
                #scope, to compute the posterior of the node we need to
                #marginalize the other variable
                #
                p={}
                for port in self.var_val:
                    p[port]=0
                    if self.PHI[anode][0] != 'X':
                        p[port]+=self.messageDB[self.PHI[anode][0]][anode][port]
                        p[port]+=log(self.cpd[port][target_val])
                self.posterior[anode][target_val]+=max(p.values())
                for asender in self.children[anode]:
                    self.posterior[anode][target_val]+=self.messageDB[asender][anode][target_val]
                if self.PHI[anode][0] == 'X':
                    self.posterior[anode][target_val]+=log(self.cpd['X'][target_val])
        pprint.pprint(self.posterior)

    # Following routine will marginalize child to get parent posterior
    # probability in order to cross check with the other approach main
    # computePosterior sub-routine

    def checkPosterior(self,target_node):
        posterior={}
        for target_val in self.var_val:
            posterior[target_val]=0
            p={}
            for port in self.var_val:
                p[port]=0
                for asender in self.children[target_node]:
                    p[port]+=self.messageDB[asender][target_node][port]
                p[port]+=log(self.cpd[target_val][port])
            posterior[target_val]+=max(p.values())
            posterior[target_val]+=self.messageDB[self.PHI[target_node][0]][target_node][target_val]
        #s=sum(posterior.values())
        #for val in posterior:
        #    posterior[val]/=s
        pprint.pprint(posterior)

    def computeMap (self):
        Map={}
        for anode in self.posterior:
            p=-10000
            for asymbol in self.posterior[anode]:
                if self.posterior[anode][asymbol] > p:
                    Map[anode]=asymbol
                    p = self.posterior[anode][asymbol]
        print "###############"
        print "Highest probable Bases based on marginals for ancestors:"
        pprint.pprint(Map)
        return Map






my_bn=bayesianNet('./bn.csv', './column.fa')
# Distribution Pass requires only one iteration if root node is given as a start
# point
my_bn.collectPass('r19')
# Collection Pass is called three times for three penultimate nodes
# In order have all downstream message computed.
# Each pass is storing and using computation done earlier (messageDB
# dictionary)therefore it is 
# NOT recomputing any value.
my_bn.distributePass('r3')
my_bn.distributePass('r8')
my_bn.distributePass('r14')
print my_bn.messageDB['r17']['r15']
my_bn.computePosterior()
my_bn.checkPosterior('r3')
my_bn.computeMap()

#pprint.pprint(my_bn.messageDB)

#pr=0
#for i in my_bn.messageDB['r19']['X']:
#  pr+= my_bn.cpd['X'][i]*my_bn.messageDB['r19']['X'][i]

#print pr

