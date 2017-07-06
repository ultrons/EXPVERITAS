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

                
    def computeMessage(self,node):
        if node in self.messageDB:
            return self.messageDB[node]
        else:
            print "Query Node: ", node
            # Check if children have communicated the message
            message={}
            self.messageDB[node]={}
            for addressee in self.var_val:
                message[addressee]=0
                # Marginalizing current node
                if node in self.observationSet:
                    message[addressee]+=self.cpd[addressee][self.observationSet[node]]
                else:    
                    for sender in self.var_val:
                        # Check if nodes have children    
                        pr=1
                        if node in self.children:
                            for child in self.children[node]:
                                if child in self.messageDB:
                                    pr*=self.messageDB[child][sender]
                                else:
                                    pr*=self.computeMessage(child)[sender]
                        pr*=self.cpd[addressee][sender]
                        message[addressee]+=pr
                self.messageDB[node][addressee]=message[addressee]
        return message



my_bn=bayesianNet('./bn.csv', './column.fa')
my_bn.computeMessage('r19')

pprint.pprint(my_bn.messageDB)
pr=0
for i in my_bn.messageDB['r19']:
   pr+= my_bn.cpd['X'][i]*my_bn.messageDB['r19'][i]

print pr

