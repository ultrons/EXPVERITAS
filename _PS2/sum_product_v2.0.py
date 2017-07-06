#!/usr/bin/python
import re
import copy
from math import log
import pprint


from optparse import OptionParser
parser = OptionParser()
parser.add_option("-t", "--tree", dest="tree", help="Tree Definition File")
parser.add_option("-e", "--evidence", dest="evidence", help="evidence File")

(options, args) = parser.parse_args()
#
#print options.section
#exit()




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
        #self.observationSet= {'r4': 'C', 'r16': 'G', 'r6': 'G', 'r7': 'C', 'r12': 'A', 'r1': 'C', 'r2': 'T', 'r9': 'A', 'r18': 'A', 'r13': 'A'}
        self.observationSet= {}
                
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
        # Initializing log likelihood to 0
        self.LL=0
        # BN is expected to be represented as ORDERED pair
        # N1, N2 means there is an edge from N1 -> N2
        file_pointer = open(BN, 'r')
        print "Reading Tree from: ", BN
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
        print "Reading Tree from: ", observationRecord
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
                    message[addressee]+=self.cpd[addressee][self.observationSet[node]]
                else:    
                    # Marginalizing current node
                    for sender in self.var_val:
                        # Check if nodes have children    
                        pr=1
                        if node in self.children:
                            for child in self.children[node]:
                                if child in self.messageDB:
                                    pr*=self.messageDB[child][node][sender]
                                else:
                                    pr*=self.collectPass(child)[sender]
                        pr*=self.cpd[addressee][sender]
                        message[addressee]+=pr
                self.messageDB[node][parent][addressee]=message[addressee]
        return message
    
    # This routine goes to each of ancestor nodes and compute the message to the
    # each of it's child
    def distributePass(self,node):
        if node in self.children:
            for child in self.children[node]:
                if child in self.messageDB[node]:
                    continue
                    #return self.messageDB[node][child]
                else:
                    self.messageDB[node][child]={}
                    message={}
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
                        for port in self.var_val:
                            p=1    
                            if (self.PHI[node][0] != 'X') :
                                if node in self.messageDB[self.PHI[node][0]]:
                                    p*=self.messageDB[self.PHI[node][0]][node][port]
                                else: 
                                    p*=self.distributePass(self.PHI[node][0])[node][port] 
                                p*=self.cpd[port][target_val]
                                message[child][target_val]+=p
                            else:
                                message[child][target_val]=1
                        for asender in senders_list:
                            if node in self.messageDB[asender]:
                                message[child][target_val]*=self.messageDB[asender][node][target_val]
                            else:
                                message[child][target_val] *=self.distributePass(asender)[node][target_val] 
                        if self.PHI[node][0] == 'X':
                            message[child][target_val]*=self.cpd['X'][target_val]

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
                for port in self.var_val:
                    p=1
                    if self.PHI[anode][0] != 'X':
                        p*=self.messageDB[self.PHI[anode][0]][anode][port]
                        p*=self.cpd[port][target_val]
                        self.posterior[anode][target_val]+=p
                    else:
                        self.posterior[anode][target_val]=1

                for asender in self.children[anode]:
                    self.posterior[anode][target_val]*=self.messageDB[asender][anode][target_val]
                if self.PHI[anode][0] == 'X':
                    self.posterior[anode][target_val]*=self.cpd['X'][target_val]
            s=sum(self.posterior[anode].values())
            #print "DBG1 : ", anode, s
            for val in self.posterior[anode]:
                self.posterior[anode][val]/=s
        print "###############"
        print "Posterior for Each of the Ancestor Node:"
        pprint.pprint(self.posterior)

    # Following routine will marginalize child to get parent posterior
    # probability in order to cross check with the other approach main
    # computePosterior sub-routine

    def checkPosterior(self,target_node):
        posterior={}
        for target_val in self.var_val:
            posterior[target_val]=0
            for port in self.var_val:
                p=1
                for asender in self.children[target_node]:
                    p*=self.messageDB[asender][target_node][port]
                p*=self.cpd[target_val][port]
                posterior[target_val]+=p
            posterior[target_val]*=self.messageDB[self.PHI[target_node][0]][target_node][target_val]
        s=sum(posterior.values())
        for val in posterior:
            posterior[val]/=s
        pprint.pprint(posterior)

    def computeMap (self):
        Map={}
        for anode in self.posterior:
            p=0
            for asymbol in self.posterior[anode]:
                if self.posterior[anode][asymbol] > p:
                    Map[anode]=asymbol
                    p = self.posterior[anode][asymbol]
        print "###############"
        print "Highest probable Bases based on marginals for ancestors:"
        pprint.pprint(Map)
        return Map
    def estimateLL(self):
        pr=0
        # Computing using root Node: DBG1 Message Verifies that it is same for
        # all
        for i in my_bn.var_val:
           pr+= (my_bn.cpd['X'][i]*my_bn.messageDB['r17']['r19'][i]*my_bn.messageDB['r18']['r19'][i])
        print "###############"
        print "ObservationSet: ", self.observationSet
        print "###############"
        print "Liklihood: ", pr, log(pr)
        print "###############"
        self.LL+=log(pr)
    def resetMessageTables(self):
        self.messageDB = {}
        self.posterior = {}

        




#my_bn=bayesianNet('./bn.csv', './column.fa')
my_bn=bayesianNet(options.tree, options.evidence)
for aset in my_bn.observationTable.keys():
    observationSet={}
    for aspecies in my_bn.observationTable[aset].keys():
        observationSet[my_bn.leaf_nodes[aspecies]]=my_bn.observationTable[aset][aspecies]
    my_bn.observationSet=observationSet
    # Distribution Pass requires only one iteration if root node is given as a start
    # point
    my_bn.collectPass('r19')
    print "Collect Pass Complete ..."
    print "###############"
    print "Printing Message Table: Format: Message[from][to] ..{ Table}"
    pprint.pprint(my_bn.messageDB)
    my_bn.estimateLL()
    # Collection Pass is called three times for three penultimate nodes
    # In order have all downstream message computed.
    # Each pass is storing and using computation done earlier (messageDB
    # dictionary)therefore it is 
    # NOT recomputing any value.
    my_bn.distributePass('r3')
    my_bn.distributePass('r8')
    my_bn.distributePass('r14')
    print "Distribute Pass Complete ..."
    my_bn.computePosterior()
    # Printing out r5 posterior computed from r3, r5 clique
    # Just verify that cliques agree on the posterior
    my_bn.checkPosterior('r3')
    my_bn.computeMap()
    my_bn.resetMessageTables()
print "###############"
print "Total Likelihood considering all all observation Sts IID: ", my_bn.LL
    
#pprint.pprint(my_bn.messageDB)


