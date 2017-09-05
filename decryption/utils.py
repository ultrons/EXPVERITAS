import re
import numpy as np

# Return a string containing the content of the file in the path
# All the characters converted to lovercase
# Characters except alphabet and space removed
def readText(path):
    return ' '.join(re.sub('[^a-z ]', '', line.strip().lower()) for line in
            open(path))

plain= "abcdefghijklmnopqrstuvwxyz "
cipher="plokmijnuhbygvtfcrdxeszaqw "

char2id={c:i for i,c in enumerate(plain)}
id2char={i:c for i,c in enumerate(plain)}

def toInt(c):
    return char2id[c]

def toStr(i):
    return id2char[i]

def toIntSeq(s):
    return [char2id[c] for c in s]

def toStrSeq(s):
    return [id2char[n] for n in s]

def encode(message):
    _message=''.join(re.sub('[^a-z ]', '', m.lower()) for m in message)
    return ''.join(cipher[char2id[m]] for m in _message)




def forward_backword (observations, startProbs, transitionProbs,
        emissionProbs):
    T=observations.shape[0]
    K=startProbs.shape[0]

    # Forward and backwards weight array
    f=np.zeros((T,K))
    b=np.zeros((T,K))
    def weight(ht_1, ht, t):
        if t == 0:
            return startProbs[ht]*emissionProbs[ht,observations[t]]
        return transitionProbs[ht_1, ht]*emissionProbs[ht,observations[t]]


    def forward (t,k):
        if t==0:
            f[t,k]=weight(0,k,t)
        else:
            for i in range(K):
                f[t,k]+=weight(i,k,t)*f[t-1,i]
        return f[t,k]

    def backword(t,k):
        if t==T-1:
            b[t,k]=1
        else:
            for i in range(K):
                b[t,k]+=weight(k,i,t+1)*b[t+1,i]
        return b[t,k]


    for t in range(T):
        for k in range(K):
            forward(t,k)

    for t in reversed(range(T)):
        for k in range(K):
            backword(t,k)
        #if(t%10 == 0):
        #    print("time step:%d" %t)
        #    print(b)

    mu=np.zeros((T,K))
    for t in range(T):
        for k in range(K):
            mu[t][k] = f[t,k]*b[t,k]
    #print('#######################')
    #print(f)
    #print(b)
    #print(mu)
    #print(observations)
    #print(startProbs)
    #print(transitionProbs)
    #print(emissionProbs)
    #print('#######################')
    mu = (mu/(mu.sum(1, keepdims=True)))
    return mu


if __name__ == '__main__':
    observations=np.random.randint(0,2, (3,1))
    startProbs=np.ones(2)/2
    transitionProbs=np.random.random((2,2))
    N=transitionProbs.sum(1, keepdims=True)
    transitionProbs/=N
    emissionProbs=np.random.random((2,2))
    N=emissionProbs.sum(1, keepdims=True)
    emissionProbs/=N
    print(forward_backword(observations, startProbs, transitionProbs,
        emissionProbs))









