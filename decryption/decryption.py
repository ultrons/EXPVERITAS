import numpy as np
import utils

K=27
startProbs=np.ones(K)/K

# Derriving Transition Probabilty
data=utils.toIntSeq(utils.readText('sample.txt'))
transCounts=np.zeros((K,K))
for i in range(1,len(data)):
    p, n = data[i-1], data[i]
    transCounts[p, n]+=1

transitionProbs=transCounts/transCounts.sum(1, keepdims=True)



code=utils.encode("I love you")
#code=utils.encode("So I lived my life alone, without anyone that I could really talk to, untl I had an accident with my plane in the Desert of Sahara")
#code=utils.encode("So I lived my life alone, without anyone that I could really talk to, untl I had an accident with my plane in the Desert of Sahara, six years ago, Something was broken in my engine. And as I with me neither a mechanic nor any passengers, I set myself to attempt the difficult repair alone")
#code=utils.encode("So I lived my life alone, without")
print(code)
observations = np.array(utils.toIntSeq(code))
emissionProbs=np.ones((K,K))/K
#emissionProbs =np.random.random((K,K))
#emissionProbs=emissionProbs/emissionProbs.sum(1, keepdims=True)


#### EM

for i in range(200):
    # E-Step
    mu = utils.forward_backword(observations, startProbs, transitionProbs,
            emissionProbs)

    print(''.join(utils.toStrSeq(list(np.argmax(mu, axis=1)))))

    # M-Step
    emissionCounts = np.zeros((K,K))
    for t, e in enumerate(observations):
        for h in range(K):
            emissionCounts[h,e]+=mu[t,h]
    emissionProbs=emissionCounts/emissionCounts.sum(1, keepdims=True)
