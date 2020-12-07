import random
import numpy as np


class ActiveAnnotation:

    def __init__(self, data, preprocessor):
        self.dataPool = data
        self.dataPoolProbs = None
        self.XtrainPos = []
        self.XtrainNeg = []
        self.posWeights = None
        self.negWeights = None
        self.classifier = None
        self.preprocessor = preprocessor

    def _negative_sample(self, n):
        sample = random.sample(self.dataPool, k=n)
        self.XtrainNeg.extend(sample)

    def heuristic_labeling(self, keywords):
        self.XtrainPos.extend([txt for txt in self.dataPool if any(w in txt.lower() for w in keywords)])
        for txt in self.XtrainPos:
            self.dataPool.remove(txt)
        
        self._negative_sample(n=len(self.XtrainPos))
        for txt in self.XtrainNeg:
            self.dataPool.remove(txt)
            
        self.posWeights = np.array([1]*len(self.XtrainPos))
        self.negWeights = np.array([1]*len(self.XtrainNeg))

    def train_model(self, classifier):
        self.classifier = classifier
        trainX = self.preprocessor(self.XtrainPos + self.XtrainNeg)
        trainY = np.array([1]*len(self.XtrainPos) + [0]*len(self.XtrainNeg))
        testX = self.preprocessor(self.dataPool)
        weights = np.concatenate([self.posWeights, self.negWeights])

        self.classifier.fit(trainX, trainY, sample_weight = weights)
        self.dataPoolProbs = classifier.predict_proba(testX)[:,1]

    def _sample(self, strategy = "high", n=20):
        if strategy == "ambigous":
            ind = np.where(np.logical_and(self.dataPoolProbs>.45, self.dataPoolProbs<.55))
            return random.sample([self.dataPool[i] for i in ind[0]], k=n)
        elif strategy == "high":
            ind = np.where(.8 < self.dataPoolProbs)
            return random.sample([self.dataPool[i] for i in ind[0]], k=n)

        elif strategy == "low":
            ind = np.where(self.dataPoolProbs < .1)
            return random.sample([self.dataPool[i] for i in ind[0]], k=n)
        
        elif strategy == "random":
            return random.sample(self.dataPool, k=n)
        else:
            raise Exception("strategy must be one of; ambigous, high, low or random")
        
        
    def annotate(self, strategy="high", n = 20, weight = 5):
        sample = self._sample(strategy = strategy, n=n)

        for item in sample:
            print("")
            print(item)
            label = input("Is this a True label?")

            if int(label) == 1:
                self.XtrainPos.append(item)
                self.posWeights = np.append(self.posWeights, weight)
                self.dataPool.remove(item)

            if int(label) == 0:
                self.XtrainNeg.append(item)
                self.negWeights = np.append(self.negWeights, weight)
                self.dataPool.remove(item)
