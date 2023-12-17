import numpy as np
import random

def create_TrainingTesting_subsets(featureMatrix, outputVector, trainingProportion):
    
    trainingSetSize = round(trainingProportion*len(outputVector))
    trainingSetIndex = random.sample(range(0, len(outputVector)-1), trainingSetSize)
    
    i = 0
    trainingFeatureMatrix = []
    testingFeatureMatrix = []
    for row in featureMatrix:
        if i in trainingSetIndex:
            trainingFeatureMatrix.extend(row.tolist())
        else:
            testingFeatureMatrix.extend(row.tolist())
        i+=1
        
    i = 0
    trainingOutputVector = []
    testingOutputVector = []
    for row in outputVector:
        if i in trainingSetIndex:
            trainingOutputVector.extend(row.tolist())
        else:
            testingOutputVector.extend(row.tolist())
        i+=1
    
    trainingFeatureMatrix = np.matrix(trainingFeatureMatrix)
    trainingOutputVector = np.matrix(trainingOutputVector)
    testingFeatureMatrix = np.matrix(testingFeatureMatrix)
    testingOutputVector = np.matrix(testingOutputVector)
    
    return trainingFeatureMatrix,trainingOutputVector,testingFeatureMatrix,testingOutputVector
