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

#Creates onehot array from ranges, for lists of numbers
def create_2d_array(original_list, ranges):
    result_2d_array = []
    for value in original_list:
        row = [1 if range[0] <= value <= range[1] else 0 for range in ranges]
        result_2d_array.append(row)

    return np.array(result_2d_array)
