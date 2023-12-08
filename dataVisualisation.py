import pandas as pd
import numpy as np
import utilities
import matplotlib.pyplot as plt
from phik import phik_matrix
import LogisticRegression as lr

#To disable sections of code
ANALYSE_MODE = False
REGRESSION_MODE = True

#Load csv into a pandas dataframe
titanicCSV = pd.read_csv('./Titanic.csv')

#Remove unwanted columns
titanicCSV = titanicCSV.drop(["Cabin", "Name", "Ticket"], axis=1)

#Remove rows with null values (null age values)
titanicCSV = titanicCSV.dropna()

#Create new seperate dataframes of only Survived and Non Survived passengers
titanicCSV_SURVIVED_True = titanicCSV[titanicCSV['Survived']==1]
titanicCSV_SURVIVED_False = titanicCSV[titanicCSV['Survived']==0]

#One hot encode binary data
titanicCSV = pd.get_dummies(titanicCSV, columns=['Sex'])
titanicCSV = pd.get_dummies(titanicCSV, columns=['Pclass'])
titanicCSV = pd.get_dummies(titanicCSV, columns=['Embarked'])

#Put analytical functions seperate so will not have to run every time
if ANALYSE_MODE:

    #Show data is correctly setup
    print(titanicCSV)

    #Initial analyis of data using built in describe function
    print(titanicCSV.describe(include='all'))

    #Show corelation values in relation to all variables and survived
    print(phik_matrix(titanicCSV, interval_cols=['PassengerId', 'Age', 'SibSp', 'Fare'])['Survived'])

    #Find frequency of survival for age ranges, then divide by frequency of all ages to remove offset
    num_of_bins = 20
    #Calculate Y(frequency) values for histogram of all ages
    ageCounts, binEdge = np.histogram(titanicCSV['Age'], bins = num_of_bins)
    #Calculate Y(frequency) values for histogram of ages of only the survivors
    ageCountsAlive, _ = np.histogram(titanicCSV_SURVIVED_True['Age'], bins = binEdge)
    #Plot the frequencys of the survivors divided by frquencys of all passengers to remove bias from the initial distribution of ages
    plt.bar(np.linspace(0,80, num_of_bins), ageCountsAlive/ageCounts, width=np.diff(binEdge))
    plt.title('Age')
    plt.show()

if REGRESSION_MODE:

    #Create matrix of features/input only then add a all ones column for the bias weight, b0
    featureMatrix = np.matrix(titanicCSV.drop(["Survived", 'PassengerId'], axis=1).values).astype(float)
    featureMatrix = np.c_[featureMatrix, np.ones((featureMatrix.shape[0], 1))]
    #Create vector of actual outcomes
    outputVector = np.matrix(titanicCSV["Survived"].values).astype(float).T
    #Create training, testing subsets
    trainingFeatureMatrix,trainingOutputVector,testingFeatureMatrix,testingOutputVector = utilities.create_TrainingTesting_subsets(featureMatrix,outputVector,0.95)

    #Create train and test model
    model = lr.LogisticModel()
    model.fit(trainingFeatureMatrix, trainingOutputVector, 0.001, 0.0000001, 100000)
    print(model.test(testingFeatureMatrix, testingOutputVector,0))
    print(model.report_model_status())

    #Write predictions from model to excel file and add predictions to the dataframe
    predictions = model.predict(featureMatrix)
    titanicCSV['Predictions'] = predictions
    titanicCSV.to_excel('./predictions.xlsx')
