import pandas as pd
import numpy as np
import utilities
import matplotlib.pyplot as plt
from phik import phik_matrix
import LogisticRegression as lr

#To disable sections of code
ANALYSE_MODE = True
REGRESSION_MODE = False

#Load csv into a pandas dataframe
titanic_dataset = pd.read_csv('./Titanic.csv')

#Remove unwanted columns
titanic_dataset = titanic_dataset.drop(["Cabin", "Name", "Ticket"], axis=1)

#Remove rows with null values (null age values)
titanic_dataset = titanic_dataset.dropna()

#One hot encode binary data
titanic_dataset = pd.get_dummies(titanic_dataset, columns=['Sex'])
titanic_dataset = pd.get_dummies(titanic_dataset, columns=['Pclass'])
titanic_dataset = pd.get_dummies(titanic_dataset, columns=['Embarked'])

#Put analytical functions seperate so will not have to run every time
if ANALYSE_MODE:

    #Show data has correctly loaded setup
    print(titanic_dataset)

    #Initial analyis of data using built in describe function
    print(titanic_dataset.describe(include='all'))

    #Show corelation values in relation to all variables and survived
    print(phik_matrix(titanic_dataset, interval_cols=['PassengerId', 'Age', 'SibSp', 'Fare'])['Survived'])
    
    #Create new seperate dataframes of only Survived and Non Survived passengers
    titanic_dataset_SURVIVED_True = titanic_dataset[titanic_dataset['Survived']==1]
    titanic_dataset_SURVIVED_False = titanic_dataset[titanic_dataset['Survived']==0]
    #Plot the frequencys of the survivors divided by frequencys of all passengers to remove bias from the initial distribution of ages
    num_of_bins = 50
    ageCountsTotal, binEdge = np.histogram(titanic_dataset['Age'], bins = num_of_bins)
    binWidth = np.diff(binEdge)
    ageCountsSurvived, _ = np.histogram(titanic_dataset_SURVIVED_True['Age'], bins = binEdge)
    plt.bar(np.linspace(binEdge[0],binEdge[-2], num_of_bins)+binWidth/2, ageCountsSurvived/ageCountsTotal, width=binWidth)
    plt.title('Age')
    plt.show()

if REGRESSION_MODE:

    #Normalise continuous data
    titanic_dataset['Age']= titanic_dataset['Age']/titanic_dataset['Age'].max()
    titanic_dataset['Fare']= titanic_dataset['Fare']/titanic_dataset['Fare'].max()

    #Create new column for has family based on parch and sibsp
    titanic_dataset['HasFamily']= (titanic_dataset['Parch'] != 0) | (titanic_dataset['SibSp'] != 0)

    #Create matrix of features/input only then add a all ones column for the bias weight, b0
    featureMatrix = np.matrix(titanic_dataset.drop(["Survived", 'PassengerId', 'Parch', 'SibSp'], axis=1).values).astype(float)
    featureMatrix = np.c_[featureMatrix, np.ones((featureMatrix.shape[0], 1))]
    #Create vector of actual outcomes
    outputVector = np.matrix(titanic_dataset["Survived"].values).astype(float).T
    #Create training, testing subsets
    trainingFeatureMatrix,trainingOutputVector,testingFeatureMatrix,testingOutputVector = utilities.create_TrainingTesting_subsets(featureMatrix,outputVector,0.95)

    #Create train and test model
    model = lr.LogisticModel()
    model.fit(trainingFeatureMatrix, trainingOutputVector, 0.001, 0.0000001, 100000)
    print(model.test(testingFeatureMatrix, testingOutputVector,0.0))
    print(model.report_model_status())

    #Write predictions from model to excel file and add predictions to the dataframe
    predictions = model.predict(featureMatrix)
    titanic_dataset['Predictions'] = predictions
    titanic_dataset.to_excel('./predictions.xlsx')
