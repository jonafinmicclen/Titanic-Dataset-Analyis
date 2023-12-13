import pandas as pd
import numpy as np
import utilities
import matplotlib.pyplot as plt
from phik import phik_matrix
import LogisticRegression as lr
import seaborn as sns

#To disable sections of code
ANALYSE_MODE = False
REGRESSION_MODE = True

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
    
    sns.kdeplot(data=titanic_dataset, x='Age', fill=True)
    sns.kdeplot(data=titanic_dataset_SURVIVED_True, x='Age', fill=True)
    sns.kdeplot(data=titanic_dataset_SURVIVED_False, x='Age', fill=True)
    plt.show()
    
    

if REGRESSION_MODE:

    #Normalise continuous data
    titanic_dataset['Age']= titanic_dataset['Age']/titanic_dataset['Age'].max()
    titanic_dataset['Fare']= titanic_dataset['Fare']/titanic_dataset['Fare'].max()

    #Create new column for has family based on parch and sibsp
    titanic_dataset['HasFamily']= (titanic_dataset['Parch'] != 0) | (titanic_dataset['SibSp'] != 0)

    #Create matrix of features
    featureMatrix = np.matrix(titanic_dataset.drop(["Survived", 'PassengerId', 'Parch', 'SibSp'], axis=1).values).astype(float)
    #Create vector of actual outcomes
    outputVector = np.matrix(titanic_dataset["Survived"].values).astype(float).T
    #Create training, testing subsets
    trainingFeatureMatrix,trainingOutputVector,testingFeatureMatrix,testingOutputVector = utilities.create_TrainingTesting_subsets(featureMatrix,outputVector,0.95)

    #Create train and test model
    model = lr.LogisticModel()
    model.fit(trainingFeatureMatrix, trainingOutputVector, 0.001, 0.0000001, 100000, bias=True)
    print(model.test(testingFeatureMatrix, testingOutputVector,0.0))
    print(model.report_model_status())

    #Write predictions from model to excel file and add predictions to the dataframe
    predictions = model.predict(featureMatrix)
    titanic_dataset['Predictions'] = predictions
    titanic_dataset.to_excel('./predictions.xlsx')
