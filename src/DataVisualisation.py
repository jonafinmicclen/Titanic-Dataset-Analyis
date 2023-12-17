import pandas as pd
import numpy as np
import Utilities
import matplotlib.pyplot as plt
from phik import phik_matrix
import LogisticRegression as lr
import seaborn as sns
from scipy.stats import pointbiserialr

ANALYSE_MODE = True #Do not set both true
REGRESSION_MODE = False

#Load csv into a pandas dataframe
titanic_dataset = pd.read_csv('./data/Titanic.csv')

if ANALYSE_MODE:
    #Initial look into data
    print(titanic_dataset.describe(include='all'))

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

    #Show phik matrix of just survived excluding non categorical data
    print(phik_matrix(titanic_dataset.drop(['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare'], axis=1))['Survived'].drop('Survived'))

    #Corelations of numerical data
    print(titanic_dataset[['Age', 'SibSp', 'Parch', 'Fare']].corr())

    #Create new column has family if either parch or sibsp is not 0 and test the corelation with point biserial
    titanic_dataset['HasFamily']= (titanic_dataset['Parch'] != 0) | (titanic_dataset['SibSp'] != 0)
    print('Corelation of HasFamily and Survived ', pointbiserialr(titanic_dataset['HasFamily'], titanic_dataset['Survived']))

    #Create new seperate dataframes of only Survived and Non Survived passengers
    titanic_dataset_SURVIVED_True = titanic_dataset[titanic_dataset['Survived']==1]
    titanic_dataset_SURVIVED_False = titanic_dataset[titanic_dataset['Survived']==0]
    
    #KDE plots of Age, PassengerID, and Fare
    sns.kdeplot(data=titanic_dataset, x='Age', fill=False, color='blue')
    sns.kdeplot(data=titanic_dataset_SURVIVED_True, x='Age', fill=False, color='green')
    plt.show()

    sns.kdeplot(data=titanic_dataset, x='PassengerId', fill=False, color='blue')
    sns.kdeplot(data=titanic_dataset_SURVIVED_True, x='PassengerId', fill=False, color='green')
    plt.show()

    sns.kdeplot(data=titanic_dataset, x='Fare', fill=False, color='blue')
    sns.kdeplot(data=titanic_dataset_SURVIVED_False, x='Fare', fill=False, color='red')
    plt.show()
    

if REGRESSION_MODE:

    #Split age and fare into bins
    age_bins = [0, 14, 31, 45, float('inf')]
    titanic_dataset['Age'] = pd.cut(titanic_dataset['Age'], bins=age_bins, right=False)

    fare_bins = [0, 30, 100, float('inf')]
    titanic_dataset['Fare'] = pd.cut(titanic_dataset['Fare'], bins=fare_bins, right=False)

    titanic_dataset = pd.get_dummies(titanic_dataset, columns=['Age', 'Fare'])

    #Create new column for has family based on parch and sibsp
    titanic_dataset['HasFamily']= (titanic_dataset['Parch'] != 0) | (titanic_dataset['SibSp'] != 0)

    #Remove unwanted columns
    titanic_dataset = titanic_dataset.drop(['PassengerId', 'Parch', 'SibSp'], axis=1)

    #Create matrix of features
    featureMatrix = np.matrix(titanic_dataset.drop(['Survived'], axis=1).values).astype(float)
    #Create vector of actual outcomes
    outputVector = np.matrix(titanic_dataset["Survived"].values).astype(float).T
    #Create training, testing subsets
    trainingFeatureMatrix,trainingOutputVector,testingFeatureMatrix,testingOutputVector = Utilities.create_TrainingTesting_subsets(featureMatrix,outputVector,0.95)

    #Create train and test model
    model = lr.LogisticModel()
    model.fit(trainingFeatureMatrix, trainingOutputVector, 0.001, 0.0000001, 100000, bias=True)
    print(model.test(testingFeatureMatrix, testingOutputVector,0.0))
    print(model.report_model_status())

    #Write predictions from model to excel file and add predictions to the dataframe
    predictions = model.predict(featureMatrix)
    titanic_dataset['Predictions'] = predictions
    titanic_dataset.to_excel('./output/predictions.xlsx')
