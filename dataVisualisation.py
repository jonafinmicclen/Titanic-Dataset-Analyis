import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#To disable sections of code
ANALYSE_MODE = True

#Load csv into a pandas dataframe
titanicCSV = pd.read_csv('./Titanic.csv')
titanicCSV = titanicCSV.dropna()

#Create new dataframe of only Survived and Non Survived passengers
titanicCSV_SURVIVED_True = titanicCSV[titanicCSV['Survived']==1]
titanicCSV_SURVIVED_False = titanicCSV[titanicCSV['Survived']==0]

#One hot encode binary data
titanicCSV = pd.get_dummies(titanicCSV, columns=['Sex'])
titanicCSV = pd.get_dummies(titanicCSV, columns=['Pclass'])



#Put analytical functions seperate so will not have to run every time
if ANALYSE_MODE:

    #Initial analyis of data using built in describe function
    print(titanicCSV.describe(include='all'))

    print('Phi Correlation (isFemale, Survived): ',pd.crosstab(titanicCSV['Sex_female'], titanicCSV['Survived']).apply(lambda r: r / r.sum(), axis=1).apply(lambda c: c / c.sum(), axis=0).iloc[1, 1])
    
    print("Correlation (survived, isFemale):", titanicCSV['Survived'].corr(titanicCSV['Sex_female']))

    print("Correlation (survived, SibSp):", titanicCSV['Survived'].corr(titanicCSV['SibSp']))

    #Find frequency of survival for age ranges, then divide by frequency of all ages to remove offset
    print(np.histogram(titanicCSV['Age'], bins = 8))
    ageCounts, binEdge = np.histogram(titanicCSV['Age'], bins = 8)
    ageCountsAlive, _ = np.histogram(titanicCSV_SURVIVED_False['Age'], bins = binEdge)
    plt.bar(np.linspace(0,80, 8), ageCountsAlive/ageCounts, width=np.diff(binEdge))
    plt.title('Age')
    plt.show()
    