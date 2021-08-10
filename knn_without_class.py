#Importing Libraries
import pandas as pd
import numpy as np
import math
import operator

#Importing Data
data=pd.read_csv("iris.csv")

#Calculating Euclidean Distance
def euclideandistance(data1,data2,length):
    distance=0
    for x in range(length):
        distance+=np.square(data1[x]-data2[x])
    return np.sqrt(distance)

#Defining KNN Model
def knn(data_training, data_test, k):
    distances=list()
    length=len(data_test)
    
    #Calculating Euclidean Distance between Each Row of Data Training and Data Test
    for i in range (len(data_training)):
        dist=euclideandistance(data_training.iloc[i], data_test, length)
        distances.append(dist)
    
    for i in range (len(distances)):
        data_training['Distances']=distances
        
    #Sorting Based on Distance
    data_training=data_training.sort_values(by=['Distances'])
    
    #Choose k-number of Rows
    data_final=data_training.head(k)
    return data_final

#Input Data Test
datatest=[7.2, 3.6, 5.1, 2.5]

#Search for The Result
result=knn(data,datatest,3)
frequent=result['Name'].mode()

#If There Are More Than 1 Result
if len(frequent)>1:
    result2=result.groupby('Name')['Distances'].sum()
    maximum=result2.max()
    
    for x in frequent:
        if result2[x]<maximum:
            min=result2[x]
            maximum=min
            
    for i in range(len(result2)):
        if result2[i]==min:
            result_final=result2.index[i]
    
else:
    result_final=frequent[0]

#Final Result
print(result_final)