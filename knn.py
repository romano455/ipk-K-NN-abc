#Importing Libraries
import pandas as pd
import numpy as np
import math
import operator

class KnnModel(object):
    
    def __init__(self, k):
        self.k=k
        self.datatrain=None
        self.datatest=None

    def import_data_train(self, datatrain):
        self.datatrain= pd.read_csv(datatrain)
    
    def import_data_test(self,datatest):
        self.datatest= datatest
    
    #Calculating Euclidean Distance
    def euclideandistance(self,data1,data2,length):
        distance=0
        for x in range(length):
            distance+=np.square(data1[x]-data2[x])
        return np.sqrt(distance)
    
    #Defining KNN Model
    def knn(self):
        
        distances=list()
        length=len(self.datatest)
    
        #Calculating Euclidean Distance between Each Row of Data Training and Data Test
        for i in range (len(self.datatrain)):
            dist=self.euclideandistance(self.datatrain.iloc[i], self.datatest, length)
            distances.append(dist)
    
        for i in range (len(distances)):
            self.datatrain['Distances']=distances
        
        #Sorting Based on Distance
        self.datatrain=self.datatrain.sort_values(by=['Distances'])
    
        #Choose k-number of Rows
        data_final=self.datatrain.head(self.k)
        return data_final
    
    #Search for The Result
    def predict(self):
        
        result=self.knn()
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