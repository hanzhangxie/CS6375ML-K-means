from pandas.io.json import json_normalize
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt


def load_json_to_data(TweetsDataFile):
    data = []
    with open(TweetsDataFile) as f:
        for line in f:
            data.append(json.loads(line))

    data = pd.DataFrame(data)
    data = data.loc[:, ['id']]
    return data.values
 
def calculateDistance(vecA, vecB):

    lx = list(map(int, vecA[0]))
    ly = list(map(int, vecB[0]))

    inter = 0
    union = len(lx)
    for index in range(len(lx)):
        if lx[index]==ly[index]:
            inter+=1
        else:
            union+=1
    return 1-(inter/union)
 
 
def minDistance(data_get, centroidList):
 
    clusterDict = dict() 
    for element in data_get:
        vecA = np.array(element)
        flag = 0
        minDis = float("inf")
        for i in range(len(centroidList)):
            vecB = np.array(centroidList[i])
            distance = calculateDistance(vecA, vecB)
            if distance < minDis:
                minDis = distance
                flag = i 
 
        if flag not in clusterDict.keys(): 
            clusterDict[flag] = list()
        clusterDict[flag].append(element) 
    return clusterDict 
 
 
def getCentroids(clusterDict):
    
    centroidList = list()
    for key in clusterDict.keys():
        map_centroid = np.zeros(18)
        x = np.array(clusterDict[key])
        counter = np.zeros((18, 10))
        for i in range (len(clusterDict[key])):
            item = list(map(int,clusterDict[key][i][0]))
            for j in range(len(item)):
                counter[j][item[j]]=counter[j][item[j]]+1

        mean_centroid = np.argmax(counter, axis=1)
        sum = 0
        for item in mean_centroid:
            sum = 10*sum+item
        centroidList.append(str(sum))
    centroidList = np.array(centroidList)
    row = centroidList.shape[0]
    centroidList = centroidList.reshape((row,1))
    return np.array(centroidList).tolist()
 
 
def calculate_Var(clusterDict, centroidList):
 
    sum = 0.0
    for key in clusterDict.keys():
        vecA = np.array(centroidList[key])
        distance = 0.0
        for item in clusterDict[key]:
            vecB = np.array(item)
            distance += calculateDistance(vecA, vecB)*calculateDistance(vecA, vecB)
        sum += distance
 
    return sum
 
def tweets_k_means(numberOfClusters,initialSeedsFile,TweetsDataFile,outputFile):

    data = load_json_to_data(TweetsDataFile)
    data = data.astype(str)
    data = data.tolist()
    centroidList = np.loadtxt(initialSeedsFile,dtype='str')
    centroidList = np.array(centroidList).reshape((numberOfClusters,1))
    centroidList = centroidList.tolist()
    clusterDict = minDistance(data, centroidList) 
    newVar = calculate_Var(clusterDict, centroidList) 
    oldVar = -0.0001 
    
    print('******************************** The first iternation **********************')
    for key in clusterDict.keys():
        print('centroid: ', centroidList[key])
        print('cluster: ',clusterDict[key])
    print('SSE: ', newVar)
     
    k = 2
    while abs(newVar - oldVar) >= 0.00001: 
        centroidList = getCentroids(clusterDict)                                                     
        clusterDict = minDistance(data, centroidList) 
        oldVar = newVar
        newVar = calculate_Var(clusterDict, centroidList)
     
        print('************************** The %dth iternation ************************' % k)
        result = pd.DataFrame()
        for key in clusterDict.keys():
            print('centroid: ', centroidList[key])
            print('cluster: ', clusterDict[key])
            cluster = pd.DataFrame(clusterDict[key]).T
            result = pd.concat([result,cluster])
        result['clu_index'] = np.arange(numberOfClusters)
        clu_index = result['clu_index']
        result.drop(labels=['clu_index'], axis=1,inplace = True)
        result.insert(0, 'clu_index', clu_index)
        # print(result)
        print('SSE is : ', newVar)
        k += 1
    result.to_csv(outputFile, header=None, index=False, sep=" ")
    fh = open(outputFile, 'a')
    fh.write("The SSE is "+ str(newVar))
    fh.close()

if __name__ == '__main__':
    tweets_k_means(25,'InitialSeeds.txt','cs6375-003.json','output.txt')

