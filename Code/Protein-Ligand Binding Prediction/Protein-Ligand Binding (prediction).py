import pandas as pd
import numpy as np
import math, time
# from sklearn.preprocessing import MinMaxScaler


#variables
src = r"HybridLBP_(positive).csv"
inputData = r"test.csv"
dest = r"prediction.txt"
pFeatureNumber = 736
lFeatureNumber = 677
nearestNeighbor = 3

#thresholds (dummy initialization)
Ethreshold1_clus=	1936.75
Ethreshold1_dis=	2339.71
Mthreshold1_clus=	6048.96
Mthreshold1_dis=	7219.63
Ethreshold2_clus=	1213495.27
Ethreshold2_dis=	1296471.59
Mthreshold2_clus=	2509168.34
Mthreshold2_dis=	2697577.54


def getDistance1(instance, X_train):
    train = X_train
    n = train.shape[0]
    nearestNeighborLimit = nearestNeighbor
    if(nearestNeighbor > n):
        nearestNeighborLimit = n
    relatedLigand = np.zeros(shape=(n,2+lFeatureNumber)) #protein distance (2), ligand data
    for i in range(n):
        EdistanceP = math.sqrt(sum((instance[:pFeatureNumber] - train[i,:pFeatureNumber])**2)) # euclidean distance
        MdistanceP = sum( abs(instance[:pFeatureNumber] - train[i,:pFeatureNumber]) )          # manhattan distance
        relatedLigand [i,0] = EdistanceP
        relatedLigand [i,1] = MdistanceP
        relatedLigand[i,2:] = train[i,pFeatureNumber:(pFeatureNumber+lFeatureNumber)]            #change it for pClass

#for euclidean
    relatedLigand = relatedLigand[relatedLigand[:,0].argsort()]

    #euclidean cluster-mean
    clusterCenter = np.mean(relatedLigand[:nearestNeighborLimit,2:], axis=0)
    Edistance1_clus = math.sqrt(sum((instance[pFeatureNumber:(pFeatureNumber+lFeatureNumber)] - clusterCenter)**2))  #related L ~ given L

    #euclidean distance-mean
    distance = np.zeros(shape=(nearestNeighborLimit))
    for i in range(nearestNeighborLimit):
        distance[i] = math.sqrt( sum( (instance[pFeatureNumber:(pFeatureNumber+lFeatureNumber)] - relatedLigand[i,2:])**2 ) )
    Edistance1_dis = np.mean(distance) #related L ~ given L



#for manhattan
    relatedLigand = relatedLigand[relatedLigand[:,1].argsort()]

    #manhattan cluster-mean
    clusterCenter = np.mean(relatedLigand[:nearestNeighborLimit,2:], axis=0)
    Mdistance1_clus = sum( abs(instance[pFeatureNumber:(pFeatureNumber+lFeatureNumber)] - clusterCenter) )  #related L ~ given L

    #manhattan distance-mean
    distance = np.zeros(shape=(nearestNeighborLimit))
    for i in range(nearestNeighborLimit):
        distance[i] =  sum( abs(instance[pFeatureNumber:(pFeatureNumber+lFeatureNumber)] - relatedLigand[i,2:]) )
    Mdistance1_dis = np.mean(distance) #related L ~ given L

    return Edistance1_clus, Edistance1_dis, Mdistance1_clus, Mdistance1_dis


def getDistance2(instance, X_train):
    train = X_train
    n = train.shape[0]
    nearestNeighborLimit = nearestNeighbor
    if(nearestNeighbor > n):
        nearestNeighborLimit = n
    relatedProtein = np.zeros(shape=(n,2+pFeatureNumber)) #ligand distance (2), protein data
    for i in range(n):
        EdistanceL = math.sqrt(sum((instance[pFeatureNumber:(pFeatureNumber+lFeatureNumber)] - train[i,pFeatureNumber:(pFeatureNumber+lFeatureNumber)])**2)) # euclidean distance
        MdistanceL = sum( abs(instance[pFeatureNumber:(pFeatureNumber+lFeatureNumber)] - train[i,pFeatureNumber:(pFeatureNumber+lFeatureNumber)]) )          # manhattan distance
        relatedProtein [i,0] = EdistanceL
        relatedProtein [i,1] = MdistanceL
        relatedProtein[i,2:] = train[i,:pFeatureNumber]

#for euclidean
    relatedProtein = relatedProtein[relatedProtein[:,0].argsort()]

    #euclidean cluster-mean
    clusterCenter = np.mean(relatedProtein[:nearestNeighborLimit,2:], axis=0)
    Edistance2_clus = math.sqrt(sum((instance[:pFeatureNumber] - clusterCenter)**2))  #related L ~ given L

    #euclidean distance-mean
    distance = np.zeros(shape=(nearestNeighborLimit))
    for i in range(nearestNeighborLimit):
        distance[i] = math.sqrt( sum( (instance[:pFeatureNumber] - relatedProtein[i,2:])**2 ) )
    Edistance2_dis = np.mean(distance) #related P ~ given P


#for manhattan
    relatedProtein = relatedProtein[relatedProtein[:,1].argsort()]

    #manhattan cluster-mean
    clusterCenter = np.mean(relatedProtein[:nearestNeighborLimit,2:], axis=0)
    Mdistance2_clus = sum( abs(instance[:pFeatureNumber] - clusterCenter) )  #related L ~ given L

    #manhattan distance-mean
    distance = np.zeros(shape=(nearestNeighborLimit))
    for i in range(nearestNeighborLimit):
        distance[i] =  sum( abs(instance[:pFeatureNumber] - relatedProtein[i,2:]) )
    Mdistance2_dis = np.mean(distance) #related L ~ given L

    return Edistance2_clus, Edistance2_dis, Mdistance2_clus, Mdistance2_dis


def majority(pred):
    unique,pos = np.unique(pred,return_inverse=True) #Finds all unique elements and their positions
    counts = np.bincount(pos)                        #Count the number of each unique element
    maxPos = counts.argmax()                         #Finds the positions of the maximum count
    return unique[maxPos]



def predict(X_test, X_train, Y_train):
    global Ethreshold1_clus, Ethreshold1_dis, Mthreshold1_clus, Mthreshold1_dis, Ethreshold2_clus, Ethreshold2_dis, Mthreshold2_clus, Mthreshold2_dis
    n = X_test.shape[0]
    prediction = np.zeros((n,21))
    
    print("  Predicting.......")
    for i in range(n):
        step = 1
        if (n > 10):
            step = math.floor(n/10)
        if((i+1)%step==0):       # print after every 10% iterations
            print("\ttest", i+1, " (", n, ")")
        instance = X_test[i,:]
        

    #distance1 = related ligand ~ given ligand
        Edistance1_clus, Edistance1_dis, Mdistance1_clus, Mdistance1_dis = getDistance1(instance,X_train[np.where(Y_train==1)])

        #accurate1
        if Edistance1_clus <= Ethreshold1_clus:
            prediction[i,0] = 1

        if Edistance1_dis <= Ethreshold1_dis:
            prediction[i,1] = 1

        if Mdistance1_clus <= Mthreshold1_clus:
            prediction[i,2] = 1

        if Mdistance1_dis <= Mthreshold1_dis:
            prediction[i,3] = 1

            
    #distance2 = related protein ~ given protein
        Edistance2_clus, Edistance2_dis, Mdistance2_clus, Mdistance2_dis = getDistance2(instance,X_train[np.where(Y_train==1)])

        #accurate2
        if Edistance2_clus <= Ethreshold2_clus:
            prediction[i,4] = 1

        if Edistance2_dis <= Ethreshold2_dis:
            prediction[i,5] = 1
        
        if Mdistance2_clus <= Mthreshold2_clus:
            prediction[i,6] = 1
        
        if Mdistance2_dis <= Mthreshold2_dis:
            prediction[i,7] = 1
        
        

    #voting for each distance
        
        prediction[i,8] = majority(prediction[i,[0,4]])
        
        prediction[i,9] = majority(prediction[i,[1,5]])

        prediction[i,10] = majority(prediction[i,[2,6]])

        prediction[i,11] = majority(prediction[i,[3,7]])
        
        

    #voting for distance 1 Eucledian
        prediction[i,12] = majority(prediction[i,[0,1]])
                
    #voting for distance 1 Manhattan
        prediction[i,13] = majority(prediction[i,[2,3]])

    #voting for distance 2 Eucledian
        prediction[i,14] = majority(prediction[i,[4,5]])
        
    #voting for distance 2 Manhattan
        prediction[i,15] = majority(prediction[i,[6,7]])

    #voting for Eucledian
        prediction[i,16] = majority(prediction[i,[0,1,4,5]])

    #voting for Manhattan
        prediction[i,17] = majority(prediction[i,[2,3,6,7]])

        
        
    #voting for distance 1 all
        prediction[i,18] = majority(prediction[i,[0,1,2,3]])
    
    #voting for distance 2 all
        prediction[i,19] = majority(prediction[i,[4,5,6,7]])
                
    #voting for all
        prediction[i,20] = majority(prediction[i,[0,1,2,3,4,5,6,7]])
        
    return prediction
        


def main():
    data = pd.read_csv(src)
    data['Bond'] = data['Bond'].map({'yes': 1, 'no': 0})
    
    yesData = data.loc[data['Bond']== 1].values
    
    X_train = yesData[:,:-1]
    Y_train = yesData[:,-1]
    
    X_test = pd.read_csv(inputData).values
    
    #data Preprocessing
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     X = scaler.fit_transform(X)
    
    #information
    info = "\nNearest Neighbor: "+ str(nearestNeighbor) + "\n"
    info += "Protein feature: " + str(pFeatureNumber) + "\n"
    info += "Ligand feature: " + str(lFeatureNumber) + "\n"
    print(info)

    start = time.time()
    
    #Prediction
    prediction = predict(X_test, X_train, Y_train)
    print("\nOutput:")
    output = ""
    for i in range(X_test.shape[0]):
        if(prediction[i,20] == 1):
            print((i+1),"\t-->\tyes")
            output += str(i+1)+"\t-->\tyes\n"
        else:
            print((i+1),"\t-->\tno")
            output += str(i+1)+"\t-->\tno\n"
    
    #Save output
    f = open(dest, "w")
    f.write(output)
    f.close()

    stop=time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(stop-start))
    print("\nTotal time: ", t)

    print("***Check \"" + str(dest) + "\" file for prediction.......\n")

if __name__ == '__main__':
    main()

