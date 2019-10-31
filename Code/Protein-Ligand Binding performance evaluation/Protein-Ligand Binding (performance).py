import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
import math, time, sys


#variables

#for Hybrid LBP
#src = r"HybridLBP_(random).csv"
src = r"HybridLBP_(cluster).csv"
pFeatureNumber = 736
lFeatureNumber = 677

#for CoMOG & PHOG
# =============================================================================
# src = r"CoMOG_PHOG_(random).csv"
# pFeatureNumber = 1021
# lFeatureNumber = 1020
# =============================================================================

model = r"model.txt"
dest = r"dump.txt"
testSplitPercentage = 30       #test data percentage
nearestNeighbor = 3
crossFold = 10

#thresholds (dummy initialization)
Ethreshold1_clus=	1639.720491
Ethreshold1_dis=	2050.569576
Mthreshold1_clus=	5086.145056
Mthreshold1_dis=	6276.431977
Ethreshold2_clus=	1035333.856
Ethreshold2_dis=	1134422.678
Mthreshold2_clus=	2057275.147
Mthreshold2_dis=	2269486.755

dump = None


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


def train(X_train, Y_train):
    global Ethreshold1_clus, Ethreshold1_dis, Mthreshold1_clus, Mthreshold1_dis, Ethreshold2_clus, Ethreshold2_dis, Mthreshold2_clus, Mthreshold2_dis
    n = X_train.shape[0]

#distance1
    minEdistance1_clus = math.inf
    minEdistance1_dis = math.inf
    minMdistance1_clus = math.inf
    minMdistance1_dis = math.inf

    maxEdistance1_clus = 0
    maxEdistance1_dis = 0
    maxMdistance1_clus = 0
    maxMdistance1_dis = 0

    avgEdistance1_clus = 0
    avgEdistance1_dis = 0
    avgMdistance1_clus = 0
    avgMdistance1_dis = 0

#distance2
    minEdistance2_clus = math.inf
    minEdistance2_dis = math.inf
    minMdistance2_clus = math.inf
    minMdistance2_dis = math.inf

    maxEdistance2_clus = 0
    maxEdistance2_dis = 0
    maxMdistance2_clus = 0
    maxMdistance2_dis = 0

    avgEdistance2_clus = 0
    avgEdistance2_dis = 0
    avgMdistance2_clus = 0
    avgMdistance2_dis = 0


    print("  Training.......")
    for i in range(n):
        step = 1
        if (n > 10):
            step = math.floor(n/10)
        if((i+1)%step==0):       # print after every 10% iterations
            print("\ttrain", i+1, " (", n, ")")
        instance = X_train[i,:]

    #distance1 = related ligand ~ given ligand
        if(Y_train[i]==0):
            Edistance1_clus, Edistance1_dis, Mdistance1_clus, Mdistance1_dis = getDistance1(instance,X_train[np.where(Y_train==1)])
        else:
            Edistance1_clus, Edistance1_dis, Mdistance1_clus, Mdistance1_dis = getDistance1(instance,X_train[np.where(Y_train==0)])

        #min1
        if Edistance1_clus < minEdistance1_clus:
            minEdistance1_clus = Edistance1_clus
        if Edistance1_dis < minEdistance1_dis:
            minEdistance1_dis = Edistance1_dis
        if Mdistance1_clus < minMdistance1_clus:
            minMdistance1_clus = Mdistance1_clus
        if Mdistance1_dis < minMdistance1_dis:
            minMdistance1_dis = Mdistance1_dis

        #max1
        if Edistance1_clus > maxEdistance1_clus:
            maxEdistance1_clus = Edistance1_clus
        if Edistance1_dis > maxEdistance1_dis:
            maxEdistance1_dis = Edistance1_dis
        if Mdistance1_clus > maxMdistance1_clus:
            maxMdistance1_clus = Mdistance1_clus
        if Mdistance1_dis > maxMdistance1_dis:
            maxMdistance1_dis = Mdistance1_dis

        #avg1
        avgEdistance1_clus += Edistance1_clus/n
        avgEdistance1_dis += Edistance1_dis/n
        avgMdistance1_clus += Mdistance1_clus/n
        avgMdistance1_dis += Mdistance1_dis/n


    #distance2 = related protein ~ given protein
        if(Y_train[i]==0):
            Edistance2_clus, Edistance2_dis, Mdistance2_clus, Mdistance2_dis = getDistance2(instance,X_train[np.where(Y_train==1)])
        else:
            Edistance2_clus, Edistance2_dis, Mdistance2_clus, Mdistance2_dis = getDistance2(instance,X_train[np.where(Y_train==0)])

        #min2
        if Edistance2_clus < minEdistance2_clus:
            minEdistance2_clus = Edistance2_clus
        if Edistance2_dis < minEdistance2_dis:
            minEdistance2_dis = Edistance2_dis
        if Mdistance2_clus < minMdistance2_clus:
            minMdistance2_clus = Mdistance2_clus
        if Mdistance2_dis < minMdistance2_dis:
            minMdistance2_dis = Mdistance2_dis

        #max2
        if Edistance2_clus > maxEdistance2_clus:
            maxEdistance2_clus = Edistance2_clus
        if Edistance2_dis > maxEdistance2_dis:
            maxEdistance2_dis = Edistance2_dis
        if Mdistance2_clus > maxMdistance2_clus:
            maxMdistance2_clus = Mdistance2_clus
        if Mdistance2_dis > maxMdistance2_dis:
            maxMdistance2_dis = Mdistance2_dis

        #avg2
        avgEdistance2_clus += Edistance2_clus/n
        avgEdistance2_dis += Edistance2_dis/n
        avgMdistance2_clus += Mdistance2_clus/n
        avgMdistance2_dis += Mdistance2_dis/n

    Ethreshold1_clus = avgEdistance1_clus
    Ethreshold1_dis = avgEdistance1_dis
    Mthreshold1_clus = avgMdistance1_clus
    Mthreshold1_dis = avgMdistance1_dis
    Ethreshold2_clus = avgEdistance2_clus
    Ethreshold2_dis = avgEdistance2_dis
    Mthreshold2_clus = avgMdistance2_clus
    Mthreshold2_dis = avgMdistance2_dis

# =============================================================================
#     Ethreshold1_clus = (avgEdistance1_clus + maxEdistance1_clus)/2
#     Ethreshold1_dis = (avgEdistance1_dis + maxEdistance1_dis)/2
#     Mthreshold1_clus = (avgMdistance1_clus + maxMdistance1_clus)/2
#     Mthreshold1_dis = (avgMdistance1_dis + maxMdistance1_dis)/2
#     Ethreshold2_clus = (avgEdistance2_clus + maxEdistance2_clus)/2
#     Ethreshold2_dis = (avgEdistance2_dis + maxEdistance2_dis)/2
#     Mthreshold2_clus = (avgMdistance2_clus + maxMdistance2_clus)/2
#     Mthreshold2_dis = (avgMdistance2_dis + maxMdistance2_dis)/2
# =============================================================================


def majority(pred):
    unique,pos = np.unique(pred,return_inverse=True) #Finds all unique elements and their positions
    counts = np.bincount(pos)                        #Count the number of each unique element
    maxPos = counts.argmax()                         #Finds the positions of the maximum count
    return unique[maxPos]



def test(X_test, X_train, Y_train):
    global Ethreshold1_clus, Ethreshold1_dis, Mthreshold1_clus, Mthreshold1_dis, Ethreshold2_clus, Ethreshold2_dis, Mthreshold2_clus, Mthreshold2_dis
    n = X_test.shape[0]
    prediction = np.zeros((n,21))
    
    print("\n  Testing.......")
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
        


def saveModel():
    dump = ""
    # Threshold
    dump += "Ethreshold1_clus = " + str(Ethreshold1_clus) + "\n"
    dump += "Ethreshold1_dis = " + str(Ethreshold1_dis) + "\n"
    dump += "Mthreshold1_clus = " + str(Mthreshold1_clus) + "\n"
    dump += "Mthreshold1_dis = " + str(Mthreshold1_dis) + "\n"
    dump += "Ethreshold2_clus = " + str(Ethreshold2_clus) + "\n"
    dump += "Ethreshold2_dis = " + str(Ethreshold2_dis) + "\n"
    dump += "Mthreshold2_clus = " + str(Mthreshold2_clus) + "\n"
    dump += "Mthreshold2_dis = " + str(Mthreshold2_dis) + "\n"

    #Output
    f = open(model, "w")
    f.write(dump)
    f.close()
    
    dump = ""

    

def performKFold(X, Y):
    global dump
    print("Cross Fold: " + str(crossFold) + "\n")
    scores = np.zeros((4,21))
    kf = KFold(n_splits=crossFold, random_state=42, shuffle=True)
    iteration = 1
    for train_index, test_index in kf.split(X):
        print("\nFold: " + str(iteration))
#         print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        #Training
        train(X_train,Y_train)

        #saving model
        saveModel()

        #Testing
        prediction = test(X_test,X_train,Y_train)
        
        #accuracy
        for i in range(21):
            scores[0,i] += accuracy_score(Y_test,prediction[:,i])/crossFold

        #sensitivity
        for i in range(21):
            scores[1,i] += recall_score(Y_test, prediction[:,i], average='binary', pos_label=1)/crossFold

        #specificity
        for i in range(21):
            tn, fp, fn, tp = confusion_matrix(Y_test, prediction[:,i], labels=[0,1]).ravel()
            specificity = tn / (tn+fp)
            if(np.isnan(specificity)):
                print("No negative data found")
                specificity = 0
            scores[2,i] += specificity/crossFold
            
        #F1 score
        for i in range(21):
            scores[3,i] += f1_score(Y_test, prediction[:,i], average='binary', pos_label=1)/crossFold
            
        iteration += 1
    
    dump = ""
    for i in range(4):
        for j in range(21):
            dump += str(scores[i,j]*100) + "%\n"


def performPercentageSplit(X, Y):
    global dump
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testSplitPercentage/100, random_state=42)
    
    info = "Train length: " + str(X_train.shape[0]) + " (" + str(100-testSplitPercentage) + "%)\n"
    info += "Test length: " + str(X_test.shape[0]) + " (" + str(testSplitPercentage) + "%)\n"
    print(info)
    
    #Training
    train(X_train,Y_train)
    
    #saving model
    saveModel()
    
    #Testing
    prediction = test(X_test,X_train,Y_train)
    
    dump = ""
    
    #accuracy
    for i in range(21):
        dump += str(accuracy_score(Y_test,prediction[:,i])*100) + "%\n"
      
    #sensitivity
    for i in range(21):
        dump += str(recall_score(Y_test, prediction[:,i], average='binary', pos_label=1)*100) + "%\n"
        
    #specificity
    for i in range(21):
        tn, fp, fn, tp = confusion_matrix(Y_test, prediction[:,i], labels=[0,1]).ravel()
        specificity = tn / (tn+fp)
        if(np.isnan(specificity)):
            print("No negative data found")
            specificity = 0
        dump += str(specificity*100) + "%\n"
        
    #F1 score
    for i in range(21):
        dump += str(f1_score(Y_test, prediction[:,i], average='binary', pos_label=1)*100) + "%\n"


def main():
    global dump
    dump = ""
    data = pd.read_csv(src)
    data['Bond'] = data['Bond'].map({'yes': 1, 'no': 0})
    
    yesData = data.loc[data['Bond']== 1].values
    
    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values
    
    #data Preprocessing
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     X = scaler.fit_transform(X)
    
    #information
    info = "\nNearest Neighbor: "+ str(nearestNeighbor) + "\n"
    info += "Protein feature: " + str(pFeatureNumber) + "\n"
    info += "Ligand feature: " + str(lFeatureNumber) + "\n"
    print(info)

    start = time.time()
    
    #cross fold
    performKFold(X,Y)   

    #without cross fold
#     performPercentageSplit(X,Y)
    
    # Nearest Neighbor
    dump += str(nearestNeighbor) + "\n"

    # Test Length
    dump += str(testSplitPercentage) + "\n"

    # Threshold
    dump += str(Ethreshold1_clus) + "\n"
    dump += str(Ethreshold1_dis) + "\n"
    dump += str(Mthreshold1_clus) + "\n"
    dump += str(Mthreshold1_dis) + "\n"
    dump += str(Ethreshold2_clus) + "\n"
    dump += str(Ethreshold2_dis) + "\n"
    dump += str(Mthreshold2_clus) + "\n"
    dump += str(Mthreshold2_dis) + "\n"

    #Output
    f = open(dest, "w")
    f.write(dump)
    f.close()

    stop=time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(stop-start))
    print("\nTime taken: " + t)
    print("\n***Copy contents of \"" + str(dest) + "\" file or below text in \"Performance.xlsx\" file***\n")
    
    print(dump)

if __name__ == '__main__':
    main()
