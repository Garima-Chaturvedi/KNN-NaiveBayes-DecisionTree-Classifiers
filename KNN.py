import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import operator
from sklearn import preprocessing

#function to check for numeric values
def is_num(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#open given dataset for reading the features           
file=open('project3_dataset1.txt' ,'r')
features=file.readlines()

kfold=10

k=4

#splitting on tab to get the features without the labels
samplecount=features.__len__()
attributes=len(features[0].split('\t'))-1

#print 'samples:', samplecount,'Attributes;', attributes

#creating matrix for storing the features
samples=np.zeros((samplecount, attributes),dtype=np.float64 )
classes=np.zeros((samplecount,1))

nominalval={}
nominalCount = 0

#insertion of data
for i in range (0,samplecount):
    attri=[]
    attri= features[i].split('\t')
    attri=np.array(attri)
    for j in range (0, attributes):
        if(is_num(attri[j])):
            samples[i][j]=attri[j]
        else:
            if(attri[j] in nominalval):
                samples[i][j]=nominalval[attri[j]]
            else:
                nominalval[attri[j]]=nominalCount
                samples[i][j]=nominalval[attri[j]]
                nominalCount=nominalCount+1
        classes[i]=attri[attributes]

min_max_scaler = preprocessing.MinMaxScaler()
scaled_sample = min_max_scaler.fit_transform(samples)
#print 'scaled samples',scaled_sample

#implementing K fold Cross Validation
testcount=int(samplecount/kfold) 
samplecount=samplecount-testcount
mincount = 0
maxcount= mincount+testcount

accuracy=0
precision=0
recall=0
fmeasure=0
for x in range(0,kfold):
    
    test_minmax=scaled_sample[mincount:maxcount,:]
    testclasses=classes[mincount:maxcount,:]
    
    train_minmax=np.delete(scaled_sample,np.s_[mincount:maxcount],0)
    trainclasses=np.delete(classes,np.s_[mincount:maxcount],0)

    mincount=maxcount
    maxcount=mincount+testcount

    #calculate distances for each test instance with every train instance
    distance = euclidean_distances(train_minmax,test_minmax,)
    distance=distance.transpose(1,0)
    
    
    def nearest(k):
        #print k
        matchcount=0
        TP=0
        FN=0
        FP=0
        TN=0
        global accuracy,precision,recall,fmeasure
        
        for i in  range(len(test_minmax)):
            enumerated=[]
            count1=0
            count2=0
            newK=k
            #get the k least distances from the distance matrix
            enumerated=(zip(*sorted(enumerate(distance[i]), key=operator.itemgetter(1)))[0][0:newK])
            tie=True
            
            #assigning labels based on majority voting
            while(tie):
                for j in range(0,newK):
                    if(trainclasses[enumerated[j]] == 1):
                        count1=count1+1
                    else:
                        count2=count2+1
                if (count1>count2):
                    #print '1',testclasses[i]
                    if(testclasses[i]==1):
                        matchcount+=1
                        TP+=1
                    else:
                        FP+=1
                    tie=False
                elif(count2>count1):
                    #print '0',testclasses[i]
                    if(testclasses[i]==0):
                        matchcount+=1
                        TN+=1
                    else:
                        FN+=1
                    tie=False
                else:
                    newK-=1
        #calucating the cost sensitive measures for each iteration in K fold
        accuracy=accuracy+(float(TP+TN)/float(TP+FN+FP+TN))
        
        precision=precision+(float(TP)/float(TP+FP))
        
        recall=recall+(float(TP)/float(TP+FN))
        
        fmeasure=fmeasure+((2*recall*precision)/(recall+precision))
        
        print 'Accuracy',accuracy
        print 'Precision',precision
        print 'Recall',recall
        print 'F Measure',fmeasure
        print'-----------------------------------------------------'
        return  
        
    nearest(k)

#calculate the average accuracy    
print 'Average Accuracy',(accuracy/kfold)*100,'%'
print 'Average Precision',precision/kfold
print 'Average Recall',recall/kfold
print 'Average F Measure',fmeasure/kfold
