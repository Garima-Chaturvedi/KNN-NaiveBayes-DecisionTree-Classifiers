import numpy as np

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

f=open('project3_dataset2.txt', 'r')
data=f.readlines();

#bincount 3 for dataset 1
#bincount 7 for dataset 2
bincount=3

kfold=10

samplecount=data.__len__()
attricount=len(data[0].split('\t'))-1

#print 'size', samplecount, attricount

samples=np.zeros((samplecount, attricount), dtype=np.float32)
classes=np.zeros((samplecount,1), dtype=np.int)

#nominal dictionary is to map string attributes to integer values.
nominal={}
nominalcount=0
nominalindex={}

#adding data into samples and classes
for i in range (0, samplecount):
    attri=[]
    attri=data[i].split('\t')
    attri=np.array(attri)
    for j in range (0, attricount):
        if (is_number(attri[j])):
            samples[i][j]=attri[j]
        else:
            if (attri[j] in nominal):
                samples[i][j]=nominal[attri[j]]
            else:
                nominal[attri[j]]=nominalcount
                samples[i][j]=nominal[attri[j]]
                nominalcount=nominalcount+1
                if (j not in nominalindex):
                    nominalindex[j]=j
    classes[i]=attri[attricount]

#checking if each column has continous values using nominalindex (if in nominal index then not continous)
#processing the continous values into bins
for j in range (0, attricount):
    if (j not in nominalindex):
        minimum= np.amin(samples[:,j:j+1], axis=None, out=None, keepdims=False)
        maximum= np.amax(samples[:,j:j+1], axis=None, out=None, keepdims=False)
        step= (maximum-minimum)/bincount
        for i in range (0, samplecount):
            binvalue=minimum+step
            for k in range (0,bincount):
                if (samples[i][j]<=binvalue):
                    samples[i][j]=binvalue
                    break
                elif (samples[i][j]>binvalue and k==bincount-1):
                    samples[i][j]=maximum
                else:
                    binvalue=binvalue+step


testcount=int(samplecount/kfold)
samplecount=samplecount-testcount
minc=0
maxc=minc+testcount

accuracy=0
precision=0
recall=0
fmeasure=0

for i in range (0,kfold):

    TP=0
    TN=0
    FP=0
    FN=0


    testsample=samples[minc:maxc,:]
    testclasses=classes[minc:maxc,:]

    trainsamples=samples
    trainclasses=classes

    trainsamples=np.delete(trainsamples, np.s_[minc:maxc], 0)
    trainclasses=np.delete(trainclasses, np.s_[minc:maxc], 0)

    minc=maxc
    maxc=minc+testcount

    #calculating class count for each class.
    PH={}

    for i in range (0, samplecount):
        if(trainclasses[i][0] in PH):
            PH[trainclasses[i][0]]=PH[trainclasses[i][0]]+1
        else:
            PH[trainclasses[i][0]]=1

    #calculating attribute prior probablity for each column and each attribute type
    PX=[]

    for j in range (0, attricount):
        temp={}
        for i in range(0, samplecount):
            if(trainsamples[i][j] in temp):
                temp[trainsamples[i][j]]=temp[trainsamples[i][j]]+1
            else:
                temp[trainsamples[i][j]]=1
        for key in temp:
            temp[key]=temp[key]/float(samplecount)
        PX.append(temp)

    #calculating descriptor posterior for each attribute for each class
    PXH0=[]
    PXH1=[]
    for j in range (0, attricount):
        temp0={}
        temp1={}
        for i in range(0, samplecount):
            if (trainclasses[i][0]==0):
                if(trainsamples[i][j] in temp0):
                    temp0[trainsamples[i][j]]=temp0[trainsamples[i][j]]+1
                else:
                    temp0[trainsamples[i][j]]=1
            if (trainclasses[i][0]==1):
                if(trainsamples[i][j] in temp1):
                    temp1[trainsamples[i][j]]=temp1[trainsamples[i][j]]+1
                else:
                    temp1[trainsamples[i][j]]=1
        for key in temp0:
            temp0[key]=temp0[key]/float(PH[0])
        for key in temp1:
            temp1[key]=temp1[key]/float(PH[1])
        PXH0.append(temp0)
        PXH1.append(temp1)

    counttest=0


    for i in range (0,testcount):
        prob0=0
        prob1=0
        PXi=-1
        PXHi0=-1
        PXHi1=-1
        counttest+=1
        for j in range(0,attricount):
            temp=PX[j]
            try:
                if (PXi==-1):
                    PXi=temp[testsample[i][j]]
                else:
                    PXi=PXi*temp[testsample[i][j]]
            except KeyError:
                i
            temp0=PXH0[j]
            temp1=PXH1[j]
            if (PXHi0==-1 and (testsample[i][j] in temp0)):
                PXHi0=temp0[testsample[i][j]]
            elif (testsample[i][j] in temp0):
                PXHi0=PXHi0*temp0[testsample[i][j]]

            if (PXHi1==-1 and (testsample[i][j] in temp1)):
                PXHi1=temp1[testsample[i][j]]
            elif (testsample[i][j] in temp1):
                PXHi1=PXHi1*temp1[testsample[i][j]]
        prob0=((PH[0]/float(samplecount))*PXHi0)/PXi
        prob1=((PH[1]/float(samplecount))*PXHi1)/PXi
        if (prob0>prob1):
            if (testclasses[i][0]==0):
                TN+=1
            else:
                FN+=1
        else:
            if (testclasses[i][0]==1):
                TP+=1
            else:
                FP+=1


    accuracy=accuracy+(float(TP+TN)/float(TP+FN+FP+TN))
    precision=precision+(float(TP)/float(TP+FP))
    recall=recall+(float(TP)/float(TP+FN))
    fmeasure=fmeasure+((2*recall*precision)/(recall+precision))
    #print 'Accuracy',accuracy
    #print 'Precision',precision
    #print 'Recall',recall
    #print 'F Measure',fmeasure
    #print '-----------------------------'


#print float(avgmatchcount*100)/float(kfold*testcount)
print 'Avg Accuracy:', accuracy/kfold*100, '%'
print 'Avg Precision:',precision/kfold
print 'Avg Recall:',recall/kfold
print 'Avg F Measure:',fmeasure/kfold
