global data
import random
import numbers
import math
from math import *
import numpy as np
import operator
from sklearn import preprocessing
class node:
    def __init__(self, left_child , right_child , decision_element , result_set , value):

        self.left_child = left_child
        self.right_child = right_child

        self.decision_element = decision_element
        self.result_set = result_set
        self.value = value



def result_map(sub_tree_data):
    #print 'result_map ' , sub_tree_data
    result = {}
    for row in sub_tree_data:
            label = row[len(row)-1]
            if result.__contains__(label):
                result[label] += 1
            else :
                result[label] = 1
    return result



def divide_tree(sub_tree_data,dec_element):

    # print 'divide_tree' , dec_element[0] , dec_element[1]
    value = sub_tree_data[int(dec_element[0])][int(dec_element[1])]
    #print value
    col = dec_element[1]
    left_branch = []
    right_branch = []
    num = False
    if isinstance(value,numbers.Real): num = True
    print num
    for row in sub_tree_data:
        if num:
            if row[col] < value:
                left_branch.append(row)
            else :
                right_branch.append(row)

        else:
            if row[col]==value:
                #print 'match'
                left_branch.append(row)
            else:
                #print 'mismatch'
                right_branch.append(row)
        # if row[col] > value :
        #     if num:
        #         right_branch.append(row)
        # elif row[col] == value:
        #     if num : right_branch.append(row)
        #     else : left_branch.append(row)
        # else :
        #     if num:
        #         left_branch.append(row)

    #print len(left_branch) , len(right_branch)
    return left_branch , right_branch

def gini_index(sub_tree):
    print len(sub_tree)


def entropy(tree_data):
    #print 'entropy'
    entropy = 0.0
    map = result_map(tree_data)
    for k in map.keys():
        p = (map[k]*1.0)/len(tree_data)
        entropy = entropy - p*math.log(p,2)
    return entropy


def split_tree(tree,row, col):
    value = tree[row][col]
    left = []
    right = []
    num = False
    if isinstance(value,numbers.Real): num = True
    for r in range(len(tree)):
        if num:
            if tree[r][col] < value :
                left.append(tree[r])
            else : right.append(tree[r])
        else:
            if tree[r][col] == value:
                left.append(tree[r])
            else:
                right.append(tree[r])

    return left , right


def get_optimal_split_element(parent_tree_data):
    entropy_parent = entropy(parent_tree_data)
    size_parent = len(parent_tree_data)*1.0
    #print 'inside get optimal .. ' , size_parent
    max_gain = 0.0
    gain = 0.0
    opt_element = []
    for row in range(len(parent_tree_data)):
        for i in range(len(parent_tree_data[row])-1):
            dec_el = []
            dec_el.append(row)
            dec_el.append(i)
            #print 'calling divide - ' , dec_el
            left , right = split_tree(parent_tree_data,row,i)
            #print 'optimal left , r ' , len(left) , len(right)
            if len(left)  >= 1 and len(right) >= 1 :
                gain = entropy_parent - (len(left)/size_parent)*entropy(left) - (len(right)/size_parent)*entropy(right)
                if gain > max_gain :
                    max_gain = gain
                    opt_element = [row,i]

    return opt_element , max_gain


def build_decision_tree(tree):
    size_tree = len(tree)
    if size_tree == 0 :
        return node()

    #print 'inside build ..'

    opt_element , gain = get_optimal_split_element(parent_tree_data=tree)

    print gain , opt_element
    if len(opt_element) > 0 or gain > 0:
        value = tree[opt_element[0]][opt_element[1]]
        left , right = split_tree(tree=tree,row = opt_element[0],col = opt_element[1])
        #print len(left) , len(right)
        left_tree =  build_decision_tree(left)
        right_tree = build_decision_tree(right)
        return node(left_child=left_tree,right_child=right_tree,decision_element=opt_element,result_set=None, value=value)

    else: return node(result_set=result_map(tree), left_child=None, right_child=None,decision_element=None,value=None)



def print_tree(tree):
    if tree.result_set == None :
        print "Root - " , tree.value , " || Left - " , tree.left_child.value , " || Right - " , tree.right_child.value
        print_tree(tree.left_child)
        print_tree(tree.right_child)



def classify(row,tree):
    label = row[len(row)-1]
    branch = None

    if tree.result_set != None :
        predicted = tree.result_set.keys()[0]
        matches = False
        if label==predicted:
            matches = True
            #print 'match'
        print " || Label = " , label , 'Predicted =  ' , predicted , '| Match -  ' , matches
        return predicted , matches
    else:
        col = tree.decision_element[1]
        tree_value = tree.value
        data_val = row[col]
        num = False
        if isinstance(data_val,numbers.Real): num = True
        if num :
            if data_val <= tree_value:
                branch = tree.left_child
            else :
                branch = tree.right_child
        else:
            if data_val == tree_value:
                #print 'match'
                branch = tree.left_child
            else : branch = tree.right_child

        return classify(row,branch)


def format_data(filename):
    file = open(filename ,'r')
    lines = file.readlines()
    data = []
    for line in lines:
        line = line.replace("\n","")
        elements = line.split("\t")
        data.append(elements)
    return data

def weights_int(data_samples):
    samplesize=data_samples.__len__()
    weights=np.zeros((samplesize,1), dtype=np.float32)

    for i in range (0, samplesize):
        weights[i]=1.0/float(samplesize)
    print 'weights', weights, 'weights end'
    return weights

def boosting_trees(data_samples,test_samples, M,mdtrees):
    global attricount
    weights=weights_int(data_samples)
    print weights
    chosensize=50
    samplesize=data_samples.__len__()
    maxalpha=0
    minalpha=100
    chosensamples=np.zeros((chosensize,attricount), dtype=np.float32)
    matchcount=0
    #sampleoutput=np.zeros((samplesize, 1))

    for  i in range (1,M):
        for j in range (1, chosensize):
            randomIndex=-1
            randomnum=random.random()*np.sum(weights)
            for k in range (0, samplesize):
                randomnum-=weights[k]
                print weights[k]
                if (randomnum<=0.0):
                    randomIndex=k
                    break

            chosensamples[j]=data_samples[randomIndex]
        rootnode=build_decision_tree(chosensamples)
        error=np.zeros((1,1), dtype=np.float32)

        sampleindex=0
        listmismatch=[]
        for t in testsample:
            #print t
            output,match=classify(t , rootnode)
            #sampleoutput[sampleindex][0]=output
            print match
            if (match==False):
                #print '*******************ERROR*************************'
                #print error
                error[0][0]= error[0][0] + weights[sampleindex]
                #print error, weights[sampleindex]
                listmismatch.append(sampleindex)
            sampleindex+=1
        error[0][0]=error[0][0]/np.sum(weights)
        print 'error', error

        alpha=1
        if (error!=0):
            alpha=0.5*math.log((1-error)/error)
            print alpha, 'alpha'
            if (alpha>maxalpha):
                maxalpha=alpha
            elif(alpha<=minalpha):
                minalpha=alpha

        for l in range (0, len(listmismatch)):
            weights[listmismatch[l]]=weights[listmismatch[l]]*exp(alpha)
        min_max_scaler=preprocessing.MinMaxScaler()
        weights=min_max_scaler.fit_transform(weights)
        mdtrees[i]=[rootnode,alpha]

    avgalpha=float((maxalpha+minalpha))/float(2)

    for t in test_samples:
        alphatest=float(0)
        for m in range (1,M):
            testvariable=mdtrees[i]
            output,match=classify(t , testvariable[0])
            if (match==True):
                alphatest=alphatest+testvariable[1]
        if (alphatest>avgalpha):
                matchcount+=1
    print 'matchcount', matchcount




def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

f=open('project3_dataset2.txt', 'r')
data=f.readlines();

samplecount=data.__len__()
attricount=len(data[0].split('\t'))

print 'size', samplecount, attricount

samples=np.zeros((samplecount, attricount), dtype=np.float32)
classes=np.zeros((samplecount,1), dtype=np.int)

kfold=10

nominal={}
nominalcount=0
nominalindex={}

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


testcount=int(samplecount/kfold)
samplecount=samplecount-testcount
minc=0
maxc=minc+testcount

accuracy=0
precision=0
recall=0
fmeasure=0

mdtrees={}
M=15

for i in range (0,1):

    TP=0
    TN=0
    FP=0
    FN=0


    testsample=samples[minc:maxc,:]

    trainsamples=samples

    trainsamples=np.delete(trainsamples, np.s_[minc:maxc], 0)

    minc=maxc
    maxc=minc+testcount


    dtree = build_decision_tree(trainsamples)

    for t in testsample:
        print t , classify(t , dtree)
    print '################################', kfold, '#####################################'

    boosting_trees(trainsamples, testsample, M, mdtrees)
