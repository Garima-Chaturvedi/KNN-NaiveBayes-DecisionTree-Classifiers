
global data
import numbers
import math
import numpy as np
import operator
from random import randint
from random import shuffle
from copy import deepcopy

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
    #print num
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

    #print gain , opt_element
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
        #print "     " , tree.value
        print_tree(tree.left_child)
        print_tree(tree.right_child)


def prt(tree, dep=0):
    ret = ""
    if tree.result_set != None :
        return ret
    else:
        if tree.right_child != None:
            ret += prt(tree.right_child,dep+1)

        ret += "\n" + ("    "*dep) + tree.value

        if tree.left_child != None :
            ret += prt(tree.left_child,dep+1)

    return ret

def classify(row,tree):
    label = row[len(row)-1]
    branch = None

    if tree.result_set != None :
        predicted = tree.result_set.keys()[0]
        return predicted
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
                branch = tree.left_child
            else : branch = tree.right_child

        return classify(row,branch)

def accuracy(test_d,dec_tree):
    m = 0.0
    t_p = 0
    t_n = 0
    f_p = 0
    f_n = 0
    total = len(test_d)
    for row in test_d:
        l = row[len(row)-1]
        p = classify(row,dec_tree)
        #print " l and p " , l , p
        if l== '0' and p == '0' :
            t_n += 1
        elif l=='0' and p=='1':
            f_p += 1
        elif l=='1' and p=='0':
            f_n += 1
        elif l=='1' and p=='1':
            t_p += 1
        if l == p :
            m += 1

    #print  " blah " , t_p , t_n , f_p , f_n
    acc = m/(total*1.0)
    if t_p + f_p < 1 :
        prec = 0
    else :  prec = 1.0*t_p/((t_p+f_p)*1.0)

    if t_p + f_n < 1 :
        recall = 0
    else :
        recall = 1.0*t_p/((t_p+f_n)*1.0)

    if prec + recall == 0 :
        f_1 = 0

    else : f_1 = 2*(prec*recall*1.0)/(prec+recall)
    return acc , prec , recall , f_1



def accuracy_forest(test_d,forest_trees):
    m = 0.0
    total = len(test_d)
    t_p = 0
    t_n = 0
    f_p = 0
    f_n = 0
    for tr in test_d:
        l_dict = {}
        label_t = tr[len(tr)-1]

        for ft in forest_trees:
            p = classify(tr,ft)
            if l_dict.__contains__(p):
                l_dict[p] += 1
            else: l_dict[p] = 1
        l_dict = sorted(l_dict.items(),key=operator.itemgetter(1),reverse=True)

        pred_f = l_dict[0][0]
        if label_t=='0' and pred_f == '0' :
            t_n += 1
        elif label_t=='0' and pred_f=='1':
            f_p += 1
        elif label_t=='1' and pred_f=='0':
            f_n += 1
        elif label_t=='1' and pred_f=='1':
            t_p += 1
        if label_t == pred_f :
            m += 1

    acc =  m/(total*1.0)
    if t_p + f_p < 1:
        prec = 0
    else : prec = 1.0*t_p/((t_p+f_p)*1.0)

    if t_p + f_n < 1 :
        recall = 0
    else :
        recall = 1.0*t_p/((t_p+f_n)*1.0)

    if prec + recall == 0.0 :
        f_1 = 0
    else : f_1 = 2*(prec*recall*1.0)/(prec+recall)

    return acc, prec, recall , f_1

def format_data(filename):
    file = open(filename ,'r')
    lines = file.readlines()
    data = []
    for line in lines:
        line = line.replace("\n","")
        elements = line.split("\t")
        data.append(elements)
    return data


data_train = format_data('project3_dataset4.txt')
data_test = format_data('project3_dataset4_test.txt')


col_length = len(data_train[0])
total_rows = len(data_train)

#start random forest


def subset(data_p_x):
    data_p = deepcopy(data_p_x)
    ran = randint(1,len(data_p)/2-1)
    #print 'removing  rows .. ' , ran
    for n in range(ran):
        del data_p[0]

    shuffle(data_p)

    return data_p


def random_forest():
    n_trees = randint(len(data_train)/2,len(data_train)/2+3)
    if n_trees%2==0 or n_trees==1: n_trees += 1

    n_cols_remove = int(math.sqrt(col_length - 1) - 1)



    forest_trees = []
    for t in range(n_trees):
        d_x = deepcopy(data_train)
        col_r = randint(0,2)
        for r in d_x:
            del r[col_r]
        data_t = []
        #print 'sending .. ' , data_train[0] , len(data_train)
        data_t = subset(d_x)
        #print 'received .. ' , data_t[0] , len(data_t)
        f_tree = build_decision_tree(data_t)
        #print f_tree.value
        forest_trees.append(f_tree)
    return forest_trees


print len(data_train)
decision_tree = build_decision_tree(data_train)


f_ts = random_forest()

count = 0

for ft in f_ts :
    count += 1
    #print "\n====================\n"
    #print 'Forest Tree ' , count , "\n"
    #print_tree(ft)
    #prt(ft)


print "\n\n\n===================\n\n"


print_tree(decision_tree)


def k_cross_val(k,train_data):

    k_index = int(len(train_data)/k)
    window = [0,k_index]
    acc_total = 0.0
    prec_total = 0.0
    rec_total = 0.0
    f1_total = 0.0
    calls = 0
    breakAfterThis = False
    while not breakAfterThis:
        w_0 = window[0]
        w_1 = window[1]
        test_x = train_data[w_0:w_1]

        if w_1 + k_index > len(train_data)-1:
            train_x =  train_data[w_1+1:]
            breakAfterThis = True
        else :
            train_x_1 =  train_data[0:w_0-1]
            train_x_2 =  train_data[w_1+1:w_1+1+k_index]
            train_x = train_x_1.append(train_x_2)

        #d_k_tree = build_decision_tree(train_x)
        #acc , prec , rec , f1 = accuracy(test_x,d_k_tree)
        #print accuracy(test_x,d_k_tree)
        window[0] += k_index
        window[1] += k_index

        if(window[1]>=len(train_data)-1):
            breakAfterThis = True

        #print "Window - " , window

    #print 'completed'




k_cross_val(10, data_train)


print prt(decision_tree,4)
#print_tree(decision_tree)


print '\nDecision Tree Accuracy ' , accuracy(data_test,decision_tree)
print '\nForest Accuracy ' , accuracy_forest(data_test,f_ts)



#print 'D Tree Prediction  ' , classify(['overcast'	,'mild'	, 'high' ,	'strong',	1],decision_tree)
#print 'Forest Prediction ' , accuracy_forest([['overcast'	,'mild'	, 'high' ,	'strong',	1]],f_ts)