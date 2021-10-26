import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

task_sigma = 1
N = 80
C = 1.5
def calcLostFunc(y1 ,y2):
    res = []
    if len(y1) == len(y2):
        for i in range(len(y1)):
            res.append((y1[i]-y2[i])*(y1[i]-y2[i]))

    return sum(res)
def writeSetsTofile(x,y,name):
    if len(x) == len(y):
        new_file = open(name, "w")
        for i in range(len(y)):
            st = "" + str(x[i]) + " , " + str(y[i])+"\n"
            new_file.write(st)
        new_file.close()
def f_of_x_withOut_nocice(alpha,x):
    C = 1.5
    return C*np.exp(alpha*x)
def getTheNewEpslone(sigma):
    return np.random.normal(0,sigma)
def y_with_nocie(alpha,x,sigma):
    return f_of_x_withOut_nocice(alpha,x) + getTheNewEpslone(sigma)
def makeOneYs(x_set,y_func_with_nocice,alpha,sigma):
    y = []
    for x in x_set:
        y.append(y_func_with_nocice(alpha,x,sigma))

    return y
def makeOneYs_no_noice(x_set,y_func_no_nocice,alpha):
    y = []
    for x in x_set:
        y.append(y_func_no_nocice(alpha,x))

    return y

def calculateLogset(xs_set ,ys_set,sizeOfSet):
    for i in range(sizeOfSet):
        if ys_set[i][0] <= 0:
            ys_set[i]=None
            xs_set[i]=None

    xs_set = list(filter(lambda x: x != None, xs_set))
    ys_set = list(filter(lambda y: y != None, ys_set))
    ys_set = list(map(lambda y: np.log(y), ys_set))
    return xs_set,ys_set


def get_C_cova_And_a_cova__from_Log_trainingset(x_set,log_y_set):
    lr = LinearRegression()
    lr.fit(x_set, log_y_set)
    beta_0 = lr.intercept_[0]
    beta_1 = lr.coef_.tolist()[0][0]
    a_cova = beta_1
    c_cova = np.exp(beta_0)
    return a_cova , c_cova


def y_cova_func_for_d (x,c_cova,a_cova):
    return c_cova * np.exp(a_cova*x)

def makePredYs_for_task1D(x_set,c,a):
    y = []
    for x in x_set:
        y.append(y_cova_func_for_d(x,c,a))

    return y


lr = LinearRegression()
alphas = np.linspace(0.1,2.1,11)

All_XS_FOR_Log_Training_SET=[]
All_YS_FOR_Log_Training_SET=[]
All_YS_FOR_Log_Training_SET_PRED=[]

All_XS_FOR_Log_Test_SET=[]
All_YS_FOR_Log_Test_SET=[]
All_YS_FOR_Log_Test_SET_PRED=[]

Cs_cova_from_Log_set = []
as_cova_from_Log_set = []

Xs_Training_set = sorted(np.random.uniform(-4, 4, N).reshape(-1, 1))
Xs_Test_set = sorted(np.random.uniform(-4, 4, N).reshape(-1, 1))

for a in alphas:
    Xs = list(Xs_Training_set)
    Xs_t = list(Xs_Test_set)
    All_XS_FOR_Log_Training_SET.append(Xs)
    All_XS_FOR_Log_Test_SET.append(Xs_t)

for i in range(len(alphas)):
    All_YS_FOR_Log_Training_SET.append(makeOneYs(All_XS_FOR_Log_Training_SET[i],y_func_with_nocice=y_with_nocie ,alpha=alphas[i] ,sigma=task_sigma))
    All_YS_FOR_Log_Test_SET.append(makeOneYs(All_XS_FOR_Log_Test_SET[i],y_func_with_nocice=y_with_nocie ,alpha=alphas[i] ,sigma=task_sigma))

for i in range(len(alphas)):
    All_XS_FOR_Log_Training_SET[i],All_YS_FOR_Log_Training_SET[i] = calculateLogset(All_XS_FOR_Log_Training_SET[i],All_YS_FOR_Log_Training_SET[i],N)
    All_XS_FOR_Log_Test_SET[i],All_YS_FOR_Log_Test_SET[i] = calculateLogset(All_XS_FOR_Log_Test_SET[i],All_YS_FOR_Log_Test_SET[i],N)


for i in range(len(alphas)):
    a , c = get_C_cova_And_a_cova__from_Log_trainingset(All_XS_FOR_Log_Training_SET[i],All_YS_FOR_Log_Training_SET[i])
    Cs_cova_from_Log_set.append(c)
    as_cova_from_Log_set.append(a)
    All_YS_FOR_Log_Training_SET_PRED.append(makePredYs_for_task1D(All_XS_FOR_Log_Training_SET[i],c,a))
    All_YS_FOR_Log_Test_SET_PRED.append(makePredYs_for_task1D(All_XS_FOR_Log_Test_SET[i],c,a))

## --after this we have results and we can calc TSE & TE

task1_D_2_res_Table = []

for i in range(len(alphas)):
    size = len(All_YS_FOR_Log_Training_SET[i])
    tse = calcLostFunc(All_YS_FOR_Log_Training_SET[i],All_YS_FOR_Log_Training_SET_PRED[i])/size
    sizete = len(All_YS_FOR_Log_Test_SET[i])
    te = calcLostFunc(All_YS_FOR_Log_Test_SET[i],All_YS_FOR_Log_Test_SET_PRED[i])/sizete
    row = {"alpha : " : alphas[i] ,"TSE" : tse[0] ,"TE" :te[0]}
    task1_D_2_res_Table.append(row)


print(as_cova_from_Log_set)
print(Cs_cova_from_Log_set)

for row in task1_D_2_res_Table:
    print(row , "\n")
