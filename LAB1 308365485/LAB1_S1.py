import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

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
    return  C*np.exp(alpha*x)

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

Xs_training = sorted(np.random.uniform(-4, 4, N).reshape(-1, 1))
Xs_test = sorted(np.random.uniform(-4,4,N).reshape(-1,1))

alphas = np.linspace(0.1,2.1,11)

lr = LinearRegression()
all_Alphas_Ys_Training_to_Xs_trainingSet = []
all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice = []
all_Alphas_Ys_test_to_Xs_testSet = []
all_predicted_Ys_from_Xs_trainingSet = []
all_predicted_Ys_from_Xs_testSet = []

tableOfResults_q2_A_and_B = []


for a in alphas:
    all_Alphas_Ys_Training_to_Xs_trainingSet_withOut_noice.append(makeOneYs_no_noice(Xs_training, y_func_no_nocice=f_of_x_withOut_nocice, alpha=a))


for a in alphas:
    all_Alphas_Ys_Training_to_Xs_trainingSet.append(makeOneYs(Xs_training, y_func_with_nocice=y_with_nocie, alpha=a, sigma=task_sigma))
    all_Alphas_Ys_test_to_Xs_testSet.append(makeOneYs(Xs_test, y_func_with_nocice=y_with_nocie, alpha=a, sigma=task_sigma))


for i in range(len(alphas)):
    print("a = ",alphas[i])
    print(pd.DataFrame({"X": Xs_training , "Y" : all_Alphas_Ys_Training_to_Xs_trainingSet[i]}))